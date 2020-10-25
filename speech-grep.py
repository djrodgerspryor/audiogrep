#!/usr/bin/env python3

import click
import os
import mimetypes
import glob
import xxhash
import sys
import boto3
from botocore.errorfactory import ClientError as BotoClientError
import re
import collections
import json
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math
import signal

TranscriptionConfig = collections.namedtuple("TranscriptionConfig", ["provider", "s3_bucket", "input_prefix", "output_prefix"])

SAVED_CONFIG_LOCATION = os.path.expanduser('~/.speech-grep')

# Enough to process big directories quickly, but without completely tying-up the 100 job limit
# in AWS Transcribe
MAX_PARALLEL_JOBS = 20

SUPPORTED_FORMATS = set([
    'audio/mpeg',
    'audio/mp4a-latm',
    'audio/mp4',
    'audio/mp4a-latm',
    'audio/ogg',
    'audio/webm',
    'audio/x-flac',
    'audio/x-wav',
])

class TranscriptionOptions:
    # TODO: maybe support vocabularies etc.
    def __init__(self, language_code=None, max_alternatives=0):
        self.language_code = language_code
        self.max_alternatives = max_alternatives

    def digest(self):
        # Hash all of the options in a consistent order
        return xxhash.xxh32(
            "{language_code:}{max_alternatives:}".format(
                language_code=self.language_code,
                max_alternatives=self.max_alternatives
            )
        ).digest()

"""Generous handling of cli paths, globs, directories etc. to product a list
of audio files to work with"""
def select_input_files(inputs, ignore_type):
    candidate_files = set()
    for input_item in set(inputs):
        if os.path.isdir(input_item):
            # Recursively search directories
            candidate_files.update(
                [
                    os.path.abspath(path)
                    for path in
                    glob.glob(
                        os.path.join(input_item, '**/*'),
                        recursive=True
                    )
                ]
            )

        elif os.path.isfile(input_item):
            # Add files directly
            candidate_files.add(
                os.path.abspath(input_item)
            )

        else:
            # Interpret the input as a glob
            candidate_files.update(
                glob.glob(input_item, recursive=True)
            )

    if ignore_type:
        input_files = candidate_file
    else:
        input_files = set()
        for candidate_file in candidate_files:
            mime_type, mime_encoding = mimetypes.guess_type(candidate_file)

            if mime_type.lower() in SUPPORTED_FORMATS:
                input_files.add(candidate_file)

    return input_files

FILE_CONTENT_HASH_CACHE = {}
"""Returns a hash of file content. Cached against the given file name."""
def cached_hash_file_content(normalised_path):
    if normalised_path not in FILE_CONTENT_HASH_CACHE:
        with open(normalised_path, 'rb') as f:
            FILE_CONTENT_HASH_CACHE[normalised_path] = xxhash.xxh32(f.read()).digest()

    return FILE_CONTENT_HASH_CACHE[normalised_path]

"""Get the path for an existing transcript file if there is one for this file/query combo"""
def get_existing_transcript(input_file, transcript_query_hash):
    # Look for any existing transcript files with this hash in the filename (the prefix can be anything — usually the original file name)
    input_directory = os.path.dirname(input_file)
    return next(iter(
        glob.glob(
            os.path.join(
                input_directory,
                "*-{}.transcript.json".format(transcript_hash(input_file, transcript_query_hash))
            )
        )
    ), None)

def transcript_hash(input_file, transcript_query_hash):
    content_hash = cached_hash_file_content(input_file)

    # Combine the file content hash, with the query hash to get a hash that we can use to cache
    # transcripts with proper invalidation characteristics
    return xxhash.xxh32(content_hash + transcript_query_hash).hexdigest()

def transcript_name(input_file, transcript_query_hash):
    # Get the input filename, without extension
    filename = os.path.splitext(os.path.basename(input_file))[0]

    # Combine the original filename (so that the transcript will sort next to the original file
    # and so that it's relationship will be obvious) with the digest which ensures that the transcript
    # is unique for this kind of query and this file content.
    return "{}-{}.transcript.json".format(filename, transcript_hash(input_file, transcript_query_hash))

"""Get transcript files for all of the inputs, either from a local cache or by generating them"""
def ensure_transcripts(input_files, transcription_options, transcription_config):
    transcript_path_by_input_path = {}

    options_digest = transcription_options.digest()

    # This might take a little time to read and hash all of the audio content, so
    # display a progress bar
    with click.progressbar(input_files, label="Checking for existing transcripts", file=sys.stderr) as input_files_to_check:
        for input_file in input_files_to_check:
            maybe_path = get_existing_transcript(input_file, options_digest)

            if maybe_path is not None:
                transcript_path_by_input_path[input_file] = maybe_path

    to_transcode = [input_file for input_file in input_files if input_file not in transcript_path_by_input_path]

    for input_file, transcript_file in zip(to_transcode, batch_transcribe(to_transcode, transcription_options, transcription_config)):
        transcript_path_by_input_path[input_file] = transcript_file

    return transcript_path_by_input_path

def batch_transcribe(input_files, transcription_options, transcription_config):
    click.echo("Preparing to transcribe {} files".format(len(input_files)), err=True)

    # Parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as pool:
        sorted_input_files = sorted(list(input_files))

        progress_description_width = 20
        progress_bars_by_input_file = {}

        try:
            for i, input_file in enumerate(sorted_input_files):
                progress_bars_by_input_file[input_file] = tqdm(
                    position=i, # This creates different bars which won't overwrite each other
                    leave=False, # Remove progress bars once they're all done
                    total=100,
                    unit="%",
                    desc="Queued".rjust(progress_description_width, ' '),
                    postfix=os.path.basename(input_file)
                )

            def update_progress(input_file, state, progress_fraction):
                # Exit this thread if there's a global exit in progress. This is a convenient
                # place for cooperative scheduling because it will be called by all threads
                # frequently
                if IS_EXITING:
                    raise SystemExit

                bar = progress_bars_by_input_file[input_file]
                new_progress_value = max(0, min(100, math.floor(progress_fraction * 100.0)))
                bar.update(new_progress_value - bar.n) # Pass in a diff value
                bar.set_description(state.rjust(progress_description_width, ' '))

            results = pool.map(
                lambda input_file: transcribe(
                    input_file,
                    transcription_options,
                    transcription_config,
                    progress_callback=lambda state, progress: update_progress(input_file, state, progress)
                ),
                sorted_input_files
            )

            # Wait for execution to finish
            results = list(results)

        finally:
            # Clean-up all progress bars once we're done
            for bar in progress_bars_by_input_file.values():
                bar.close()

    return results


class ExistingS3ObjectInvalid(Exception):
    pass

def transcribe(input_file, transcription_options, transcription_config, progress_callback=lambda state, progress: None):
    aws_s3_client = boto3.client('s3')
    aws_transcribe_client = boto3.client('transcribe')

    progress_callback("Checking S3", 0.00)

    file_size = os.path.getsize(input_file)

    # Use the content hash to identify the file in S3 to prevent unnecessary multiple uploads
    upload_key = os.path.join(transcription_config.input_prefix, cached_hash_file_content(input_file).hex())

    # Ensure that the audio file is uploaded to S3
    try:
        existing_object = aws_s3_client.head_object(Bucket=transcription_config.s3_bucket, Key=upload_key)

        # In addition to the content hash in the key, this should protect against reading the wrong
        # file (due to incomplete/partial uploads in the case of this check)
        if file_size != existing_object['ContentLength']:
            raise ExistingS3ObjectInvalid("Mismatched content length: {} (expected: {})".format(existing_object['ContentLength'], file_size))

    # Object doesn't already exist
    except (ExistingS3ObjectInvalid, BotoClientError) as e:
        progress_callback("Uploading", 0.00)

        uploaded_bytes_so_far = 0
        def upload_progress_callback(chunk_size):
            nonlocal uploaded_bytes_so_far
            uploaded_bytes_so_far += chunk_size
            progress_callback("Uploading", float(uploaded_bytes_so_far) / float(file_size))

        # Upload the source file
        aws_s3_client.upload_file(
            Filename=input_file,
            Bucket=transcription_config.s3_bucket,
            Key=upload_key,
            Callback=upload_progress_callback
        )

    progress_callback("Preparing", 0.0)

    # Ensure that there is a transcription job for this file and query
    job_name = transcript_hash(input_file, transcription_options.digest())
    transcript_file_name = transcript_name(input_file, transcription_options.digest())
    try:
        transcribe_job = aws_transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )

        if transcribe_job['TranscriptionJob']['TranscriptionJobStatus'] == 'FAILED':
            # Delete the old failed job so that we can re-use the name
            aws_transcribe_client.delete_transcription_job(
                TranscriptionJobName=job_name
            )
            raise "Previous job failed"

    # We expect this unless a transcription job was already running or completed
    except:
        job_kwargs = {
            "TranscriptionJobName": job_name,
            "Media": {
                "MediaFileUri":  "s3://" + os.path.join(transcription_config.s3_bucket, upload_key),
            },
            "OutputBucketName": transcription_config.s3_bucket,
            "OutputKey": os.path.join(transcription_config.output_prefix, transcript_file_name),

            # Disabled because this requires configuring IAM roles which is a bit of a PITA
            # "JobExecutionSettings": {
            #     "AllowDeferredExecution": True,
            # }
        }

        # Fill out the Settings field if we need to
        transcribe_settings = {}
        if transcription_options.max_alternatives > 0:
            transcribe_settings["ShowAlternatives"] = True
            transcribe_settings["MaxAlternatives"] = transcription_options.max_alternatives
        if len(transcribe_settings) > 0:
            job_kwargs["Settings"] = transcribe_settings

        # Either auto-detect language or specify it explicitly
        if transcription_options.language_code is None:
            job_kwargs["IdentifyLanguage"] = True
        else:
            job_kwargs["LanguageCode"] = transcription_options.language_code

        transcribe_job = aws_transcribe_client.start_transcription_job(**job_kwargs)

    progress_callback("Transcoding Queued", 0.00)

    wait_counter = 0
    while transcribe_job['TranscriptionJob']['TranscriptionJobStatus'] in ['QUEUED', 'IN_PROGRESS']:
        time.sleep(1)

        transcribe_job = aws_transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )

        if transcribe_job['TranscriptionJob']['TranscriptionJobStatus'] == 'QUEUED':
            progress_callback("Transcoding Queued", progress)
        else:
            # Make a fake progress counter which asymptotically approaches complete
            wait_counter += 1
            progress = max(1.0 - (10.0 / wait_counter), 0.05)

            if transcribe_job['TranscriptionJob']['TranscriptionJobStatus'] == 'IN_PROGRESS':
                progress_callback("Transcoding", progress)
            elif transcribe_job['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                progress_callback("Transcoding", 1.0)

    if transcribe_job['TranscriptionJob']['TranscriptionJobStatus'] != 'COMPLETED':
        progress_callback("ERROR", 1.0)
        click.echo("Transcription job failed!\n{}".format(json.dumps(transcribe_job, sort_keys=True, indent=4)), err=True)
        raise "Transcription error"

    progress_callback("Downloading", 0.0)
    local_transcript_path = os.path.join(os.path.dirname(input_file), transcript_file_name)

    # Parse the given HTTP S3 URI into the correct format for boto to use
    result_s3_bucket, *result_s3_key_components = urlparse(transcribe_job['TranscriptionJob']['Transcript']['TranscriptFileUri']).path.strip('/').split('/')
    aws_s3_client.download_file(
        Bucket=result_s3_bucket,
        Key="/".join(result_s3_key_components),
        Filename=local_transcript_path
    )

    progress_callback("Done", 1.0)

    return local_transcript_path

CACHED_CONFIG = None
def get_saved_config():
    global CACHED_CONFIG

    if CACHED_CONFIG is None:
        try:
            with open(SAVED_CONFIG_LOCATION, 'r') as f:
                CACHED_CONFIG = json.loads(f.read())
        except FileNotFoundError:
            CACHED_CONFIG = {}

    return CACHED_CONFIG

def upsert_config(new_config):
    config = get_saved_config()
    config.update(new_config)

    with open(SAVED_CONFIG_LOCATION, 'w+') as f:
        f.write(json.dumps(config, sort_keys=True, indent=4))

    click.echo("Config saved to ~/.speech-grep", err=True)

@click.command(help="""
        Search audio transcripts with a grep-like interface. Currently only supports transcripts in the format produced by AWS Transcribe.

        QUERY — Search term. Regular expressions are supported (unless --mode is set to simple).
        Punctuation is stripped from the transcript before matching, so do not include it
        in your query. Fuzzy expressions are supported eg. (?:some text){e<3} will match the
        phrase 'some text' with fewer than 3 edits. See https://pypi.org/project/regex/ for
        full documentation of supported regex operations.

        INPUTS — A set of paths to search
    """
)
@click.argument('query')
@click.argument('inputs', nargs=-1)
@click.option(
    '--mode',
    '-m',
    type=click.Choice(
        ['re', 'regexp', 'regex', 'simple'],
        case_sensitive=False
    ),
    default="regex",
    help="""
        Search mode. re regexp and regex are equivalent.
        Simple mode just finds the first exact match (ignoring
        punctuation). Both modes ignore case. regex is the default.
    """
)
@click.option('--ignore-type', is_flag=True, help="By default inputs which don't look like audio files will be ignored. With this enabled, all files will be processed.")
@click.option('--padding', '-p', type=float, default=0.05, help="Padding in seconds to apply around match timestamps")
@click.option('--context', '-c', type=int, default=0, help="Number of words to show either side of the match")
@click.option('--s3-bucket', type=str, default=lambda: os.environ.get('S3_BUCKET', get_saved_config().get('default_s3_bucket', '')), help="S3 bucket to use for AWS transcription")
@click.option('--input-prefix', type=str, default="transcribe/inputs/", help="S3 prefix to use for uploaded audio files")
@click.option('--output-prefix', type=str, default="transcribe/outputs/", help="S3 prefix to use for generated transcript files")
def grep(query, inputs, mode, ignore_type, padding, context, s3_bucket, input_prefix, output_prefix):
    input_files = select_input_files(inputs, ignore_type)

    # Allow callers to specify a working S3 bucket interactively
    if s3_bucket == '':
        s3_bucket = click.prompt('S3 Bucket not specified. Which S3 bucket would you like to use for transcoding?', type=str)
        if click.confirm('Would you like to save this S3 bucket as the default for future use?'):
            upsert_config({ "default_s3_bucket": s3_bucket })

    transcript_path_by_input_path = ensure_transcripts(input_files, TranscriptionOptions(), TranscriptionConfig("aws_transcribe", s3_bucket, input_prefix, output_prefix))

IS_EXITING = False
def keyboard_interrupt_handler(signal, frame):
    global IS_EXITING
    IS_EXITING = True
    click.echo('Exiting', err=True)
    sys.exit(1)

if __name__ == '__main__':
    # Ensure proper exit when ctrl-c is pressed, even if we're in a thread
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    grep()
