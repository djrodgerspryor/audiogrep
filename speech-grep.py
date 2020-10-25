#!/usr/bin/env python3

import click
import os
import mimetypes
import glob
import xxhash
import sys

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

"""Generous handling of cli paths, globs, directories etc. to product a list
of audio files to work with"""
def select_input_files(inputs, ignore_type):
    candidate_files = set()
    for input_item in set(inputs):
        if os.path.isdir(input_item):
            print(input_item, "glob: {}".format(os.path.join(input_item, '**/*')))
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
def cached_hash_file_content(normalised_path):
    if normalised_path not in FILE_CONTENT_HASH_CACHE:
        with open(normalised_path, 'rb') as f:
            FILE_CONTENT_HASH_CACHE[normalised_path] = xxhash.xxh32(f.read()).digest()

    return FILE_CONTENT_HASH_CACHE[normalised_path]

def get_existing_transcript(input_file, transcript_query_hash):
    content_hash = cached_hash_file_content(input_file)

    # Combine the file content hash, with the query hash to get a hash that we can use to cache
    # transcripts with proper invalidation characteristics
    expected_transcript_hash_hex = xxhash.xxh32(content_hash + transcript_query_hash).hexdigest()

    # Look for any existing transcript files with this hash in the filename (the prefix can be anything — usually the original file name)
    input_directory = os.path.dirname(input_file)
    return next(iter(
        glob.glob(
            os.path.join(
                input_directory,
                "*-transcript-{}.json".format(expected_transcript_hash_hex)
            )
        )
    ), None)

def ensure_transcripts(input_files):
    transcript_paths = {}

    # This might take a little time to read and hash all of the audio content, so
    # display a progress bar
    with click.progressbar(input_files, label="Checking for existing transcripts", file=sys.stderr) as input_files_to_check:
        for input_file in input_files_to_check:
            maybe_path = get_existing_transcript(input_file, b'TODO')

            if maybe_path is not None:
                transcript_paths[input_file] = maybe_path

    to_transcode = [input_file for input_file in input_files if input_file not in transcript_paths]

    print(to_transcode)

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
def grep(query, inputs, mode, ignore_type, padding, context):
    input_files = select_input_files(inputs, ignore_type)

    ensure_transcripts(input_files)



if __name__ == '__main__':
    grep()
