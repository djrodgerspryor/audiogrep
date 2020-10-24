#!/usr/bin/env python3

import click
import json
from intervaltree import IntervalTree
import math
import regex
import collections

TextSlice = collections.namedtuple("TextSlice", ["start_index", "end_index"])
TimeSlice = collections.namedtuple("TimeSlice", ["start_time", "end_time"])

class SearchResult:
    def __init__(self, parent_transcript, text_slice, override_time_slice=None):
        self.parent_transcript = parent_transcript
        self.text_slice = text_slice

        if override_time_slice is None:
            self.time_slice = TimeSlice(
                self.parent_transcript.find_timestamp(self.text_slice.start_index),
                self.parent_transcript.find_timestamp(self.text_slice.end_index),
            )
        else:
            self.time_slice = override_time_slice

        # The whole match string
        self.text = self.parent_transcript.transcript_string[self.text_slice.start_index:self.text_slice.end_index]

    def pad(self, time_padding_seconds):
        return SearchResult(
            self.parent_transcript,
            self.text_slice,
            TimeSlice(
                max(0.0, self.time_slice.start_time - time_padding_seconds),
                self.time_slice.end_time + time_padding_seconds,
            )
        )

    def text_with_context(self, context_word_count):
        if context_word_count <= 0:
            return ("", self.text, "")

        # Take the first n words from before and after the match in the transcript
        before_context = " ".join(self.parent_transcript.transcript_string[:self.text_slice.start_index].split()[-context_word_count:])
        after_context = " ".join(self.parent_transcript.transcript_string[self.text_slice.end_index:].split()[:context_word_count])

        # Return them all as a tuple (with padding so that they can all just be concated together)
        return (before_context + " ", self.text, " " + after_context)

class CandidateTranscription:
    def __init__(self, transcription_items):
        self.tree = IntervalTree()

        chunks = []
        total_length_so_far = 0
        total_log_probability = 0
        for item in transcription_items:
            # Ignore punctuation
            if item['type'] == 'punctuation':
                continue

            if item['type'] != 'pronunciation':
                raise "Unknown item type: {}".format(item['type'])

            alternative = item['alternatives'][0]
            chunk = alternative['content']
            start_time = float(item['start_time'])
            end_time = float(item['end_time'])

            # Account for the space which will be added
            if total_length_so_far != 0:
                # Insert an interval to account for the space
                self.tree.addi(
                    total_length_so_far,
                    total_length_so_far + 1,
                    {
                        'start_time': self.tree[self.tree.end() - 1].pop().data['end_time'],
                        'end_time': start_time,
                    }
                )
                total_length_so_far += 1

            beginning_chunk_index = total_length_so_far
            total_length_so_far += len(chunk)

            # Add this interval to the tree, mapping the character index interval to the time interval
            self.tree.addi(
                beginning_chunk_index,
                total_length_so_far,
                {
                    'start_time': start_time,
                    'end_time': end_time,
                }
            )

            # Record the chunk so that we can build up the transcript as a single string
            chunks.append(chunk)

            # Add up how likely this whole candidate transcript is
            confidence = float(alternative['confidence'])
            if confidence > 0:
                total_log_probability += math.log(confidence)

        # Build the transcript string for text searching
        self.transcript_string = " ".join(chunks)

        # This is a convenient format for working with small probabilities.
        self.negative_log_probability = -total_log_probability

    # Runs a simple .find() query against the transcript and return a 2-tuple describing the timestamp slice
    # of the audio file
    def simple_find(self, query):
        # Strip common punctuation from the query, since we don't expect it in the transcript
        normalised_query = (
            query
                .lower()
                .replace('.', '')
                .replace(',', '')
                .replace(';', '')
                .replace('(', '')
                .replace(')', '')
                .replace('|', '')
                .replace(';', '')
                .replace('-', '')
                .replace('–', '')
                .replace('—', '')
                .replace("'", '')
                .replace('"', '')
                .replace('/', '')
                .replace('\\', '')
        )

        match_start_index = self.transcript_string.lower().find(normalised_query)

        # If the query wasn't found, return None
        if match_start_index < 0:
            return None

        match_end_index = match_start_index + len(normalised_query)

        return SearchResult(self, TextSlice(match_start_index, match_end_index))

    def regex_findall(self, query):
        return [
            SearchResult(self, TextSlice(match.start(), match.end()))
            for match
            in regex.finditer(query, self.transcript_string, flags=regex.IGNORECASE | regex.BESTMATCH)
        ]

    # Look up an index in the transcript string and turn it into an audio time-stamp
    def find_timestamp(self, transcript_character_index):
        # If this is the last character in the tree
        if transcript_character_index >= self.tree.end():
            # Special case handling: pick the last interval and interpolate to the end
            interval = self.tree[self.tree.end() - 1].pop()
            interpolation_fraction = 1.0
        else:
            # Lookup the matching interval
            interval = self.tree[transcript_character_index].pop()

            # Work out how far through this chunk the index is
            interpolation_fraction = float(transcript_character_index - interval.begin) / float(interval.end - interval.begin)

        # Turn that fraction into a timestamp
        return interval.data['start_time'] + (interpolation_fraction * (interval.data['end_time'] - interval.data['start_time']))

@click.command()
@click.argument('query')
@click.argument('input', type=click.File('r'))
@click.option('--simple', '-s', default=False, is_flag=True)
@click.option('--padding', '-p', type=float, default=0.05, help="Padding in seconds to apply around match timestamps")
@click.option('--context', '-c', type=int, default=0, help="Number of words to show either side of the match")
def grep(query, input, simple, padding, context):
    transcription_data = json.loads(input.read())

    # The items field contains the maximum likelihood transcription value
    maximum_liklihood_transcription = CandidateTranscription(transcription_data['results']['items'])

    results = []
    if simple:
        result = maximum_liklihood_transcription.simple_find(query)

        if result is None:
            click.echo(click.style("No match found", fg='red'), err=True)
            exit(1)

        results.append(result)

    else:
        for result in maximum_liklihood_transcription.regex_findall(query):
            results.append(result)

    click.echo(click.style("{} matches found!".format(len(results)), fg='green'), err=True)

    padded_results = [s.pad(padding) for s in results]

    # Render the heading row
    click.echo(
        "\t".join([
            click.style(heading, underline=True, dim=True)
            for heading
            in ("Start", "End", "Text")
        ]),
        err=True
    )

    # And then the actual matches, line-by-line
    for result in padded_results:
        before_context, text, after_context = result.text_with_context(context)
        print("{:.2f}\t{:.2f}\t{}{}{}".format(
            result.time_slice.start_time,
            result.time_slice.end_time,
            before_context,
            click.style(text, fg="red", bold=True),
            after_context
        ))

if __name__ == '__main__':
    grep()
