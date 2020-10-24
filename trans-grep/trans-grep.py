#!/usr/bin/env python3

import click
import json
from intervaltree import IntervalTree
import math
import regex
import collections

SearchResult = collections.namedtuple("SearchResult", ["start_time", "end_time", "text"])

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

        return SearchResult(
            self.find_timestamp(match_start_index),
            self.find_timestamp(match_end_index),
            self.transcript_string.slice(match_start_index, match_end_index)
        )

    def regex_findall(self, query):
        return [
            SearchResult(
                self.find_timestamp(match.start()),
                self.find_timestamp(match.end()),
                match.group(0) # The whole match
            )
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

def pad_result(search_result, padding):
    return SearchResult(
        search_result.start_time - padding,
        search_result.end_time + padding,
        search_result.text
    )


@click.command()
@click.argument('query')
@click.argument('input', type=click.File('r'))
@click.option('--simple', '-s', default=False, is_flag=True)
@click.option('--padding', '-p', type=float, default=0.05)
def grep(query, input, simple, padding):
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

    padded_results = [pad_result(s, padding) for s in results]

    click.echo(click.style("Start\tEnd\tText", bold=True), err=True)
    for result in padded_results:
        print("{:.2f}\t{:.2f}\t{}".format(result.start_time, result.end_time, result.text))

if __name__ == '__main__':
    grep()
