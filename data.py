"""Functions for reading data from the sentiment dictionary and tweet files."""

import os
import re
import string
import sys
from datetime import datetime
from ucb import main, interact
from TwitterSearch import *

# Look for data directory
PY_PATH = sys.argv[0]
if PY_PATH.endswith('doctest.py') and len(sys.argv) > 1:
    PY_PATH = sys.argv[1]
DATA_PATH = os.path.join(os.path.dirname(PY_PATH), 'data') + os.sep
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data' + os.sep

def load_sentiments(path=DATA_PATH + "sentiments.csv"):
    """Read the sentiment file and return a dictionary containing the sentiment
    score of each word, a value from -1 to +1.
    """
    with open(path, encoding='utf8') as sentiment_file:
        scores = [line.split(',') for line in sentiment_file]
        return {word: float(score.strip()) for word, score in scores}

word_sentiments = load_sentiments()

def file_name_for_term(term, unfiltered_name):
    """Return a valid file name for an arbitrary term string."""
    valid_characters = '-_' + string.ascii_letters + string.digits
    no_space_lower = term.lower().replace(' ', '_')
    valid_only = [c for c in no_space_lower if c in valid_characters]
    return ''.join(valid_only) + '_' +  unfiltered_name

def generate_filtered_file(unfiltered_name, term):
    """Return the path to a file containing tweets that match term, generating
    that file if necessary.
    """
    filtered_path = DATA_PATH + file_name_for_term(term, unfiltered_name)
    if not os.path.exists(filtered_path):
        msg = 'Generating filtered tweets file for "{0}" using tweets from {1}'
        print(msg.format(term, unfiltered_name))
        r = re.compile('\W' + term + '\W', flags=re.IGNORECASE)
        with open(filtered_path, mode='w', encoding='utf8') as out:
            with open(DATA_PATH + unfiltered_name, encoding='utf8') as full:
                matches = [line for line in full if term in line.lower()]
            for line in matches:
                if r.search(line):
                    out.write(line)
            print('Wrote {}.'.format(filtered_path))
    return filtered_path

def load_tweets(term='cali'):
    """Return the list of tweets returned by a twitter search with the term.

    """
    # filtered_path = generate_filtered_file(file_name, term)
    # with open(filtered_path, encoding='utf8') as tweets:
    #     return [tweet_from_line(line, make_tweet) for line in tweets
    #             if len(line.strip().split("\t")) >= 4]
    try:
        tso = TwitterSearchOrder()
        tso.add_keyword(term)

        ts = TwitterSearch(
            consumer_key = '',
            consumer_secret = '',
            access_token = '',
            access_token_secret = ''
            )

        tweetList = ts.search_tweets_iterable(tso)

        return [t for t in tweetList if t['coordinates'] is not None]

    except TwitterSearchException as e:
        print(e)

