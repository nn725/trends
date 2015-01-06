"""Visualizing Twitter Sentiment Across America"""

import sys
from data import word_sentiments, load_tweets
from datetime import datetime
from geo import us_states, geo_distance, make_position, longitude, latitude
try:
    import tkinter
    from maps import draw_state, draw_name, draw_dot, wait
    HAS_TKINTER = True
except ImportError as e:
    print('Could not load tkinter: ' + str(e))
    HAS_TKINTER = False
from string import ascii_letters


###################################
# Phase 1: The Feelings in Tweets #
###################################

# tweet data abstraction (A), represented as a list
# -------------------------------------------------

def tweet_text(tweet):
    """Return a string, the words in the text of a tweet."""
    return tweet['text']

def tweet_time(tweet):
    """Return the datetime representing when a tweet was posted."""
    return tweet['created_at']

def tweet_location(tweet):
    """Return a position representing a tweet's location."""
    return make_position(tweet['coordinates']['coordinates'][0],tweet['coordinates']['coordinates'][1])

### === +++ ABSTRACTION BARRIER +++ === ###

def tweet_string(tweet):
    """Return a string representing a tweet."""
    location = tweet_location(tweet)
    point = (latitude(location), longitude(location))
    return '"{0}" @ {1}'.format(tweet_text(tweet), point)

def tweet_words(tweet):
    """Return the words in a tweet."""
    return extract_words(tweet_text(tweet))

def extract_words(text):
    """Return the words in a tweet, not including punctuation.

    >>> extract_words('anything else.....not my job')
    ['anything', 'else', 'not', 'my', 'job']
    >>> extract_words('i love my job. #winning')
    ['i', 'love', 'my', 'job', 'winning']
    >>> extract_words('make justin # 1 by tweeting #vma #justinbieber :)')
    ['make', 'justin', 'by', 'tweeting', 'vma', 'justinbieber']
    >>> extract_words("paperclips! they're so awesome, cool, & useful!")
    ['paperclips', 'they', 're', 'so', 'awesome', 'cool', 'useful']
    >>> extract_words('@(cat$.on^#$my&@keyboard***@#*')
    ['cat', 'on', 'my', 'keyboard']
    """
    #get list of characters
    char_list = list(text)

    result = []

    #current_word keeps track of the current word, is added to result at first non-letter
    current_word = ''
    for i in range(0,len(char_list)):
        if char_list[i] in ascii_letters:
            current_word = current_word + char_list[i]
        elif current_word != '':
            result = result + [current_word]
            current_word = ''
    if current_word != '':
        result = result + [current_word]
    return result

def make_sentiment(value):
    """Return a sentiment, which represents a value that may not exist.

    >>> positive = make_sentiment(0.2)
    >>> neutral = make_sentiment(0)
    >>> unknown = make_sentiment(None)
    >>> has_sentiment(positive)
    True
    >>> has_sentiment(neutral)
    True
    >>> has_sentiment(unknown)
    False
    >>> sentiment_value(positive)
    0.2
    >>> sentiment_value(neutral)
    0
    """
    assert (value is None) or (-1 <= value <= 1), 'Bad sentiment value'
    return [value]

def has_sentiment(s):
    """Return whether sentiment s has a value."""
    return s[0] is not None

def sentiment_value(s):
    """Return the value of a sentiment s."""
    assert has_sentiment(s), 'No sentiment value'
    return s[0]

def get_word_sentiment(word):
    """Return a sentiment representing the degree of positive or negative
    feeling in the given word.

    >>> sentiment_value(get_word_sentiment('good'))
    0.875
    >>> sentiment_value(get_word_sentiment('bad'))
    -0.625
    >>> sentiment_value(get_word_sentiment('winning'))
    0.5
    >>> has_sentiment(get_word_sentiment('Berkeley'))
    False
    """
    # Learn more: http://docs.python.org/3/library/stdtypes.html#dict.get
    return make_sentiment(word_sentiments.get(word))

def analyze_tweet_sentiment(tweet):
    """Return a sentiment representing the degree of positive or negative
    feeling in the given tweet, averaging over all the words in the tweet
    that have a sentiment value.

    If no words in the tweet have a sentiment value, return
    make_sentiment(None).

    >>> positive = make_tweet('i love my job. #winning', None, 0, 0)
    >>> round(sentiment_value(analyze_tweet_sentiment(positive)), 5)
    0.29167
    >>> negative = make_tweet("saying, 'i hate my job'", None, 0, 0)
    >>> sentiment_value(analyze_tweet_sentiment(negative))
    -0.25
    >>> no_sentiment = make_tweet('berkeley golden bears!', None, 0, 0)
    >>> has_sentiment(analyze_tweet_sentiment(no_sentiment))
    False
    """
    total_sentiment = 0
    num_sentiments = 0

    #get text of tweet
    text = tweet_text(tweet)

    #check each word in text, add to total_sentiment if there is a sentiment
    for word in extract_words(text):
        if has_sentiment(get_word_sentiment(word)):
            total_sentiment += sentiment_value(get_word_sentiment(word))
            num_sentiments += 1

    #if there are no sentimental words, return None
    if num_sentiments == 0:
        return make_sentiment(None)

    return make_sentiment(total_sentiment/num_sentiments)


#################################
# Phase 2: The Geometry of Maps #
#################################

def apply_to_all(map_fn, s):
    return [map_fn(x) for x in s]

def keep_if(filter_fn, s):
    return [x for x in s if filter_fn(x)]

def find_centroid(polygon):
    """Find the centroid of a polygon. If a polygon has 0 area, use the latitude
    and longitude of its first position as its centroid.

    http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon

    Arguments:
    polygon -- A list of positions, in which the first and last are the same

    Returns 3 numbers: centroid latitude, centroid longitude, and polygon area.

    >>> p1 = make_position(1, 2)
    >>> p2 = make_position(3, 4)
    >>> p3 = make_position(5, 0)
    >>> triangle = [p1, p2, p3, p1] # First vertex is also the last vertex
    >>> round_all = lambda s: [round(x, 5) for x in s]
    >>> round_all(find_centroid(triangle))
    [3.0, 2.0, 6.0]
    >>> round_all(find_centroid([p1, p3, p2, p1])) # reversed
    [3.0, 2.0, 6.0]
    >>> apply_to_all(float, find_centroid([p1, p2, p1])) # A zero-area polygon
    [1.0, 2.0, 0.0]
    """
    #find the area
    area = 0
    for i in range(len(polygon) - 1):
        area += latitude(polygon[i])*longitude(polygon[i+1]) - latitude(polygon[i+1])*longitude(polygon[i])
    area *= 0.5

    #check if area is 0 before we divide
    if area != 0:
        factor = 1 / (6 * area)
    #if area is 0, return the latitude and longitude of first point
    else:
        return [latitude(polygon[0]), longitude(polygon[0]), 0]

    #find C_x
    c_x = 0
    for i in range(len(polygon) - 1):
        c_x += (latitude(polygon[i]) + latitude(polygon[i+1])) * (latitude(polygon[i])*longitude(polygon[i+1]) - latitude(polygon[i+1])*longitude(polygon[i]))
    c_x *= factor

    #find C_y
    c_y = 0
    for i in range(len(polygon) - 1):
        c_y += (longitude(polygon[i]) + longitude(polygon[i+1])) * (latitude(polygon[i])*longitude(polygon[i+1]) - latitude(polygon[i+1])*longitude(polygon[i]))
    c_y *= factor

    #return the positive area
    return [c_x, c_y, abs(area)]

def find_state_center(polygons):
    """Compute the geographic center of a state, averaged over its polygons.

    The center is the average position of centroids of the polygons in
    polygons, weighted by the area of those polygons.

    Arguments:
    polygons -- a list of polygons

    >>> ca = find_state_center(us_states['CA'])  # California
    >>> round(latitude(ca), 5)
    37.25389
    >>> round(longitude(ca), 5)
    -119.61439

    >>> hi = find_state_center(us_states['HI'])  # Hawaii
    >>> round(latitude(hi), 5)
    20.1489
    >>> round(longitude(hi), 5)
    -156.21763
    """

    #find the centroids of all polygons in argument
    centroids = []
    for polygon in polygons:
        centroids = centroids + [find_centroid(polygon)]

    #find total area
    total_area = 0
    for i in range(len(polygons)):
        total_area += centroids[i][2]

    #find total weighted C_x
    total_weighted_c_x = 0
    for i in range(len(polygons)):
        total_weighted_c_x += centroids[i][0] * centroids[i][2]

    #find total weighted C_y
    total_weighted_c_y = 0
    for i in range(len(polygons)):
        total_weighted_c_y += centroids[i][1] * centroids[i][2]

    #return averages
    return make_position(total_weighted_c_x / total_area, total_weighted_c_y / total_area)


###################################
# Phase 3: The Mood of the Nation #
###################################

def group_by_key(pairs):
    """Return a dictionary that relates each unique key in [key, value] pairs
    to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_key(example)
    {1: [2, 3, 2], 2: [4], 3: [2, 1]}
    """
    # Optional: This implementation is slow because it traverses the list of
    #           pairs one time for each key. Can you improve it?
    keys = [key for key, _ in pairs]
    return {key: [y for x, y in pairs if x == key] for key in keys}

def group_tweets_by_state(tweets):
    """Return a dictionary that groups tweets by their nearest state center.

    The keys of the returned dictionary are state names and the values are
    lists of tweets that appear closer to that state center than any other.

    Arguments:
    tweets -- a sequence of tweet abstract data types

    >>> sf = make_tweet("welcome to san francisco", None, 38, -122)
    >>> ny = make_tweet("welcome to new york", None, 41, -74)
    >>> two_tweets_by_state = group_tweets_by_state([sf, ny])
    >>> len(two_tweets_by_state)
    2
    >>> california_tweets = two_tweets_by_state['CA']
    >>> len(california_tweets)
    1
    >>> tweet_string(california_tweets[0])
    '"welcome to san francisco" @ (38, -122)'
    """

    #creates a list of pairs [state, state center]
    state_center_pairs = []
    for state in us_states:
        state_center_pairs = state_center_pairs + [[state, find_state_center(us_states[state])]]

    state_tweet_list = []

    #for each tweet, find the state with minimum distance as a pair [minimum distance, tweet]
    for tweet in tweets:
        min_distance = [state_center_pairs[0][0], geo_distance(state_center_pairs[0][1],tweet_location(tweet))] 
        
        #for each state, check if distance to state center is smaller than the current minimum
        for s in state_center_pairs:
            d = geo_distance(s[1],tweet_location(tweet))
            if min_distance[1] > d:
                min_distance = [s[0], d]
        state_tweet_list = state_tweet_list + [[min_distance[0], tweet]]

    #return the dictionary that groups the tweets by state
    return group_by_key(state_tweet_list)

def average_sentiments(tweets_by_state):
    """Calculate the average sentiment of the states by averaging over all
    the tweets from each state. Return the result as a dictionary from state
    names to average sentiment values (numbers).

    If a state has no tweets with sentiment values, leave it out of the
    dictionary entirely. Do NOT include states with no tweets, or with tweets
    that have no sentiment, as 0. 0 represents neutral sentiment, not unknown
    sentiment.

    Arguments:
    tweets_by_state -- A dictionary from state names to lists of tweets
    """

    #returns a number that is the average of the sentiments of the tweets
    def average_tweets(tweets):
        total = 0
        i = 0
        sentiments = False
        for tweet in tweets:
            if has_sentiment(analyze_tweet_sentiment(tweet)):
                sentiments = True
                total += sentiment_value(analyze_tweet_sentiment(tweet))
                i += 1
        return total/i if sentiments else None

    final_dictionary = {}
    
    #for each state, if the the states have a known sentiment, put it in the dictionary
    for state in tweets_by_state:
        x = average_tweets(tweets_by_state[state])
        if x is not None:
            final_dictionary[state] = x
    return final_dictionary

##########################
# Command Line Interface #
##########################

def uses_tkinter(func):
    """A decorator that designates a function as one that uses tkinter.
    If tkinter is not supported, will not allow these functions to run.
    """
    def tkinter_checked(*args, **kwargs):
        if HAS_TKINTER:
            return func(*args, **kwargs)
        print('tkinter not supported, cannot call {0}'.format(func.__name__))
    return tkinter_checked

def print_sentiment(text='Are you virtuous or verminous?'):
    """Print the words in text, annotated by their sentiment scores."""
    words = extract_words(text.lower())
    layout = '{0:>' + str(len(max(words, key=len))) + '}: {1:+}'
    for word in words:
        s = get_word_sentiment(word)
        if has_sentiment(s):
            print(layout.format(word, sentiment_value(s)))

@uses_tkinter
def draw_centered_map(center_state='TX', n=10):
    """Draw the n states closest to center_state."""
    centers = {name: find_state_center(us_states[name]) for name in us_states}
    center = centers[center_state.upper()]
    distance = lambda name: geo_distance(center, centers[name])
    for name in sorted(centers, key=distance)[:int(n)]:
        draw_state(us_states[name])
        draw_name(name, centers[name])
    draw_dot(center, 1, 10)  # Mark the center state with a red dot
    wait()

@uses_tkinter
def draw_state_sentiments(state_sentiments):
    """Draw all U.S. states in colors corresponding to their sentiment value.

    Unknown state names are ignored; states without values are colored grey.

    Arguments:
    state_sentiments -- A dictionary from state strings to sentiment values
    """
    for name, shapes in us_states.items():
        draw_state(shapes, state_sentiments.get(name))
    for name, shapes in us_states.items():
        center = find_state_center(shapes)
        if center is not None:
            draw_name(name, center)

@uses_tkinter
def draw_map_for_query(term='my job'):
    """Draw the sentiment map corresponding to the tweets that contain term.

    Some term suggestions:
    New York, Texas, sandwich, my life, justinbieber
    """
    tweets = load_tweets(term)
    tweets_by_state = group_tweets_by_state(tweets)
    state_sentiments = average_sentiments(tweets_by_state)
    draw_state_sentiments(state_sentiments)
    for tweet in tweets:
        s = analyze_tweet_sentiment(tweet)
        if has_sentiment(s):
            draw_dot(tweet_location(tweet), sentiment_value(s))
    wait()


def run(*args):
    """Read command-line arguments and calls corresponding functions."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Trends")
    parser.add_argument('--print_sentiment', '-p', action='store_true')
    parser.add_argument('--draw_centered_map', '-d', action='store_true')
    parser.add_argument('--draw_map_for_query', '-m', type=str)
    parser.add_argument('text', metavar='T', type=str, nargs='*',
                        help='Text to process')
    args = parser.parse_args()
    if args.draw_map_for_query:
        draw_map_for_query(args.draw_map_for_query)
        return
    for name, execute in args.__dict__.items():
        if name != 'text' and name != 'tweets_file' and execute:
            globals()[name](' '.join(args.text))

if __name__ == '__main__': run(sys.argv[1:])
