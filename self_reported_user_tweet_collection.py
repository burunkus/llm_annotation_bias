import re
import emoji
import string
import random
import csv
import os
import json
import datetime
import sys
import logging
import logging.handlers
from twarc import Twarc2, expansions
from collections import defaultdict


def get_filename():
    current_file_name = os.path.basename(__file__).split('.')[0]
    log_name = current_file_name
    return log_name


def get_logger(log_folder,log_filename):
    if os.path.exists(log_folder) == False:
        os.makedirs(log_folder)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]:  %(message)s",
        datefmt="%m-%d-%Y %H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(log_folder, log_filename+'.log'), mode='w'),
        logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

"""
Tutorial examples:
https://github.com/twitterdev/getting-started-with-the-twitter-api-v2-for-academic-research/blob/main/modules/6a-labs-code-academic-python.md

Tutorials:
https://twarc-project.readthedocs.io/en/latest/tutorials/
"""

# Your bearer token here
TOKEN = "Enter your token here"
client = Twarc2(bearer_token=TOKEN, metadata=False)


def get_tweets(users_path, save_as, next_user=None):
    
    users = []
    with open(users_path) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader, 1):
            user_id = line[0]
            user_race = line[1]
            users.append(user_id)
    
    users_count = len(users)
    with open(save_as, "a+") as file_handler:
        for i, user_id in enumerate(users, 1):
            logging.info(f'User ({user_id}) [{i}/{users_count}]')
            user_timeline = client.timeline(user_id, max_results=5, exclude_retweets=True)

            # Get all results page by page:
            for page in user_timeline:
                # The Twitter API v2 returns the Tweet information and the user, media etc. separately
                # so we use expansions.flatten to get all the information in a single JSON
                result = expansions.flatten(page)

                # Do something with the page of results:
                for tweet in result:
                    file_handler.write(f'{json.dumps(tweet)}\n')


def data_stats(data_path):
    
    with open(data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    logging.info(f'Number of tweets in {data_path}: {dataset_size}')
    
    user_stats = defaultdict(int)
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            author_id = line[1]
            tweet = line[2]
            user_stats[author_id] += 1
    logging.info(f'Number of users in {data_path}: {len(user_stats)}')
    logging.info(f'Users and tweet counts: {user_stats}')


def extract_tweets(data_path, save_as):
    
    with open(save_as, 'a') as new_file_handle:
        with open(data_path) as file_handle:
            for line in file_handle:
                tweet_object = json.loads(line)
                tweet = tweet_object['text'].strip().strip('\n')
                tweet = tweet.replace('\n', ' ')
                tweet = tweet.replace('\r', ' ')
                author_id = tweet_object['author_id']
                tweet_id = tweet_object['id']
                new_file_handle.write(tweet_id + '\t' + author_id + '\t' + tweet + '\n')
                    
    
def sample_from_black_users(data_path, save_as):
    
    user_ids = set()
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            author_id = line[1].strip()
            user_ids.add(author_id)
            
    # sample 14 from user ids
    random.seed(0)
    sampled_users = random.sample(user_ids, 14)
    sampled_users = set(sampled_users)
    
    # sample the tweets of the sampled 14 users
    with open(save_as, 'w') as new_file_handle:
        with open(data_path) as file_handle:
            for i, row in enumerate(file_handle):
                line = row.split('\t')
                tweet_id = line[0].strip()
                author_id = line[1].strip()
                tweet = line[2].strip()
                if author_id in sampled_users:
                    new_file_handle.write(tweet_id + '\t' + author_id + '\t' + tweet + '\n')


def preprocess(tweet):
    
    punct_chars = list((set(string.punctuation) | {
    "’", "‘", "–", "—", "~", "|", "“", "”", "…", "'", "`", "_",
    "“"
    }) - set(["#", "@"]))
    punct_chars.sort()
    punctuation = "".join(punct_chars)
    replace = re.compile("[%s]" % re.escape(punctuation))
    
    tweet = tweet.strip('"')
    tweet = tweet.lower()

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^rt[\s]+', '', tweet)
    # replace hyperlinks with URL
    tweet = re.sub(r'(https?:\/\/[a-zA-Z0-9]+\.[^\s]{2,})', 'HTTPURL', tweet)
    # remove hashtags - only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # remove punctuations
    tweet = replace.sub(" ", tweet)
    # replace emojis with emoji text
    tweet = emoji.demojize(tweet, delimiters=("", ""))
    # replace numbers with NUMBER
    tweet = re.sub(r'^\d+$', 'NUMBER', tweet)
    # replace handles with @USER
    tweet = re.sub(r'@\w+', '@USER', tweet)
    # replace all whitespace with a single space
    tweet = re.sub(r"\s+", " ", tweet)
    # strip off spaces on either end
    tweet = tweet.strip()
    return tweet


def merge_black_and_white_users_tweets(white_data_path, black_data_path, save_as, k=5000):
    
    random.seed(0)
    with open(white_data_path, 'r') as file_handle:
        white_data = file_handle.readlines()
    sampled_white_data = random.sample(white_data, k)
    
    with open(black_data_path, 'r') as file_handle:
        black_data = file_handle.readlines()
    sampled_black_data = random.sample(black_data, k)
    
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, row in enumerate(sampled_white_data):
            line = row.split('\t')
            tweet_id = line[0].strip()
            author_id = line[1].strip()
            tweet = line[2].strip()
            tweet = preprocess(tweet)
            label = '0'
            csv_handler.writerow([tweet_id, author_id, tweet, label])
                
        for i, row in enumerate(sampled_black_data):
            line = row.split('\t')
            tweet_id = line[0].strip()
            author_id = line[1].strip()
            tweet = line[2].strip()
            tweet = preprocess(tweet)
            label = '1'
            csv_handler.writerow([tweet_id, author_id, tweet, label])
            
            
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    custom_name = 'merge_black_and_white_users'
    _ = get_logger(log_dir, get_filename() + f'_{custom_name}')
    
    # users_path = "self_reported_user_data/white_users.csv"
    # save_as = "self_reported_user_data/white_users.jsonl"
    # get_tweets(users_path, save_as)
    
    # extract black users tweets
    # data_path = "self_reported_user_data/aa_users.jsonl"
    # save_as = "self_reported_user_data/aa_users.txt"
    # extract_tweets(data_path, save_as)
    
    # stats of black users
    # data_stats(save_as)
    
    # extract white users tweets 
    # data_path = "self_reported_user_data/white_users.jsonl"
    # save_as = "self_reported_user_data/white_users.txt"
    # extract_tweets(data_path, save_as)
    
    # stats of white users 
    # data_stats(save_as)
    
    # Since there are only 14 white users, only data of 14 black users
    # data_path = "self_reported_user_data/aa_users.txt"
    # save_as = "self_reported_user_data/aa_users_sampled.txt"
    # sample_from_black_users(data_path, save_as)
    # data_stats(save_as) 
    
    # Sample K tweets from white and black user, merge white and black users and assign them a label, preprocess each tweet. 
    # white_data_path = "self_reported_user_data/white_users.txt"
    # black_data_path = "self_reported_user_data/aa_users_sampled.txt"
    # save_as = "self_reported_user_data/self_reported_user_tweets_preprocessed.csv"
    # number_of_tweets_to_sample = 5000
    # merge_black_and_white_users_tweets(white_data_path, black_data_path, save_as, k=number_of_tweets_to_sample)
