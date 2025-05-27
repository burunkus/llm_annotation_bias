import time
import random
import json
import csv
import os
import sys
import re
import emoji
import string
import logging
import logging.handlers
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split
from pprint import pprint

LABEL_MAP = {
        "waseem": {1: "Racism", 2: "Sexism", 3: "Racism and Sexism", 0: "Neither"},
        "waseem_and_hovy": {1: "Racism", 2: "Sexism", 0: "None"},
        "founta": {1: "Hateful", 2: "Abusive", 0: "Normal"},
        "davidson": {1: "Hate", 2: "Offensive", 0: "Normal"},
        "golbeck": {1: "Harassment", 0: "Non-harassment"},
        "hateval": {1: "Hate", 0: "Non-hate"},
        "offenseval": {1: "Offensive", 0: "Non-offensive"},
        "abuseval": {1: "Explicit abuse", 2: "Implicit abuse", 0: "Not abusive"}
    }


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


def data_stats(data_path, dataset_name, with_gpt_label=False):
    
    label_map = {
        "waseem": {1: "Racism", 2: "Sexism", 3: "Racism and Sexism", 0: "Neither"},
        "waseem_and_hovy": {1: "Racism", 2: "Sexism", 0: "None"},
        "founta": {1: "Hateful", 2: "Abusive", 0: "Normal"},
        "davidson": {1: "Hate", 2: "Offensive", 0: "Normal"},
        "golbeck": {1: "Harassment", 0: "Non-harassment"},
        "hateval": {1: "Hate", 0: "Non-hate"},
        "offenseval": {1: "Offensive", 0: "Non-offensive"},
        "abuseval": {1: "Explicit abuse", 2: "Implicit abuse", 0: "Not abusive"}
    }
    
    with open(data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    logging.info(f'------ Processing {dataset_name} -------')
    logging.info(f'Number of tweets: {dataset_size}')
    
    counts = defaultdict(int)
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            if with_gpt_label:
                gpt_label = int(line[3].strip())
                counts[gpt_label] += 1
            else:
                human_label = int(line[2].strip())
                counts[human_label] += 1
    
    for key, value in counts.items():
        logging.info(f'{label_map[dataset_name][key]}: {value}')
    logging.info(f'--------------------------------------')


def data_stats_binary(data_path, dataset_name, with_gpt_label=False):
    
    label_map = {
        "waseem": {1: "Hate", 0: "Neither"},
        "waseem_and_hovy": {1: "Hate", 0: "None"},
        "founta": {1: "Hate", 0: "Normal"},
        "davidson": {1: "Hate", 0: "Normal"},
        "golbeck": {1: "Harassment", 0: "Non-harassment"},
        "hateval": {1: "Hate", 0: "Non-hate"},
        "offenseval": {1: "Offensive", 0: "Non-offensive"},
        "abuseval": {1: "Abusive", 0: "Not abusive"}
    }
    
    dialect_list_str = {1: 'aae', 0: 'sae'}
    
    with open(data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    
    total = 0
    all_data = []
    with open(data_path, 'r') as file_handle:
        for line in file_handle:
            all_data.append(line)
            total += 1
    
    logging.info(f'------ Processing {dataset_name} -------')
    logging.info(f'Number of tweets: {dataset_size}, {total}')
    
    counts = defaultdict(lambda: defaultdict(int))
    counts2 = {}
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            if with_gpt_label:
                gpt_label_binary = int(line[4].strip())
                dialect_label = int(line[5].strip())
                counts[label_map[dataset_name][gpt_label_binary]][dialect_list_str[dialect_label]] += 1
                counts[label_map[dataset_name][gpt_label_binary]]['count'] += 1
                
                curr_class = label_map[dataset_name][gpt_label_binary]
                if curr_class in counts2:
                    curr_dialect = dialect_list_str[dialect_label]
                    if curr_dialect in counts2[curr_class]:
                        counts2[curr_class][curr_dialect] += 1
                    else:
                        counts2[curr_class][curr_dialect] = 1
                        
                    if 'count' in counts2[curr_class]:
                        counts2[curr_class]['count'] += 1
                    else:
                        counts2[curr_class]['count'] = 1
                else:
                    counts2[curr_class] = {}
                    curr_dialect = dialect_list_str[dialect_label]
                    counts2[curr_class][curr_dialect] = 1
                    counts2[curr_class]['count'] = 1
                    
            else:
                human_label_binary = int(line[3].strip())
                dialect_label = int(line[4].strip())
                counts[label_map[dataset_name][human_label_binary]][dialect_list_str[dialect_label]] += 1
                counts[label_map[dataset_name][human_label_binary]]['count'] += 1
                
                curr_class = label_map[dataset_name][human_label_binary]
                if curr_class in counts2:
                    curr_dialect = dialect_list_str[dialect_label]
                    if curr_dialect in counts2[curr_class]:
                        counts2[curr_class][curr_dialect] += 1
                    else:
                        counts2[curr_class][curr_dialect] = 1
                        
                    if 'count' in counts2[curr_class]:
                        counts2[curr_class]['count'] += 1
                    else:
                        counts2[curr_class]['count'] = 1
                else:
                    counts2[curr_class] = {}
                    curr_dialect = dialect_list_str[dialect_label]
                    counts2[curr_class][curr_dialect] = 1
                    counts2[curr_class]['count'] = 1
    
    pprint(counts)
    pprint(counts2)
    logging.info(f'--------------------------------------')
    
    
def sample_from_waseem_orginal_data(original_data_root_path, data_path, save_path):
    
    data = []
    neither_class = []
    
    with open(original_data_root_path + data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
     
            if label == 0:
                neither_class.append(row)
            else:
                data.append(row)
    
    # Sample 500 tweets from only the neither class
    random.seed(23)
    sampled_neither_class = random.sample(neither_class, 500)
    data += sampled_neither_class
    
    absolute_path, file_name = os.path.split(save_path + data_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    # Write data to file
    with open(save_path + data_path, 'w') as save_file_handle:
        for item in data:
            save_file_handle.write(item)


def sample_from_waseem_and_hovy_orginal_data(original_data_root_path, data_path, save_path):
    
    data = []
    neither_class = []
    sexism_class = []
    
    with open(original_data_root_path + data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
     
            if label == 0:
                neither_class.append(row)
            elif label == 2:
                sexism_class.append(row)
            else:
                data.append(row)
    
    # Sample 500 tweets from the neither class
    random.seed(23)
    sampled_neither_class = random.sample(neither_class, 500)
    data += sampled_neither_class
    
    # Sample 500 tweets from the sexism class
    sampled_sexism_class = random.sample(sexism_class, 500)
    data += sampled_sexism_class
    
    absolute_path, file_name = os.path.split(save_path + data_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    # Write data to file
    with open(save_path + data_path, 'w') as save_file_handle:
        for item in data:
            save_file_handle.write(item)     
            

def sample_from_founta_orginal_data(original_data_root_path, data_path, save_path):
    
    abusive_class = []
    hateful_class = []
    normal_class = []
    
    with open(original_data_root_path + data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
     
            if label == 0:
                normal_class.append(row)
            elif label == 1:
                hateful_class.append(row)
            else:
                abusive_class.append(row)
    
    data = []
    # Sample 500 tweets from the normal class
    random.seed(23)
    sampled_normal_class = random.sample(normal_class, 500)
    data += sampled_normal_class
    
    # Sample 500 tweets from the hateful class
    sampled_hateful_class = random.sample(hateful_class, 500)
    data += sampled_hateful_class
    
    # Sample 500 tweets from the abusive class
    sampled_abusive_class = random.sample(abusive_class, 500)
    data += sampled_abusive_class
    
    absolute_path, file_name = os.path.split(save_path + data_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    # Write data to file
    with open(save_path + data_path, 'w') as save_file_handle:
        for item in data:
            save_file_handle.write(item)    
            

def sample_from_davidson_orginal_data(original_data_root_path, data_path, save_path):
    
    offensive_class = []
    hateful_class = []
    normal_class = []
    
    with open(original_data_root_path + data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
     
            if label == 0:
                normal_class.append(row)
            elif label == 1:
                hateful_class.append(row)
            else:
                offensive_class.append(row)
    
    data = []
    # Sample 500 tweets from the normal class
    random.seed(23)
    sampled_normal_class = random.sample(normal_class, 500)
    data += sampled_normal_class
    
    # Sample 500 tweets from the hateful class
    sampled_hateful_class = random.sample(hateful_class, 500)
    data += sampled_hateful_class
    
    # Sample 500 tweets from the abusive class
    sampled_offensive_class = random.sample(offensive_class, 500)
    data += sampled_offensive_class
    
    absolute_path, file_name = os.path.split(save_path + data_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    # Write data to file
    with open(save_path + data_path, 'w') as save_file_handle:
        for item in data:
            save_file_handle.write(item) 
            

def sample_from_golbeck_orginal_data(original_data_root_path, data_path, save_path):
    
    harassment_class = []
    normal_class = []
    
    with open(original_data_root_path + data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
     
            if label == 0:
                normal_class.append(row)
            elif label == 1:
                harassment_class.append(row)
    
    data = []
    # Sample 500 tweets from the normal class
    random.seed(23)
    sampled_normal_class = random.sample(normal_class, 500)
    data += sampled_normal_class
    
    # Sample 500 tweets from the hateful class
    sampled_harassment_class = random.sample(harassment_class, 500)
    data += sampled_harassment_class
    
    absolute_path, file_name = os.path.split(save_path + data_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    # Write data to file
    with open(save_path + data_path, 'w') as save_file_handle:
        for item in data:
            save_file_handle.write(item)
            

def sample_from_hateval_orginal_data(original_data_root_path, data_path, save_path):
    
    hateful_class = []
    normal_class = []
    
    with open(original_data_root_path + data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
     
            if label == 0:
                normal_class.append(row)
            elif label == 1:
                hateful_class.append(row)
    
    data = []
    # Sample 500 tweets from the normal class
    random.seed(23)
    sampled_normal_class = random.sample(normal_class, 500)
    data += sampled_normal_class
    
    # Sample 500 tweets from the hateful class
    sampled_hateful_class = random.sample(hateful_class, 500)
    data += sampled_hateful_class
    
    absolute_path, file_name = os.path.split(save_path + data_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    # Write data to file
    with open(save_path + data_path, 'w') as save_file_handle:
        for item in data:
            save_file_handle.write(item)
            

def sample_from_offenseval_orginal_data(original_data_root_path, data_path, save_path):
    
    offensive_class = []
    normal_class = []
    
    with open(original_data_root_path + data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
     
            if label == 0:
                normal_class.append(row)
            elif label == 1:
                offensive_class.append(row)
    
    data = []
    # Sample 500 tweets from the normal class
    random.seed(23)
    sampled_normal_class = random.sample(normal_class, 500)
    data += sampled_normal_class
    
    # Sample 500 tweets from the hateful class
    sampled_offensive_class = random.sample(offensive_class, 500)
    data += sampled_offensive_class
    
    absolute_path, file_name = os.path.split(save_path + data_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    # Write data to file
    with open(save_path + data_path, 'w') as save_file_handle:
        for item in data:
            save_file_handle.write(item)


def sample_from_abuseval_orginal_data(original_data_root_path, data_path, save_path):
    
    implicit_class = []
    explicit_class = []
    normal_class = []
    
    with open(original_data_root_path + data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
     
            if label == 0:
                normal_class.append(row)
            elif label == 1:
                explicit_class.append(row)
            else:
                implicit_class.append(row)
    
    data = []
    # Sample 500 tweets from the normal class
    random.seed(23)
    sampled_normal_class = random.sample(normal_class, 500)
    data += sampled_normal_class
    
    # Sample 500 tweets from the explicit class
    sampled_explicit_class = random.sample(explicit_class, 500)
    data += sampled_explicit_class
    
    # Sample 500 tweets from the implicit class
    sampled_implicit_class = random.sample(implicit_class, 500)
    data += sampled_implicit_class
    
    absolute_path, file_name = os.path.split(save_path + data_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    # Write data to file
    with open(save_path + data_path, 'w') as save_file_handle:
        for item in data:
            save_file_handle.write(item)
            

def sample_exemplars_from_data(data_path, save_path):
    
    data = defaultdict(list)
    
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            label = int(line[2].strip())
            
            # Add each class as a key and tweets belonging to the class as a value represented as an element in a list
            data[label].append(row)
    
    sampled_exemplars = []
    # Sample 2 tweets from each class
    random.seed(3)
    for label, rows in data.items():
        sampled_label_exemplars = random.sample(rows, 2)
        sampled_exemplars += sampled_label_exemplars
    
    absolute_path, file_name = os.path.split(save_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
       
    new_file_name = file_name.split('.')[0] + '_exemplars_2_samples.txt'
    new_save_path = absolute_path + '/' + new_file_name
    # Write exemplars to file
    with open(new_save_path, 'w') as save_file_handle:
        for item in sampled_exemplars:
            save_file_handle.write(item)
            
            
def get_tweet_dialect(inferred_dialect_data_path):
    '''
    Dialect can be predicted using the fine-tuned AAEBERT dialect classifier. However,
    the inference has been done in our previous work so just extract the dialect of the relevant
    tweets needed in this work. 
    Args:
        inferred_dialect_data_path (String): location of the file that contain tweets and their inferred dialect
    '''
    
    tweet_id_and_dialect_map = defaultdict(str)
    with open(inferred_dialect_data_path) as file_handle:
        for row in file_handle:
            data = row.split('\t')
            uid = data[0]
            tweet = data[1]
            label = data[2]
            dialect_label = data[3]
            tweet_id_and_dialect_map[uid] = dialect_label
      
    return tweet_id_and_dialect_map
    
    
def preprocess(file_path, save_as, dataset=None, with_gpt_label=True):
    """
    Proprocess tweets by lower casing, normalize by converting
    user mentions to @USER, url to HTTPURL, and number to NUMBER. 
    Convert emoji to text string and remove duplicate tweets
    Args:
        file_path(String): location of file to preprocess
        save_as(String): name to save the preproced tweet in.
    Return:
        None
    """
    
    punct_chars = list((set(string.punctuation) | {
    "’", "‘", "–", "—", "~", "|", "“", "”", "…", "'", "`", "_",
    "“"
    }) - set(["#", "@"]))
    punct_chars.sort()
    punctuation = "".join(punct_chars)
    replace = re.compile("[%s]" % re.escape(punctuation))
    
    with open(save_as, 'w') as new_file:
        with open(file_path) as old_file:
            for i, line in enumerate(old_file, 1):
                line = line.split("\t")
                tweet_id = line[0].strip()
                tweet = line[1].strip()
                human_label = line[2].strip()
                # Convert other negative classses - offensive, hate, abusive, racism, sexism, racism and sexism to hate
                binary_human_label = None
                if human_label == '0':
                    # Non-hate
                    binary_human_label = '0'
                else:
                    # Hate
                    binary_human_label = '1'
                    
                if with_gpt_label:
                    binary_gpt_label = None
                    gpt_label = line[3].strip()
                    if gpt_label == '0':
                        binary_gpt_label = '0'
                    else:
                        binary_gpt_label = '1'
                
                tweet = tweet.strip('"')
                tweet = tweet.lower()
                
                # remove stock market tickers like $GE
                tweet = re.sub(r'\$\w*', '', tweet)
                # remove old style retweet text "RT"
                tweet = re.sub(r'^rt[\s]+', '', tweet)
                # replace hyperlinks with URL
                if dataset == 'golbeck':
                    tweet = re.sub(r'(https?:\s?\/\s?\/[a-zA-Z0-9]*\.?[a-zA-Z0-9]+\s?\/?[a-zA-Z0-9]*|https?:\\\/\\\/[a-zA-Z0-9]*\.?[a-zA-Z0-9]+\s?\\\/?[a-zA-Z0-9]*|https?:?\s?\/?\s?\/?)', 'HTTPURL', tweet)
                elif dataset == 'offenseval2019':
                    tweet = re.sub(r'(url)', 'HTTPURL', tweet)
                else:
                    tweet = re.sub(r'(https?:\/\/[a-zA-Z0-9]+\.[^\s]{2,})', 'HTTPURL', tweet)
                # remove hashtags - only removing the hash # sign from the word
                tweet = re.sub(r'#', '', tweet)
                # replace emojis with emoji text
                tweet = emoji.demojize(tweet, delimiters=("", ""))
                # replace numbers with NUMBER
                tweet = re.sub(r'^\d+$', 'NUMBER', tweet)
                # replace handles with @USER
                tweet = re.sub(r'@\w+', '@USER', tweet)
                # remove punctuations
                tweet = replace.sub(" ", tweet)
                # replace all whitespace with a single space
                tweet = re.sub(r"\s+", " ", tweet)
                # strip off spaces on either end
                tweet = tweet.strip()
                if with_gpt_label:
                    new_file.write(f"{tweet_id}\t{tweet}\t{human_label}\t{gpt_label}\t{binary_gpt_label}\n")
                else:
                    new_file.write(f"{tweet_id}\t{tweet}\t{human_label}\t{binary_human_label}\n")


def preprocess_with_dialect(file_path, save_as, dataset=None, with_gpt_label=True):
    """
    Proprocess tweets by lower casing, normalize by converting
    user mentions to @USER, url to HTTPURL, and number to NUMBER. 
    Convert emoji to text string and remove duplicate tweets
    Args:
        file_path(String): location of file to preprocess
        save_as(String): name to save the preproced tweet in.
    Return:
        None
    """
    
    punct_chars = list((set(string.punctuation) | {
    "’", "‘", "–", "—", "~", "|", "“", "”", "…", "'", "`", "_",
    "“"
    }) - set(["#", "@"]))
    punct_chars.sort()
    punctuation = "".join(punct_chars)
    replace = re.compile("[%s]" % re.escape(punctuation))
    
    with open(save_as, 'w') as new_file:
        with open(file_path) as old_file:
            for i, line in enumerate(old_file, 1):
                line = line.split("\t")
                tweet_id = line[0].strip()
                tweet = line[1].strip()
                human_label = line[2].strip()
                # Convert other negative classses - offensive, hate, abusive, racism, sexism, racism and sexism to hate
                binary_human_label = None
                if human_label == '0':
                    # Non-hate
                    binary_human_label = '0'
                else:
                    # Hate
                    binary_human_label = '1'
                    
                if with_gpt_label:
                    binary_gpt_label = None
                    gpt_label = line[3].strip()
                    if gpt_label == '0':
                        binary_gpt_label = '0'
                    else:
                        binary_gpt_label = '1'
                
                tweet = tweet.strip('"')
                tweet = tweet.lower()
                
                # remove stock market tickers like $GE
                tweet = re.sub(r'\$\w*', '', tweet)
                # remove old style retweet text "RT"
                tweet = re.sub(r'^rt[\s]+', '', tweet)
                # replace hyperlinks with URL
                if dataset == 'golbeck':
                    tweet = re.sub(r'(https?:\s?\/\s?\/[a-zA-Z0-9]*\.?[a-zA-Z0-9]+\s?\/?[a-zA-Z0-9]*|https?:\\\/\\\/[a-zA-Z0-9]*\.?[a-zA-Z0-9]+\s?\\\/?[a-zA-Z0-9]*|https?:?\s?\/?\s?\/?)', 'HTTPURL', tweet)
                elif dataset == 'offenseval2019':
                    tweet = re.sub(r'(url)', 'HTTPURL', tweet)
                else:
                    tweet = re.sub(r'(https?:\/\/[a-zA-Z0-9]+\.[^\s]{2,})', 'HTTPURL', tweet)
                # remove hashtags - only removing the hash # sign from the word
                tweet = re.sub(r'#', '', tweet)
                # replace emojis with emoji text
                tweet = emoji.demojize(tweet, delimiters=("", ""))
                # replace numbers with NUMBER
                tweet = re.sub(r'^\d+$', 'NUMBER', tweet)
                # replace handles with @USER
                tweet = re.sub(r'@\w+', '@USER', tweet)
                # remove punctuations
                tweet = replace.sub(" ", tweet)
                # replace all whitespace with a single space
                tweet = re.sub(r"\s+", " ", tweet)
                # strip off spaces on either end
                tweet = tweet.strip()
                
                dialect_label = line[4].strip()
                if with_gpt_label:
                    new_file.write(f"{tweet_id}\t{tweet}\t{human_label}\t{gpt_label}\t{binary_gpt_label}\t{dialect_label}\n")
                else:
                    new_file.write(f"{tweet_id}\t{tweet}\t{human_label}\t{binary_human_label}\t{dialect_label}\n")
                    
                    
def split_dataset(data_path, save_train_as, save_test_as, train_size=0.80):
    """
    Splits the sampled dataset into train and test set using stratified sampling for datasets with high
    class imbalance. This function assumes there is a file containing text and label in each line separated by tab.
    Files created will be of the form: id \t tweet \t human label
    Args:
        data_path (String): path where the full dataset test set is contained
        save_train_as (String): absolute path to the training set
        save_test_as (String): absolute path to the test set
        train_size(Float): The percentage to divide the training set by, the test
        set will be 100 - train_split * 100. Defaults to 0.8 (80%). 
    Returns:
        None
    """
    
    data, labels = [], []
    seed = 42
    
    with open(data_path) as file_handler:
        for i, line in enumerate(file_handler):
            line_array = line.strip().split("\t")
            tweet_id = line_array[0]
            tweet = line_array[1]
            label = line_array[2]
            data.append((tweet_id, tweet))
            labels.append(label)
    
    if "waseem" in data_path or "waseem_and_hovy" in data_path:
        # Stratify train and test set split to manage class imbalance 
        train_data, test_data, train_label, test_label = train_test_split(
                data, labels, train_size=train_size, random_state=seed, stratify=labels
            )
    else:
        # Create train and test set split
        train_data, test_data, train_label, test_label = train_test_split(
                data, labels, train_size=train_size, random_state=seed
            )
    
    # Write train set
    with open(save_train_as, "w") as file_handler:
        for i, (tweet_id, tweet) in enumerate(train_data):
            label = train_label[i]
            file_handler.write(tweet_id + '\t' + tweet + '\t' + label + '\n')

    # Write test set
    with open(save_test_as, "w") as file_handler:
        for i, (tweet_id, tweet) in enumerate(test_data):
            label = test_label[i]
            file_handler.write(tweet_id + '\t' + tweet + '\t' + label + '\n')
            
                    
def split_dataset_with_gpt_label(data_path, save_train_as, save_test_as, train_size=0.80):
    """
    Splits a dataset into train and test set. This function assumes there is
    a file containing text and label in each line separated by tab
    Files created will be of the form: id \t tweet \t human-label \t gpt-label
    Args:
        data_path (String): path where the full dataset test set is contained
        save_train_as (String): absolute path to the training set
        save_test_as (String): absolute path to the test set
        train_size(Float): The percentage to divide the training set by, the test
        set will be 100 - train_split * 100. Defaults to 0.8 (80%). 
    Returns:
        None
    """
    
    data, labels = [], []
    seed = 42
    
    with open(data_path) as file_handler:
        for i, line in enumerate(file_handler):
            line_array = line.strip().split("\t")
            tweet_id = line_array[0]
            tweet = line_array[1]
            label = line_array[2]
            gpt_label = line_array[3]
            data.append((tweet_id, tweet, label))
            labels.append(gpt_label)
    
    if "waseem" in data_path or "waseem_and_hovy" in data_path:
        # Stratify train and test set split to manage class imbalance 
        train_data, test_data, train_label, test_label = train_test_split(
                data, labels, train_size=train_size, random_state=seed, stratify=labels
            )
    else:
        # Create train and test set split
        train_data, test_data, train_label, test_label = train_test_split(
                data, labels, train_size=train_size, random_state=seed
            )
    
    # Write train set
    with open(save_train_as, "w") as file_handler:
        for i, (tweet_id, tweet, human_label) in enumerate(train_data):
            gpt_label = train_label[i]
            file_handler.write(tweet_id + '\t' + tweet + '\t' + human_label + '\t' + gpt_label + '\n')

    # Write test set
    with open(save_test_as, "w") as file_handler:
        for i, (tweet_id, tweet, human_label) in enumerate(test_data):
            gpt_label = test_label[i]
            file_handler.write(tweet_id + '\t' + tweet + '\t' + human_label + '\t' + gpt_label + '\n')


def compare_human_vs_gpt_labels(data_path, dataset_name):
    label_map = {
        "waseem": {1: "Racism", 2: "Sexism", 3: "Racism and Sexism", 0: "Neither"},
        "waseem_and_hovy": {1: "Racism", 2: "Sexism", 0: "None"},
        "founta": {1: "Hateful", 2: "Abusive", 0: "Normal"},
        "davidson": {1: "Hate", 2: "Offensive", 0: "Normal"},
        "golbeck": {1: "Harassment", 0: "Non-harassment"},
        "hateval": {1: "Hate", 0: "Non-hate"},
        "offenseval": {1: "Offensive", 0: "Non-offensive"},
        "abuseval": {1: "Explicit abuse", 2: "Implicit abuse", 0: "Not abusive"}
    }

    logging.info(f'------ Processing {dataset_name} -------')
    
    counts = defaultdict(lambda: defaultdict(int))
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            
            if human_label != gpt_label:
                str_human_label = label_map[dataset_name][human_label]
                str_gpt_label = label_map[dataset_name][gpt_label]
                counts[str_human_label][str_gpt_label] += 1
    
    for true_class, gpt_data in counts.items():
        for gpt_class, count in gpt_data.items():
            logging.info(f'{true_class} ---> {gpt_class}: {count}')
    logging.info(f'--------------------------------------')


def count_dialect_labels(path):
    
    main_dir = sorted(os.listdir(path))
    print(main_dir)
    for data_dir in main_dir:
        all_files = sorted(os.listdir(path+data_dir))
        for file in all_files:
            name = file.split('_')[-1].split('.')
            label = name[0]
            ext = name[1]
            num_aae = 0
            num_non_aae = 0
            if label == "label" and ext == "txt":
                with open(path + data_dir + '/' + file) as file_handle:
                    for i, line in enumerate(file_handle, 1):
                        uid, tweet, primary_label, gtp_label, dialect_label = line.split('\t')
                        dialect_label = dialect_label.strip()
                        if dialect_label == "1":
                            num_aae += 1
                        else:
                            num_non_aae += 1
                    print(f"{file} has {i} tweets, # aae: {num_aae}, # non aae: {num_non_aae}")
                    

def data_stats_for_dialect_labels(data_path, dataset_name, with_gpt_label=True):
    label_map = {
        "waseem": {1: "Racism", 2: "Sexism", 3: "Racism and Sexism", 0: "Neither"},
        "waseem_and_hovy": {1: "Racism", 2: "Sexism", 0: "None"},
        "founta": {1: "Hateful", 2: "Abusive", 0: "Normal"},
        "davidson": {1: "Hate", 2: "Offensive", 0: "Normal"},
        "golbeck": {1: "Harassment", 0: "Non-harassment"},
        "hateval": {1: "Hate", 0: "Non-hate"},
        "offenseval": {1: "Offensive", 0: "Non-offensive"},
        "abuseval": {1: "Explicit abuse", 2: "Implicit abuse", 0: "Not abusive"}
    }
    
    dialect_map = {
        0: 'sae',
        1: 'aae'
    }
    
    counts = defaultdict(lambda: defaultdict(int))
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            if with_gpt_label:
                gpt_label = int(line[3])
                binary_label = int(line[4])
                dialect_label = int(line[5])
                counts[label_map[dataset_name][gpt_label]][dialect_map[dialect_label]] += 1
            else:
                binary_label = int(line[3])
                dialect_label = int(line[4])
                counts[label_map[dataset_name][human_label]][dialect_map[dialect_label]] += 1
           
    for label, dialect_data in counts.items():
        for dialect, count in dialect_data.items():
            logging.info(f'{label}: {dialect}: {count}')
    logging.info(f'--------------------------------------')
    
    
def filter_tweets_by_term(black_aligned_path, 
                          save_black_aligned_as, 
                          white_aligned_path, 
                          save_white_aligned_as, 
                          term="nigga",
                          include_term=False):
    
    count_tweet_without_term = 0
    with open(save_black_aligned_as, 'w') as save_black_aligned_handler:
        with open(black_aligned_path, encoding="utf-8") as file_handler:
            for line in file_handler:
                tweet = line.strip()
                tweet_set = set(tweet.split())
                if include_term:
                    if term in tweet_set:
                        save_black_aligned_handler.write(f"{tweet}\n")
                else:
                    if term not in tweet_set:
                        save_black_aligned_handler.write(f"{tweet}\n")
                        count_tweet_without_term += 1
    
    white_aligned_without_term = []
    with open(white_aligned_path, encoding="utf-8") as file_handler:
        for line in file_handler:
            tweet = line.strip()
            tweet_set = set(tweet.split())
            if not include_term:
                if term not in tweet_set:
                    white_aligned_without_term.append(tweet)
    
    # Since the number of tweets without term is greater in black-aligned tweets. Sample the number of tweets without term in black-aligned tweets from white-aligned tweets
    # Sample 2 tweets from each class
    random.seed(23)
    sampled_white_aligned = random.sample(white_aligned_without_term, count_tweet_without_term)
    with open(save_white_aligned_as, 'w') as save_white_aligned_handler:
        for tweet in sampled_white_aligned:
            save_white_aligned_handler.write(f"{tweet}\n")
            

def count_number_of_term_in_corpus(corpus_path, term="nigga"):
    
    logging.info(f'Processing: {corpus_path}')
    count = 0
    with open(corpus_path, encoding="utf-8") as file_handler:
        for line in file_handler:
            tweet = line.strip()
            tweet_set = set(tweet.split())
            if term in tweet_set:
                count += 1
    
    logging.info(f'Number of tweets with {term}: {count}\n')
    
    
def count_number_of_filtered_tweets(data_path):
    
    logging.info(f'Processing: {data_path}')
    with open(data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    logging.info(f'Number of tweets: {dataset_size}\n')
    

def dataset_for_race_priming(data_path, dialect_path, save_as):
    """
    data_path: path to the original tweet
    dialect_path: path to the preprocessed tweet with inferred dialect
    save_as: path to save the original tweet and inferred dialect
    """
    
    test_correctness = set()
    data_map = defaultdict(int)
    with open(dialect_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = line[2].strip()
            combined_label = line[3].strip()
            dialect_label = line[4].strip()
            data_map[tweet_id] = dialect_label
            if i < 2:
                test_correctness.add(tweet_id)
                print(row)
    
    directory, file_name = os.path.split(save_as)
    os.makedirs(os.path.dirname(save_as), exist_ok=True)
    
    with open(save_as, 'w') as file_handle:
        with open(data_path) as file_handle_original:
            for i, row in enumerate(file_handle_original):
                line = row.split('\t')
                tweet_id = line[0]
                tweet = line[1]
                human_label = line[2].strip()
                dialect_label = data_map[tweet_id]
                if tweet_id in test_correctness:
                    print(row, dialect_label)
                file_handle.write(f'{tweet_id}\t{tweet}\t{human_label}\t{dialect_label}\n')
    print()
    
    
def exemplars_with_dialects(exemplar_path, train_path_with_dialect, test_path_with_dialect, save_as):
    
    data_map = defaultdict(str)
    with open(train_path_with_dialect) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = line[2].strip()
            combined_label = line[3].strip()
            dialect_label = line[4].strip()
            data_map[tweet_id] = dialect_label
            
    with open(test_path_with_dialect) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = line[2].strip()
            combined_label = line[3].strip()
            dialect_label = line[4].strip()
            data_map[tweet_id] = dialect_label
    
    directory, file_name = os.path.split(save_as)
    os.makedirs(os.path.dirname(save_as), exist_ok=True)
    
    with open(save_as, 'w') as file_handle:
        with open(exemplar_path) as file_handle_original:
            for i, row in enumerate(file_handle_original):
                line = row.split('\t')
                tweet_id = line[0]
                tweet = line[1]
                human_label = line[2].strip()
                dialect_label = data_map[tweet_id]
                file_handle.write(f'{tweet_id}\t{tweet}\t{human_label}\t{dialect_label}\n')


def sample_exemplars_from_data_for_dialect_priming(data_path, 
                                                   dialect_data_path_train, 
                                                   dialect_data_path_test, 
                                                   save_path, 
                                                   seed=3):
    '''
    Sample exemplars (1 AAE and 1 SAE) from each class in data_path
    '''
    dialect_map = defaultdict(str)
    with open(dialect_data_path_train) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0].strip()
            tweet = line[1].strip()
            label = line[2].strip()
            combined_label = line[3].strip()
            dialect_label = line[4].strip()
            dialect_map[tweet_id] = dialect_label
    
    with open(dialect_data_path_test) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0].strip()
            tweet = line[1].strip()
            label = line[2].strip()
            combined_label = line[3].strip()
            dialect_label = line[4].strip()
            dialect_map[tweet_id] = dialect_label
            
    data_aae = defaultdict(list)
    data_sae = defaultdict(list)
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0].strip()
            tweet = line[1].strip()
            label = line[2].strip()
            dialect = dialect_map[tweet_id]
            new_row = [tweet_id, tweet, label, dialect]
            if dialect == '1':
                # Add each class as a key and the tweet details belonging to the class as a value represented as an element in a list
                data_aae[label].append(new_row)
            else:
                data_sae[label].append(new_row)

    sampled_exemplars = []
    # Sample 1 tweet from each class in aae
    
    random.seed(seed)
    for label, rows in data_aae.items():
        sampled_label_exemplar = random.choice(rows)
        sampled_exemplars.append(sampled_label_exemplar)
    
    # Sample 1 tweet from each class in aae
    random.seed(seed)
    for label, rows in data_sae.items():
        sampled_label_exemplar = random.choice(rows)
        sampled_exemplars.append(sampled_label_exemplar)
    
    absolute_path, file_name = os.path.split(save_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
       
    new_file_name = file_name.split('.')[0] + '_exemplars_2_samples_balanced_dialect.txt'
    new_save_path = absolute_path + '/' + new_file_name
    # Write exemplars to file
    with open(new_save_path, 'w') as save_file_handle:
        for item in sampled_exemplars:
            #line = item.split('\t')
            tweet_id = item[0].strip()
            tweet = item[1].strip()
            label = item[2].strip()
            dialect_label = item[3].strip()
            save_file_handle.write(f'{tweet_id}\t{tweet}\t{label}\t{dialect_label}\n')
            

def self_reported_user_data_stats(data_path):
    '''
    A few variables are missing for various reasons - these are 'nan' in the table. You will see fractional values for some attributes - this means that those users did the survey multiple times and entered different values for that item and the number is an average.
    For race, the coding is:
    1 - African-American
    2 - Latino/Hispanic
    3 - Asian
    4 - White
    5 - Multiracial
    NULL - didn't answer or other race (usually native american - these are all US users)  
    '''
    
    race_count = defaultdict(int)
    with open(data_path) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader, 1):
            # Skip the row with title
            if i == 1:
                continue 
            
            user_id = line[0]
            user_race = line[3]
            if user_race == '1':
                race_count['AA'] += 1
            elif user_race == '4':
                race_count['White'] += 1
    
    directory, file_name = os.path.split(data_path)
    logging.info(f'directory: {directory}, file name: {file_name}')
    logging.info(f'AA: {race_count["AA"]}, White: {race_count["White"]}')
            


def sample_users_from_self_reported_user_data(data_path):
    
    aa_users = []
    white_users = []
    
    with open(data_path) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader, 1):
            # Skip the row with title
            if i == 1:
                continue 
            
            user_id = line[0]
            user_race = line[3]
            if user_race == '1':
                aa_users.append(user_id)
            elif user_race == '4':
                white_users.append(user_id)
            
    # Sample 280 users from African-American users and 280 users White users
    # Twitter offers limited and slow data collection in the basing plan. So only collect tweets of 280 users from each race of interest
    aa_users = random.sample(aa_users, 280)
    white_users = random.sample(white_users, 280)
    
    directory, file_name = os.path.split(data_path)
    
    # Write sampled AA users to file
    with open(directory + '/aa_users.csv', mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, user_id in enumerate(aa_users):
            csv_handler.writerow([user_id, '1'])
    
    # Write sampled White users to file changing the white label to 0 from 4
    with open(directory + '/white_users.csv', mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, user_id in enumerate(white_users):
            csv_handler.writerow([user_id, '0'])
            

def stats_for_black_white_corpus(data_paths):
    
    for data_path in data_paths:
        with open(data_path, 'r') as file:
            # all_tweets = file.readlines()
            for i, line in enumerate(file, 1):
                continue
            logging.info(f"{data_path}: {i}\n")
                     

                
def combine_datasets(path, datasets, save_as, with_dialect_priming=False):
    
    absolute_path, file_name = os.path.split(save_as)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
    
    with open(save_as, 'w') as save_file_handle:
        for dataset in datasets:
            # We skip waseem, waseem and hove, and abuseval in dialect priming
            if with_dialect_priming and ('waseem' in dataset or 'abuseval' in dataset):
                continue
            with open(path + dataset) as file_handle:
                for i, row in enumerate(file_handle):
                    save_file_handle.write(row)
            
        
def main():
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    
    # DATASET_PATH = "/project/luofeng/socbd/eokpala/new_public_datasets/"
    # data_paths = [
    #     ("waseem/waseem_with_default_classes.txt", "waseem"),
    #     ("waseem-and-hovy/waseem_and_hovy_with_default_classes.txt", "waseem_and_hovy"),
    #     ("founta/founta_with_default_classes.txt", "founta"),
    #     ("davidson/davidson_with_default_classes.txt", "davidson"),
    #     ("golbeck/golbeck_with_default_classes.txt", "golbeck"),
    #     ("hateval2019/hateval2019_en_train.txt", "hateval"),
    #     ("hateval2019/hateval2019_en_test.txt", "hateval"),
    #     ("hateval2019/hateval2019_en_dev.txt", "hateval"),
    #     ("offenseval2019/offenseval_train.txt", "offenseval"),
    #     ("offenseval2019/offenseval_test.txt", "offenseval"),
    #     ("abuseval/abuseval_train_with_default_classes.txt", "abuseval"),
    #     ("abuseval/abuseval_test_with_default_classes.txt", "abuseval")
    # ]
    
    # Process data statistics
    # for data_path, dataset_name in data_paths:
    #     data_stats(DATASET_PATH + data_path, dataset_name, with_gpt_label=False)
    
    # To reduce cost of labeling using GPT, sample from orginal tweets
#     original_data_root_path = DATASET_PATH
#     save_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
    
#     data_path = "waseem/waseem_with_default_classes.txt"
#     sample_from_waseem_orginal_data(original_data_root_path, data_path, save_path)
    
#     data_path = "waseem-and-hovy/waseem_and_hovy_with_default_classes.txt"
#     sample_from_waseem_and_hovy_orginal_data(original_data_root_path, data_path, save_path)
    
#     data_path = "founta/founta_with_default_classes.txt"
#     sample_from_founta_orginal_data(original_data_root_path, data_path, save_path)
    
#     data_path = "davidson/davidson_with_default_classes.txt"
#     sample_from_davidson_orginal_data(original_data_root_path, data_path, save_path)
    
#     data_path = "golbeck/golbeck_with_default_classes.txt"
#     sample_from_golbeck_orginal_data(original_data_root_path, data_path, save_path)
    
#     data_path = "hateval2019/hateval2019_en_train.txt"
#     sample_from_hateval_orginal_data(original_data_root_path, data_path, save_path)
    
#     data_path = "hateval2019/hateval2019_en_test.txt"
#     sample_from_hateval_orginal_data(original_data_root_path, data_path, save_path)
    
    # Since we are not sampling from the val set of hateval, copy it over to save_path
#     data_path = "hateval2019/hateval2019_en_dev.txt"
#     shutil.copy(original_data_root_path + data_path, save_path + data_path)
    
#     data_path = "offenseval2019/offenseval_train.txt"
#     sample_from_offenseval_orginal_data(original_data_root_path, data_path, save_path)
    
#     # Since we are not sampling from the val set of offenseval, copy it over to save_path
#     data_path = "offenseval2019/offenseval_test.txt"
#     shutil.copy(original_data_root_path + data_path, save_path + data_path)
    
#     data_path = "abuseval/abuseval_train_with_default_classes.txt"
#     sample_from_abuseval_orginal_data(original_data_root_path, data_path, save_path)
    
    # Since we are not sampling from the val set of abuseval, copy it over to save_path
    # data_path = "abuseval/abuseval_test_with_default_classes.txt"
    # shutil.copy(original_data_root_path + data_path, save_path + data_path)
    
#     data_paths = [
#         ("waseem/waseem_with_default_classes.txt", "waseem"),
#         ("waseem-and-hovy/waseem_and_hovy_with_default_classes.txt", "waseem_and_hovy"),
#         ("founta/founta_with_default_classes.txt", "founta"),
#         ("davidson/davidson_with_default_classes.txt", "davidson"),
#         ("golbeck/golbeck_with_default_classes.txt", "golbeck"),
#         ("hateval2019/hateval2019_en_train.txt", "hateval"),
#         ("hateval2019/hateval2019_en_test.txt", "hateval"),
#         ("hateval2019/hateval2019_en_dev.txt", "hateval"),
#         ("offenseval2019/offenseval_train.txt", "offenseval"),
#         ("offenseval2019/offenseval_test.txt", "offenseval"),
#         ("abuseval/abuseval_train_with_default_classes.txt", "abuseval"),
#         ("abuseval/abuseval_test_with_default_classes.txt", "abuseval")
#     ]
    
#     DATASET_PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     # Process data statistics after sampling from original dataset
#     for data_path, dataset_name in data_paths:
#         data_stats(DATASET_PATH + data_path, dataset_name, with_gpt_label=False)
    
    # Perform general prompt annotation with GPT using gpt_annotation.py
    
    # To perform few-shot and CoT annotation with GPT, extract exemplars, 2 from each class in each dataset
#     DATASET_PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     data_paths = [
#         "waseem/waseem_with_default_classes.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes.txt",
#         "founta/founta_with_default_classes.txt",
#         "davidson/davidson_with_default_classes.txt",
#         "golbeck/golbeck_with_default_classes.txt",
#         "hateval2019/hateval2019_en_train.txt",
#         "offenseval2019/offenseval_train.txt",
#         "abuseval/abuseval_train_with_default_classes.txt"
#     ]
    
#     SAVE_PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/few_shot_prompt_annotation/"
#     for data_path in data_paths:
#         sample_exemplars_from_data(DATASET_PATH + data_path, SAVE_PATH + data_path)
    
    # Perform few-shot prompt annotation with GPT using gpt_annotation.py
    
    # Perform CoT prompt annotation with GPT using gpt_annotation.py
    
    # Split dataset that were not originally split into train and test. Split the sampled tweets without GPT labels
#     path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
    
#     data_paths = [
#         "waseem/waseem_with_default_classes.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes.txt",
#         "founta/founta_with_default_classes.txt",
#         "davidson/davidson_with_default_classes.txt",
#         "golbeck/golbeck_with_default_classes.txt"
#     ]
    
#     for data in data_paths:
#         dataset = data.split('/')
#         dataset_name = dataset[0]
#         dataset_file = dataset[1]
#         name_list = dataset_file.split('.')
#         save_train_as = dataset_name + "/" + name_list[0] + "_train.txt"
#         save_test_as = dataset_name + "/" + name_list[0] + "_test.txt"
#         split_dataset(path + data, path + save_train_as, path + save_test_as)
    
#     data_paths = [
#         "waseem/waseem_with_default_classes_train.txt",
#         "waseem/waseem_with_default_classes_test.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_train.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_test.txt",
#         "founta/founta_with_default_classes_train.txt",
#         "founta/founta_with_default_classes_test.txt",
#         "davidson/davidson_with_default_classes_train.txt",
#         "davidson/davidson_with_default_classes_test.txt",
#         "golbeck/golbeck_with_default_classes_train.txt",
#         "golbeck/golbeck_with_default_classes_test.txt"
#     ]
    
#     # Check stats of split data
#     for data_path in data_paths:
#         dataset_name = data_path.split('/')[0]
#         if dataset_name == "waseem-and-hovy":
#             dataset_name = "waseem_and_hovy"
#         data_stats(path + data_path, dataset_name, with_gpt_label=False)
    
    # Should we use stratified sampling for GPT annotated tweets?
    # Before deciding to split dataset with gpt labels with stratified splitting, check the class balance
    # Data stats after GPT annoation
#     data_paths = [
#         ("waseem/waseem_with_default_classes.txt", "waseem"),
#         ("waseem-and-hovy/waseem_and_hovy_with_default_classes.txt", "waseem_and_hovy"),
#         ("founta/founta_with_default_classes.txt", "founta"),
#         ("davidson/davidson_with_default_classes.txt", "davidson"),
#         ("golbeck/golbeck_with_default_classes.txt", "golbeck"),
#         ("hateval2019/hateval2019_en_train.txt", "hateval"),
#         ("hateval2019/hateval2019_en_test.txt", "hateval"),
#         ("hateval2019/hateval2019_en_dev.txt", "hateval"),
#         ("offenseval2019/offenseval_train.txt", "offenseval"),
#         ("offenseval2019/offenseval_test.txt", "offenseval"),
#         ("abuseval/abuseval_train_with_default_classes.txt", "abuseval"),
#         ("abuseval/abuseval_test_with_default_classes.txt", "abuseval")
#     ]
    
#     DATASET_PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     paths = [
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation/",
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/few_shot_prompt_annotation/",    
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/cot_prompt_annotation/"
#     ]
#     for path in paths:
#         logging.info(f'GPT annotation using {path}')
#         for data_path, dataset_name in data_paths:
#             data_stats(path + data_path, dataset_name, with_gpt_label=True)

    # Split dataset that were not originally split into train and test. Split the GPT annotated tweets
#     paths = [
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation/",
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/few_shot_prompt_annotation/",    
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/cot_prompt_annotation/"
#     ]
    
#     data_paths = [
#         "waseem/waseem_with_default_classes.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes.txt",
#         "founta/founta_with_default_classes.txt",
#         "davidson/davidson_with_default_classes.txt",
#         "golbeck/golbeck_with_default_classes.txt"
#     ]
    
#     for path in paths:
#         for data in data_paths:
#             dataset = data.split('/')
#             dataset_name = dataset[0]
#             dataset_file = dataset[1]
#             name_list = dataset_file.split('.')
#             save_train_as = dataset_name + "/" + name_list[0] + "_train.txt"
#             save_test_as = dataset_name + "/" + name_list[0] + "_test.txt"
#             split_dataset_with_gpt_label(path + data, path + save_train_as, path + save_test_as)
    
#     # Data stats of split GPT annotated data
#     split_data_paths = [
#         "waseem/waseem_with_default_classes_train.txt",
#         "waseem/waseem_with_default_classes_test.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_train.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_test.txt",
#         "founta/founta_with_default_classes_train.txt",
#         "founta/founta_with_default_classes_test.txt",
#         "davidson/davidson_with_default_classes_train.txt",
#         "davidson/davidson_with_default_classes_test.txt",
#         "golbeck/golbeck_with_default_classes_train.txt",
#         "golbeck/golbeck_with_default_classes_test.txt"
#     ]
    
#     # Check stats of split data
#     for path in paths:
#         logging.info(f'GPT annotation stats for {path}')
#         for data_path in split_data_paths:
#             dataset_name = data_path.split('/')[0]
#             if dataset_name == "waseem-and-hovy":
#                 dataset_name = "waseem_and_hovy"
#             data_stats(path + data_path, dataset_name, with_gpt_label=True)
    
    # Compute the confusion metrics of human vs gpt label - i.e for each tweet belonging to a human labeled class in a dataset
    # how many did GPT get wrong be assigning it a different classes
#     paths = [
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation/",
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/few_shot_prompt_annotation/",    
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/cot_prompt_annotation/"
#     ]
    
#     data_paths = [
#         ("waseem/waseem_with_default_classes.txt", "waseem"),
#         ("waseem-and-hovy/waseem_and_hovy_with_default_classes.txt", "waseem_and_hovy"),
#         ("founta/founta_with_default_classes.txt", "founta"),
#         ("davidson/davidson_with_default_classes.txt", "davidson"),
#         ("golbeck/golbeck_with_default_classes.txt", "golbeck"),
#         ("hateval2019/hateval2019_en_train.txt", "hateval"),
#         ("hateval2019/hateval2019_en_test.txt", "hateval"),
#         ("hateval2019/hateval2019_en_dev.txt", "hateval"),
#         ("offenseval2019/offenseval_train.txt", "offenseval"),
#         ("offenseval2019/offenseval_test.txt", "offenseval"),
#         ("abuseval/abuseval_train_with_default_classes.txt", "abuseval"),
#         ("abuseval/abuseval_test_with_default_classes.txt", "abuseval")
#     ]
    
#     for path in paths:
#         logging.info(f'Human vs GPT annotation confusion matrix for {path}')
#         for data_path, dataset_name in data_paths:
#             compare_human_vs_gpt_labels(path + data_path, dataset_name)
    

    # Preprocess each dataset
    # Without GPT labels
#     data_paths = [
#         "waseem/waseem_with_default_classes_train.txt",
#         "waseem/waseem_with_default_classes_test.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_train.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_test.txt",
#         "founta/founta_with_default_classes_train.txt",
#         "founta/founta_with_default_classes_test.txt",
#         "davidson/davidson_with_default_classes_train.txt",
#         "davidson/davidson_with_default_classes_test.txt",
#         "golbeck/golbeck_with_default_classes_train.txt",
#         "golbeck/golbeck_with_default_classes_test.txt",
#         "hateval2019/hateval2019_en_train.txt",
#         "hateval2019/hateval2019_en_test.txt",
#         "hateval2019/hateval2019_en_dev.txt",
#         "offenseval2019/offenseval_train.txt",
#         "offenseval2019/offenseval_test.txt",
#         "abuseval/abuseval_train_with_default_classes.txt",
#         "abuseval/abuseval_test_with_default_classes.txt"
#     ]
    
#     path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     for data in data_paths:
#         dataset = data.split('/')
#         dataset_name = dataset[0]
#         dataset_file = dataset[1]
#         save_as = dataset_name + '/' + dataset_file.split('.')[0] + '_preprocessed.txt'
#         preprocess(path + data, path + save_as, dataset=dataset_name, with_gpt_label=False)
    
#     # With GPT labels
#     data_paths = [
#         "waseem/waseem_with_default_classes_train.txt",
#         "waseem/waseem_with_default_classes_test.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_train.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_test.txt",
#         "founta/founta_with_default_classes_train.txt",
#         "founta/founta_with_default_classes_test.txt",
#         "davidson/davidson_with_default_classes_train.txt",
#         "davidson/davidson_with_default_classes_test.txt",
#         "golbeck/golbeck_with_default_classes_train.txt",
#         "golbeck/golbeck_with_default_classes_test.txt",
#         "hateval2019/hateval2019_en_train.txt",
#         "hateval2019/hateval2019_en_test.txt",
#         "hateval2019/hateval2019_en_dev.txt",
#         "offenseval2019/offenseval_train.txt",
#         "offenseval2019/offenseval_test.txt",
#         "abuseval/abuseval_train_with_default_classes.txt",
#         "abuseval/abuseval_test_with_default_classes.txt"
#     ]
    
#     paths = [
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation/",
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/few_shot_prompt_annotation/",    
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/cot_prompt_annotation/"
#     ]
        
#     for path in paths:
#         for data in data_paths:
#             dataset = data.split('/')
#             dataset_name = dataset[0]
#             dataset_file = dataset[1]
#             save_as = dataset_name + '/' + dataset_file.split('.')[0] + '_preprocessed.txt'
#             preprocess(path + data, path + save_as, dataset=dataset_name, with_gpt_label=True)
        
    # Infer dialect using dialect_inference.py
    
    # Stats for inferred dialects
#     data_paths = [
#         "waseem/waseem_with_default_classes_train_preprocessed_dialect_label.txt",
#         "waseem/waseem_with_default_classes_test_preprocessed_dialect_label.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_train_preprocessed_dialect_label.txt",
#         "waseem-and-hovy/waseem_and_hovy_with_default_classes_test_preprocessed_dialect_label.txt",
#         "founta/founta_with_default_classes_train_preprocessed_dialect_label.txt",
#         "founta/founta_with_default_classes_test_preprocessed_dialect_label.txt",
#         "davidson/davidson_with_default_classes_train_preprocessed_dialect_label.txt",
#         "davidson/davidson_with_default_classes_test_preprocessed_dialect_label.txt",
#         "golbeck/golbeck_with_default_classes_train_preprocessed_dialect_label.txt",
#         "golbeck/golbeck_with_default_classes_test_preprocessed_dialect_label.txt",
#         "hateval2019/hateval2019_en_train_preprocessed_dialect_label.txt",
#         "hateval2019/hateval2019_en_test_preprocessed_dialect_label.txt",
#         "hateval2019/hateval2019_en_dev_preprocessed_dialect_label.txt",
#         "offenseval2019/offenseval_train_preprocessed_dialect_label.txt",
#         "offenseval2019/offenseval_test_preprocessed_dialect_label.txt",
#         "abuseval/abuseval_train_with_default_classes_preprocessed_dialect_label.txt",
#         "abuseval/abuseval_test_with_default_classes_preprocessed_dialect_label.txt"
#     ]
    
#     paths = [
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation/",
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/few_shot_prompt_annotation/",    
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/cot_prompt_annotation/"
#     ]
    
    # Dialect stats for GPT annotated data
    # for prompt_path in paths:
    #     logging.info(f'----- Processing {prompt_path}-----')
    #     for data_path in data_paths:
    #         logging.info(f'{data_path}')
    #         dataset = data_path.split('/')
    #         dataset_name = dataset[0]
    #         if dataset_name == "waseem-and-hovy":
    #             dataset_name = "waseem_and_hovy"
    #         elif dataset_name == "hateval2019":
    #             dataset_name = "hateval"
    #         elif dataset_name == "offenseval2019":
    #             dataset_name = "offenseval"
    #         data_stats_for_dialect_labels(prompt_path + data_path, dataset_name, with_gpt_label=True)
    
    # Dialect stats for human annotated data
    # paths = ["/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"]
    # for prompt_path in paths:
    #     logging.info(f'----- Processing {prompt_path}-----')
    #     for data_path in data_paths:
    #         dataset = data_path.split('/')
    #         dataset_name = dataset[0]
    #         logging.info(f'{data_path}')
    #         if dataset_name == "waseem-and-hovy":
    #             dataset_name = "waseem_and_hovy"
    #         elif dataset_name == "hateval2019":
    #             dataset_name = "hateval"
    #         elif dataset_name == "offenseval2019":
    #             dataset_name = "offenseval"
    #         data_stats_for_dialect_labels(prompt_path + data_path, dataset_name, with_gpt_label=False)
    
    # Stats for hovy
    # data_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/waseem-and-hovy/waseem_and_hovy_with_default_classes_test_preprocessed_dialect_label.txt"
    # dataset_name = "waseem_and_hovy"
    # data_stats(data_path, dataset_name, with_gpt_label=False)
    
    # Today (04/10/24): We will be skipping W&H in our general prompt annotation analysis because there is no aae dialect tweet labeled Hate in the test set making roc_auc_score function unable to compute a score due to the error - "y_true contains only one class". Verifying shows that y_true contains only the "0" class
    # data_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation/waseem-and-hovy/waseem_and_hovy_with_default_classes_test_preprocessed_dialect_label.txt"
    # dataset_name = "waseem_and_hovy"
    # data_stats_binary(data_path, dataset_name, with_gpt_label=True)
    
    # Count number of "nigga" in black and white-aligned tweets
    # black_aligned_dataset = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_preprocessed.txt"
    # white_aligned_dataset = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_white_aligned_preprocessed.txt"
    # count_number_of_term_in_corpus(black_aligned_dataset, term="nigga")
    # count_number_of_term_in_corpus(white_aligned_dataset, term="nigga")
    
    # Extract tweets not containing the term "nigga" in both black and white-alined tweets
    # black_aligned_dataset = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_preprocessed.txt"
    # white_aligned_dataset = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_white_aligned_preprocessed.txt"
    # save_black_aligned_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_preprocessed_without_nigga.txt"
    # save_white_aligned_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_white_aligned_preprocessed_without_nigga.txt"
    # filter_tweets_by_term(black_aligned_dataset, 
    #                       save_black_aligned_as, 
    #                       white_aligned_dataset, 
    #                       save_white_aligned_as,
    #                       term="nigga",
    #                       include_term=False)
    
    # count_number_of_filtered_tweets(save_black_aligned_as)
    # count_number_of_filtered_tweets(save_white_aligned_as)
    
    
    # Dialect Priming 
    
    # Organize the datasets to be used for dialect priming. The dialect of each tweet in each dataset have already been inferred. 
    # However, the inferrence was done on preprocessed tweets. We want to pass the original tweet and not the preprocessed to GPT. So, extract 
    # the original tweet and the inferred dialect from the preprocessed version
    
    # path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
    # save_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data_for_dialect_priming/"
    # data_paths = [
    #     "founta/founta_with_default_classes_train.txt",
    #     "founta/founta_with_default_classes_test.txt",
    #     "davidson/davidson_with_default_classes_train.txt",
    #     "davidson/davidson_with_default_classes_test.txt",
    #     "golbeck/golbeck_with_default_classes_train.txt",
    #     "golbeck/golbeck_with_default_classes_test.txt",
    #     "hateval2019/hateval2019_en_train.txt",
    #     "hateval2019/hateval2019_en_test.txt",
    #     "offenseval2019/offenseval_train.txt",
    #     "offenseval2019/offenseval_test.txt"
    # ]

    # path with dialect
    # dialect_data_paths = [
    #     "founta/founta_with_default_classes_train_preprocessed_dialect_label.txt",
    #     "founta/founta_with_default_classes_test_preprocessed_dialect_label.txt",
    #     "davidson/davidson_with_default_classes_train_preprocessed_dialect_label.txt",
    #     "davidson/davidson_with_default_classes_test_preprocessed_dialect_label.txt",
    #     "golbeck/golbeck_with_default_classes_train_preprocessed_dialect_label.txt",
    #     "golbeck/golbeck_with_default_classes_test_preprocessed_dialect_label.txt",
    #     "hateval2019/hateval2019_en_train_preprocessed_dialect_label.txt",
    #     "hateval2019/hateval2019_en_test_preprocessed_dialect_label.txt",
    #     "offenseval2019/offenseval_train_preprocessed_dialect_label.txt",
    #     "offenseval2019/offenseval_test_preprocessed_dialect_label.txt"
    # ]

    # Each tweet in save_path + dataset will be of the form id \t tweet \t label \t dialect
    # for i, dataset in enumerate(data_paths):
    #     dataset_for_race_priming(path + dataset, path + dialect_data_paths[i], save_path + dataset)
    
    
    # Extract exemplars that are balanced based on dialect for each tweet in each dataset to better guide GPT
#     data_location = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     data_paths = [
#         "founta/founta_with_default_classes.txt",
#         "davidson/davidson_with_default_classes.txt",
#         "golbeck/golbeck_with_default_classes.txt",
#         "hateval2019/hateval2019_en_train.txt",
#         "offenseval2019/offenseval_train.txt",
#     ]

#     dialect_location = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     dialect_data_paths = [
#         ("founta/founta_with_default_classes_train_preprocessed_dialect_label.txt", "founta/founta_with_default_classes_test_preprocessed_dialect_label.txt"),
#         ("davidson/davidson_with_default_classes_train_preprocessed_dialect_label.txt", "davidson/davidson_with_default_classes_test_preprocessed_dialect_label.txt"),
#         ("golbeck/golbeck_with_default_classes_train_preprocessed_dialect_label.txt", "golbeck/golbeck_with_default_classes_test_preprocessed_dialect_label.txt"),
#         ("hateval2019/hateval2019_en_train_preprocessed_dialect_label.txt", "hateval2019/hateval2019_en_test_preprocessed_dialect_label.txt"),
#         ("offenseval2019/offenseval_train_preprocessed_dialect_label.txt", "offenseval2019/offenseval_test_preprocessed_dialect_label.txt")
#     ]
    
#     save_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/few_shot_prompt_annotation/"
#     for i, data_path in enumerate(data_paths):
#         dialect_train_path, dialect_test_path = dialect_data_paths[i]
#         save_as = save_path + data_path
#         sample_exemplars_from_data_for_dialect_priming(data_location + data_path, 
#                                                        dialect_location + dialect_train_path, 
#                                                        dialect_location + dialect_test_path, 
#                                                        save_as)
        
    # regenerate exemplars for Golbeck because the previous not-harassment exemplar was actually being harassment. And Founta, the previous hate exemplar was not hate. 
#     data_location = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     data_paths = [
#         "founta/founta_with_default_classes.txt"
#     ]

#     dialect_location = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     dialect_data_paths = [
#         ("founta/founta_with_default_classes_train_preprocessed_dialect_label.txt", "founta/founta_with_default_classes_test_preprocessed_dialect_label.txt")
#     ]
    
#     save_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/few_shot_prompt_annotation/"
#     for i, data_path in enumerate(data_paths):
#         dialect_train_path, dialect_test_path = dialect_data_paths[i]
#         save_as = save_path + data_path
#         sample_exemplars_from_data_for_dialect_priming(data_location + data_path, 
#                                                        dialect_location + dialect_train_path, 
#                                                        dialect_location + dialect_test_path, 
#                                                        save_as,
#                                                        seed=1995)
    
#     data_location = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     data_paths = [
#         "golbeck/golbeck_with_default_classes.txt"
#     ]

#     dialect_location = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
#     dialect_data_paths = [
#         ("golbeck/golbeck_with_default_classes_train_preprocessed_dialect_label.txt", "golbeck/golbeck_with_default_classes_test_preprocessed_dialect_label.txt")
#     ]
    
#     save_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/few_shot_prompt_annotation/"
#     for i, data_path in enumerate(data_paths):
#         dialect_train_path, dialect_test_path = dialect_data_paths[i]
#         save_as = save_path + data_path
#         sample_exemplars_from_data_for_dialect_priming(data_location + data_path, 
#                                                        dialect_location + dialect_train_path, 
#                                                        dialect_location + dialect_test_path, 
#                                                        save_as,
#                                                        seed=45)
        
    
    # Perform annotation using general, few-shot, and CoT prompting technique with dialect priming
    
    # Preprocess dialect primed GPT annotated data
#     data_paths = [
#         "founta/founta_with_default_classes_train.txt",
#         "founta/founta_with_default_classes_test.txt",
#         "davidson/davidson_with_default_classes_train.txt",
#         "davidson/davidson_with_default_classes_test.txt",
#         "golbeck/golbeck_with_default_classes_train.txt",
#         "golbeck/golbeck_with_default_classes_test.txt",
#         "hateval2019/hateval2019_en_train.txt",
#         "hateval2019/hateval2019_en_test.txt",
#         "offenseval2019/offenseval_train.txt",
#         "offenseval2019/offenseval_test.txt"
#     ]

#     paths = [
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/general_prompt_annotation/",
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/few_shot_prompt_annotation/",    
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/cot_prompt_annotation/"
#     ]
        
#     for path in paths:
#         for data in data_paths:
#             dataset = data.split('/')
#             dataset_name = dataset[0]
#             dataset_file = dataset[1]
#             save_as = dataset_name + '/' + dataset_file.split('.')[0] + '_preprocessed_dialect_label.txt'
#             preprocess_with_dialect(path + data, path + save_as, dataset=dataset_name, with_gpt_label=True)

    # data_path ='self_reported_user_data/users-demo.csv'
    # self_reported_user_data_stats(data_path)
    # sample_users_from_self_reported_user_data(data_path)
    
    # Get the stats of the black and white corpus
    # white = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/white_aligned_preprocessed.txt"
    # black = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/black_aligned_preprocessed.txt"
    
    # Stats for the black corpus used in retraining BERTweet
    # black_aaebertweet = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_train_set_preprocessed.txt"
    # stats_for_black_white_corpus([white, black, black_aaebertweet])
    
    '''
    ICWSM Revision 
    '''
    # Perform general prompt annotation with LLaMa3
    
    # Compute Cohen Kappa
    
    # Preprocess each LLaMa3 annotated dataset without dialect priming
    # With GPT labels
    # data_paths = [
    #     "waseem/waseem_with_default_classes_train.txt",
    #     "waseem/waseem_with_default_classes_test.txt",
    #     "waseem-and-hovy/waseem_and_hovy_with_default_classes_train.txt",
    #     "waseem-and-hovy/waseem_and_hovy_with_default_classes_test.txt",
    #     "founta/founta_with_default_classes_train.txt",
    #     "founta/founta_with_default_classes_test.txt",
    #     "davidson/davidson_with_default_classes_train.txt",
    #     "davidson/davidson_with_default_classes_test.txt",
    #     "golbeck/golbeck_with_default_classes_train.txt",
    #     "golbeck/golbeck_with_default_classes_test.txt",
    #     "hateval2019/hateval2019_en_train.txt",
    #     "hateval2019/hateval2019_en_test.txt",
    #     "hateval2019/hateval2019_en_dev.txt",
    #     "offenseval2019/offenseval_train.txt",
    #     "offenseval2019/offenseval_test.txt",
    #     "abuseval/abuseval_train_with_default_classes.txt",
    #     "abuseval/abuseval_test_with_default_classes.txt"
    # ]

#     paths = [
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama/general_prompt_annotation/"
#     ]
        
#     for path in paths:
#         for data in data_paths:
#             dataset = data.split('/')
#             dataset_name = dataset[0]
#             dataset_file = dataset[1]
#             save_as = dataset_name + '/' + dataset_file.split('.')[0] + '_preprocessed.txt'
#             preprocess(path + data, path + save_as, dataset=dataset_name, with_gpt_label=True)
    
    # Perform general prompt annotation with LLaMa3 using dialect priming 
    
    # Compute Cohen Kappa

    # Preprocess each LLaMa3 annotated dataset with dialect priming
    # data_paths = [
    #     "founta/founta_with_default_classes_train.txt",
    #     "founta/founta_with_default_classes_test.txt",
    #     "davidson/davidson_with_default_classes_train.txt",
    #     "davidson/davidson_with_default_classes_test.txt",
    #     "golbeck/golbeck_with_default_classes_train.txt",
    #     "golbeck/golbeck_with_default_classes_test.txt",
    #     "hateval2019/hateval2019_en_train.txt",
    #     "hateval2019/hateval2019_en_test.txt",
    #     "offenseval2019/offenseval_train.txt",
    #     "offenseval2019/offenseval_test.txt"
    # ]
    
#     paths = [
#         "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/general_prompt_annotation/"
#     ]
    
#     for path in paths:
#         for data in data_paths:
#             dataset = data.split('/')
#             dataset_name = dataset[0]
#             dataset_file = dataset[1]
#             save_as = dataset_name + '/' + dataset_file.split('.')[0] + '_preprocessed_dialect_label.txt'
#             preprocess_with_dialect(path + data, path + save_as, dataset=dataset_name, with_gpt_label=True)

    # Combine the datasets - human annotated 
    # data_train_paths = [
    #     "waseem/waseem_with_default_classes_train_preprocessed_dialect_label.txt",
    #     "waseem-and-hovy/waseem_and_hovy_with_default_classes_train_preprocessed_dialect_label.txt",
    #     "founta/founta_with_default_classes_train_preprocessed_dialect_label.txt",
    #     "davidson/davidson_with_default_classes_train_preprocessed_dialect_label.txt",
    #     "golbeck/golbeck_with_default_classes_train_preprocessed_dialect_label.txt",
    #     "hateval2019/hateval2019_en_train_preprocessed_dialect_label.txt",
    #     "offenseval2019/offenseval_train_preprocessed_dialect_label.txt",
    #     "abuseval/abuseval_train_with_default_classes_preprocessed_dialect_label.txt"
    # ]
    # data_test_paths = [
    #     "waseem/waseem_with_default_classes_test_preprocessed_dialect_label.txt",
    #     "waseem-and-hovy/waseem_and_hovy_with_default_classes_test_preprocessed_dialect_label.txt",
    #     "founta/founta_with_default_classes_test_preprocessed_dialect_label.txt",
    #     "davidson/davidson_with_default_classes_test_preprocessed_dialect_label.txt",
    #     "golbeck/golbeck_with_default_classes_test_preprocessed_dialect_label.txt",
    #     "hateval2019/hateval2019_en_test_preprocessed_dialect_label.txt",
    #     "offenseval2019/offenseval_test_preprocessed_dialect_label.txt",
    #     "abuseval/abuseval_test_with_default_classes_preprocessed_dialect_label.txt"
    # ]
    
    # path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
    # save_train_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/all_data/all_data_train_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_train_paths, save_train_path)
    # save_test_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/all_data/all_data_test_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_test_paths, save_test_path)
    
    # Combine the datasets - GPT4 general prompt annotated 
    # path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation/"
    # save_train_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation_combined_data/all_data/all_data_train_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_train_paths, save_train_path)
    # save_test_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation_combined_data/all_data/all_data_test_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_test_paths, save_test_path)
    
    # Combine the datasets - GPT4 general prompt annotated using dialect priming
    # path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/general_prompt_annotation/"
    # save_train_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/general_prompt_annotation_combined_data/all_data/all_data_train_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_train_paths, save_train_path, with_dialect_priming=True)
    # save_test_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/general_prompt_annotation_combined_data/all_data/all_data_test_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_test_paths, save_test_path, with_dialect_priming=True)
    
    # Combine the datasets - LLaMa3 general prompt annotated
    # path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama/general_prompt_annotation/"
    # save_train_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama/general_prompt_annotation_combined_data/all_data/all_data_train_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_train_paths, save_train_path)
    # save_test_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama/general_prompt_annotation_combined_data/all_data/all_data_test_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_test_paths, save_test_path)
    
    # Combine the datasets - LLaMa3 general prompt annotated using dialect priming
    # path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/general_prompt_annotation/"
    # save_train_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/general_prompt_annotation_combined_data/all_data/all_data_train_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_train_paths, save_train_path, with_dialect_priming=True)
    # save_test_path = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/general_prompt_annotation_combined_data/all_data/all_data_test_preprocessed_dialect_label.txt"
    # combine_datasets(path, data_test_paths, save_test_path, with_dialect_priming=True)
    
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    custom_name = ''
    _ = get_logger(log_dir, get_filename() + f'_{custom_name}')
    main()