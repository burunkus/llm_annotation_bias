import os
import re
import emoji
import json
import datetime
import csv
import sys
import numpy as np
import torch
import pickle
import random
import string
from sklearn.model_selection import train_test_split
from _datetime import datetime as dt
    
    
def preprocess(file_path, save_as, golbeck=False):
    """
    Proprocess tweets by lower casing, normalize by converting
    user mentions to @USER, url to HTTPURL, and number to NUMBER. 
    Convert emoji to text string and remove duplicate tweets
    Args:
        file_path(String): location of file to preprocess
        save_as(String): name to save the preproced tweet in.
        golbeck (Boolean): The Golbeck dataset contains wierd urls, setting to True tries to handle most of these urls
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
    
    seen = set()
    with open(save_as, 'a+') as new_file:
        with open(file_path) as old_file:
            for i, line in enumerate(old_file, 1):
                line = line.split("\t")
                tweet_id = line[0].strip()
                tweet = line[1].strip()
                label = line[2].strip()
                tweet = tweet.strip('"')
                tweet = tweet.lower()
                if tweet not in seen:
                    # remove stock market tickers like $GE
                    tweet = re.sub(r'\$\w*', '', tweet)
                    # remove old style retweet text "RT"
                    tweet = re.sub(r'^rt[\s]+', '', tweet)
                    # replace hyperlinks with URL
                    if golbeck:
                        tweet = re.sub(r'(https?:\s?\/\s?\/[a-zA-Z0-9]*\.?[a-zA-Z0-9]+\s?\/?[a-zA-Z0-9]*|https?:\\\/\\\/[a-zA-Z0-9]*\.?[a-zA-Z0-9]+\s?\\\/?[a-zA-Z0-9]*|https?:?\s?\/?\s?\/?)', 'HTTPURL', tweet)
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
                    new_file.write(f"{tweet_id}\t{tweet}\t{label}\n")
                    seen.add(tweet)
                    

def preprocess_offenseval(file_path, save_as):
    """
    Proprocess OffensEval/AbusEval tweets by lower casing, normalize by converting
    user mentions to @USER, urls to HTTPURL, and number to NUMBER. 
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
    
    seen = set()
    with open(save_as, 'a+') as new_file:
        with open(file_path) as old_file:
            for i, line in enumerate(old_file, 1):
                line = line.split("\t")
                tweet_id = line[0].strip()
                tweet = line[1].strip()
                label = line[2].strip()
                tweet = tweet.strip()
                tweet = tweet.strip('"')
                tweet = tweet.lower()
                if tweet not in seen:
                    # remove stock market tickers like $GE
                    tweet = re.sub(r'\$\w*', '', tweet)
                    # remove old style retweet text "RT"
                    tweet = re.sub(r'^rt[\s]+', '', tweet)
                    # replace hyperlinks with HTTPURL
                    tweet = re.sub(r'(url)', 'HTTPURL', tweet)
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
                    new_file.write(f"{tweet_id}\t{tweet}\t{label}\n")
                    seen.add(tweet)
                    
                    
def handle_waseem_and_hovy_with_default_classes(in_file, save_text_to):
    """
    Extract original labels from the Waseem and Hovy dataset
    Args:
        in_file(String): the path to the training/test set in jsonl format of waseem and hovy
        each line is expected to contain a tweet json and label separated by tab
        save_text_to (String): the path to save the extracted tweets and lables in
    Returns:
        None
    """
    
    absolute_path, file_name = os.path.split(save_text_to)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    with open(in_file) as file_handle:
        for i, line in enumerate(file_handle, 1):
            tweet_json, label = line.split('\t')
            label = label.strip()
            tweet_object = json.loads(tweet_json)
            tweet_id = tweet_object['id']
            tweet = tweet_object['text']
            tweet = tweet.replace("\n", " ")
            tweet = tweet.replace("\r", " ")
            
            if label == "racism":
                label = "1"
            elif label == "sexism":
                label = "2"
            elif label == "none":
                label = "0"
                
            with open(save_text_to, "a+") as to_text_file:
                to_text_file.write(f"{tweet_id}\t{tweet}\t{label}\n")
            

def handle_waseem_with_default_classes(in_file, save_text_to):
    """
    Extract original labels from the Waseem dataset
    Args:
        in_file(String): the path to the training/test set in jsonl format of waseem
        each line is expected to contain a tweet json and label separated by tab
        save_text_to (String): the path to save the extracted tweets and lables in
    Returns:
        None
    """
    
    absolute_path, file_name = os.path.split(save_text_to)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    with open(in_file) as file_handle:
        for i, line in enumerate(file_handle, 1):
            tweet_json, label = line.split('\t')
            label = label.strip()
            tweet_object = json.loads(tweet_json)
            tweet_id = tweet_object['id']
            tweet = tweet_object['text']
            tweet = tweet.replace("\n", " ")
            tweet = tweet.replace("\r", " ")
            
            if label == "racism":
                label = "1"
            elif label == "sexism":
                label = "2"
            elif label == "both":
                label = "3"
            elif label == "neither":
                label = "0"
                
            with open(save_text_to, "a+") as to_text_file:
                to_text_file.write(f"{tweet_id}\t{tweet}\t{label}\n")


def handle_founta_with_default_classes(in_file, save_text_to):
    """
    Extract original labels from the Founta dataset. Note: we do not process spam 
    Args:
        in_file(String): the path to the training/test set in jsonl format of founta
        each line is expected to contain a tweet json and label separated by tab
        save_text_to (String): the path to save the extracted tweets and lables in
    Returns:
        None
    """
    
    absolute_path, file_name = os.path.split(save_text_to)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    with open(in_file) as file_handle:
        for i, line in enumerate(file_handle, 1):
            tweet_json, label = line.split('\t')
            label = label.strip()
            tweet_object = json.loads(tweet_json)
            tweet_id = tweet_object['id']
            tweet = tweet_object['text']
            tweet = tweet.replace("\n", " ")
            tweet = tweet.replace("\r", " ")
            
            if label == "hateful":
                label = "1"
            elif label == "abusive":
                label = "2"
            elif label == "normal":
                label = "0"
                
            with open(save_text_to, "a+") as to_text_file:
                to_text_file.write(f"{tweet_id}\t{tweet}\t{label}\n")

                
def handle_davidson_with_default_classes(in_file, save_text_to):
    """
    Extract original labels from the Davidson
    Args:
        in_file(String): the path to the training/test set in jsonl format of davidson
        each line is expected to contain a tweet json and label separated by tab
        save_text_to (String): the path to save the extracted tweets and lables in
    Returns:
        None
    """
    
    absolute_path, file_name = os.path.split(save_text_to)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    with open(in_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, line in enumerate(csv_reader):
            # Skip the column row
            if i == 0:
                continue
            
            uid = i
            tweet = line[-1].strip()
            tweet = tweet.replace("\n", " ")
            tweet = tweet.replace("\r", " ")
            label = line[-2].strip()

            if label == "0":   # Hate
                label = "1"
            elif label == "1": # Offensive
                label = "2"
            else:              # Normal
                label = "0"

            with open(save_text_to, 'a+') as to_text_file:
                to_text_file.write(f"{uid}\t{tweet}\t{label}\n")


def handle_golbeck_with_default_classes(in_file, save_text_to):
    # Send email to jgolbeck@umd.edu to have access to this dataset

    absolute_path, file_name = os.path.split(save_text_to)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    # Remove the Spam class only considering abuse, hate, and normal
    with open(in_file, encoding="ISO-8859-1") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for i, line in enumerate(csv_reader):
            # Skip the column row
            if i == 0:
                continue
                
            #print(line)
            uid = line[0]
            label = line[1]
            tweet = line[2]
            
            tweet = tweet.strip()
            tweet = tweet.strip("'")
            tweet = tweet.replace("\n", " ")
            tweet = tweet.replace("\r", " ")
            tweet = tweet.replace("\t", " ")

            if label == "H":
                label = "1"
            elif label == "N":
                label = "0"

            with open(save_text_to, 'a+') as to_text_file:
                to_text_file.write(f"{uid}\t{tweet}\t{label}\n")
                
                
def handle_hat_eval_19_with_default_classes(in_file, save_text_to):
    """
    Extract tweets from the HatEval 2019 dataset
    Data is located at: http://hatespeech.di.unito.it/hateval.html
    Args:
        in_file(String): the path to the train/dev/test set
        save_text_to (String): the path to save the extracted tweets and lables in
    Returns:
        None
    """

    absolute_path, file_name = os.path.split(save_text_to)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    # Remove the Spam class only considering abuse, hate, and normal
    with open(in_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, line in enumerate(csv_reader):
            # Skip the column row
            if i == 0:
                continue
            
            uid = line[0]
            tweet = line[1]
            tweet = tweet.replace("\n", " ")
            tweet = tweet.replace("\r", " ")
            label = line[2]

            if label == "1":
                label = "1"
            elif label == "0":
                label = "0"

            with open(save_text_to, 'a+') as to_text_file:
                to_text_file.write(f"{uid}\t{tweet}\t{label}\n")
                

def handle_offens_eval_19_with_default_classes(in_file, save_text_to, training_data=True, labels_file=None):
    """
    Processes the OffensEval19 dataset, dataset contains train and test set.
    The train set is of the format id tweet subtask_a subtask_b subtask_c
    The test set os subtask_a is of the format id label
    Extract the tweets and labels and store in another file for our model

    Data is located at: https://sites.google.com/site/offensevalsharedtask/olid
    We use only sub task A.
    NOTE: keep @USER and URL in mind during preprocessing as authors added them
    already
    Args:
        in_file(String): the file path to OffensEval_19 dataset(training or test)
        save_text_to (String): the path to save the extracted tweets and lables in
        training_data (Boolean): Whether we are processing the training set or
        the test set (requires the different label file for each sub task)
        labels_file (NoneType): the path to the test set labels file of a sub task
    Returns:
        None
    """

    absolute_path, file_name = os.path.split(save_text_to)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    if training_data:
        with open(in_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for i, line in enumerate(csv_reader):
                # Skip the column row
                if i == 0:
                    continue

                uid = line[0]
                tweet = line[1]
                tweet = tweet.replace("\n", " ")
                tweet = tweet.replace("\r", " ")
                label = line[2]
                if label != "NULL":
                    if label == "OFF":
                        label = "1"
                    elif label == "NOT":
                        label = "0"

                    with open(save_text_to, 'a+') as to_text_file:
                        to_text_file.write(f"{uid}\t{tweet}\t{label}\n")
    else:
        id_label_map = {}
        with open(labels_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, line in enumerate(csv_reader):
                uid = line[0]
                label = line[1]
                id_label_map[uid] = label

        with open(in_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for i, line in enumerate(csv_reader):
                # Skip the column row
                if i == 0:
                    continue

                uid = line[0]
                tweet = line[1]
                tweet = tweet.replace("\n", " ")
                tweet = tweet.replace("\r", " ")
                label = id_label_map[uid]
                if label == "OFF":
                    label = "1"
                elif label == "NOT":
                    label = "0"

                with open(save_text_to, 'a+') as to_text_file:
                    to_text_file.write(f"{uid}\t{tweet}\t{label}\n")
                    
                    
def handle_abus_eval_with_default_classes(in_file, save_text_to, labels_file):
    """
    Process the AbusEval dataset. AbusEval uses the same dataset as OffensEval
    with a modified annotation - Explicit/Implicit
    Data is located at: https://github.com/tommasoc80/AbuseEval
    Args:
        in_file(String): the path to the training/test set of OffensEval. If
        in_file is the test set of OffensEval, it should be testset-levela.tsv
        save_text_to (String): the path to save the extracted tweets and lables in
        labels_file(String): the path to the training/test set file containing the
        labels of the AbusEval
    Returns:
        None
    """

    absolute_path, file_name = os.path.split(save_text_to)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    # use the new labels of AbusEval
    id_label_map = {}
    with open(labels_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for i, line in enumerate(csv_reader):
            # Skip the column row
            if i == 0:
                continue

            uid = line[0]
            label = line[1]
            id_label_map[uid] = label

    # Using the dataset from OffensEval extract the tweets according to the
    # AbusEval annotation
    with open(in_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for i, line in enumerate(csv_reader):
            # Skip the column row
            if i == 0:
                continue

            uid = line[0]
            tweet = line[1]
            tweet = tweet.replace("\n", " ")
            tweet = tweet.replace("\r", " ")
            label = id_label_map[uid]
            if label == "EXP":
                label = "1"
            elif label == "IMP":
                label = "2"
            elif label == "NOTABU":
                label = "0"
            
            with open(save_text_to, 'a+') as to_text_file:
                to_text_file.write(f"{uid}\t{tweet}\t{label}\n")
            

def split_datasets(data_path, save_train_as, save_test_as):
    """
    Splits a dataset into train and test set. This function assumes there is
    a file containing text and label in each line separated by tab
    Args:
        data_path (String): path where the full dataset test set is contained
        save_train_as (String): absolute path to the training set
        save_test_as (String): absolute path to the test set
    Returns:
        None
    """

    data, labels = [], []
    seed = 42

    with open(data_path) as file_handler:
        for i, line in enumerate(file_handler):
            line = line.split("\t")
            uid = line[0].strip()
            tweet = line[1].strip()
            label = line[2].strip()
            #print(tweet, label)
            data.append((tweet, uid))
            labels.append(label)
    
    # Create test set from train
    train_data, test_data, train_label, test_label = train_test_split(
            data, labels, train_size=0.80, random_state=seed
        )

    # Write train set
    with open(save_train_as, "a+") as file_handler:
        for i, tweet in enumerate(train_data):
            tweet, uid = tweet
            label = train_label[i]
            file_handler.write(f"{uid}\t{tweet}\t{label}\n")

    # Write test set
    with open(save_test_as, "a+") as file_handler:
        for i, tweet in enumerate(test_data):
            tweet, uid = tweet
            label = test_label[i]
            file_handler.write(f"{uid}\t{tweet}\t{label}\n")
            
            
if __name__ == "__main__":
    save_path = "/project/luofeng/socbd/eokpala/new_public_datasets/"
    
    # This is the main process that works with the commented classify function above
    # rehydrate waseem_and_hovy
    save_json_to = "/home/eokpala/antiblack-research/external_datasets_rehydrated/waseem-and-hovy/waseem_and_hovy.jsonl"
    save_text_to = save_path + "waseem-and-hovy/waseem_and_hovy_with_default_classes.txt"
    handle_waseem_and_hovy_with_default_classes(save_json_to, save_text_to)
    save_processed_to = save_path + "waseem-and-hovy/waseem_and_hovy_preprocessed_with_default_classes.txt"
    preprocess(save_text_to, save_processed_to)
    save_train_to = save_path + "waseem-and-hovy/waseem_and_hovy_train_preprocessed_with_default_classes.txt"
    save_test_to = save_path + "waseem-and-hovy/waseem_and_hovy_test_preprocessed_with_default_classes.txt"
    split_datasets(save_processed_to, save_train_to, save_test_to)
    
    # rehydrate waseem
    save_json_to = "/home/eokpala/antiblack-research/external_datasets_rehydrated/waseem/waseem.jsonl"
    save_text_to = save_path + "waseem/waseem_with_default_classes.txt"
    handle_waseem_with_default_classes(save_json_to, save_text_to)
    save_processed_to = save_path + "waseem/waseem_preprocessed_with_default_classes.txt"
    preprocess(save_text_to, save_processed_to)
    save_train_to = save_path + "waseem/waseem_train_preprocessed_with_default_classes.txt"
    save_test_to = save_path + "waseem/waseem_test_preprocessed_with_default_classes.txt"
    split_datasets(save_processed_to, save_train_to, save_test_to)
    
    # rehydrate founta
    save_json_to = "/home/eokpala/antiblack-research/external_datasets_rehydrated/founta/founta.jsonl"
    save_text_to = save_path + "founta/founta_with_default_classes.txt"
    handle_founta_with_default_classes(save_json_to, save_text_to)
    save_processed_to = save_path + "founta/founta_preprocessed_with_default_classes.txt"
    preprocess(save_text_to, save_processed_to)
    save_train_to = save_path + "founta/founta_train_preprocessed_with_default_classes.txt"
    save_test_to = save_path + "founta/founta_test_preprocessed_with_default_classes.txt"
    split_datasets(save_processed_to, save_train_to, save_test_to)
    
    # Extract tweets from Davidson
    in_file = "/home/eokpala/antiblack-research/external_datasets/davidson/labeled_data.csv"
    save_text_to = save_path + "davidson/davidson_with_default_classes.txt"
    handle_davidson_with_default_classes(in_file, save_text_to)
    save_processed_to = save_path + "davidson/davidson_preprocessed_with_default_classes.txt"
    preprocess(save_text_to, save_processed_to)
    save_train_to = save_path + "davidson/davidson_train_preprocessed_with_default_classes.txt"
    save_test_to = save_path + "davidson/davidson_test_preprocessed_with_default_classes.txt"
    split_datasets(save_processed_to, save_train_to, save_test_to)
    
    # Extract tweets from Golbeck
    in_file = "/home/eokpala/antiblack-research/external_datasets/golbeck/onlineHarassmentDataset.tdf"
    save_text_to = save_path + "golbeck/golbeck_with_default_classes.txt"
    handle_golbeck_with_default_classes(in_file, save_text_to)
    save_processed_to = save_path + "golbeck/golbeck_preprocessed_with_default_classes.txt"
    preprocess(save_text_to, save_processed_to, golbeck=True)
    save_train_to = save_path + "golbeck/golbeck_train_preprocessed_with_default_classes.txt"
    save_test_to = save_path + "golbeck/golbeck_test_preprocessed_with_default_classes.txt"
    split_datasets(save_processed_to, save_train_to, save_test_to)
    
    # Extract tweets from HatEval19 training set
    in_file = "/home/eokpala/antiblack-research/external_datasets/hateval2019/hateval2019_en_train.csv"
    save_text_to = save_path + "hateval2019/hateval2019_en_train.txt"
    handle_hat_eval_19_with_default_classes(in_file, save_text_to)
    save_train_to = save_path + "hateval2019/hateval2019_en_train_preprocessed_with_default_classes.txt"
    preprocess(save_text_to, save_train_to)
    
    # Extract tweets from HatEval19 test set
    in_file = "/home/eokpala/antiblack-research/external_datasets/hateval2019/hateval2019_en_test.csv"
    save_text_to = save_path + "hateval2019/hateval2019_en_test.txt"
    handle_hat_eval_19_with_default_classes(in_file, save_text_to)
    save_test_to = save_path + "hateval2019/hateval2019_en_test_preprocessed_with_default_classes.txt"
    preprocess(save_text_to, save_test_to)
    
    # Extract tweets from HatEval19 validation set
    in_file = "/home/eokpala/antiblack-research/external_datasets/hateval2019/hateval2019_en_dev.csv"
    save_text_to = save_path + "hateval2019/hateval2019_en_dev.txt"
    handle_hat_eval_19_with_default_classes(in_file, save_text_to)
    save_val_to = save_path + "hateval2019/hateval2019_en_validation_preprocessed_with_default_classes.txt"
    preprocess(save_text_to, save_val_to)
    
    # Extract tweets from OffensEval train set
    in_file = "/home/eokpala/antiblack-research/external_datasets/offenseval2019/olid-training-v1.0.tsv"
    save_text_to = save_path + "offenseval2019/offenseval_train.txt"
    handle_offens_eval_19_with_default_classes(in_file, save_text_to, training_data=True, labels_file=None)
    save_train_to = save_path + "offenseval2019/offenseval_train_preprocessed_with_default_classes.txt"
    preprocess_offenseval(save_text_to, save_train_to)
    
    # Extract tweets from OffensEval test set
    in_file = "/home/eokpala/antiblack-research/external_datasets/offenseval2019/testset-levela.tsv"
    save_text_to = save_path + "offenseval2019/offenseval_test.txt"
    labels_file = "/home/eokpala/antiblack-research/external_datasets/offenseval2019/labels-levela.csv"
    handle_offens_eval_19_with_default_classes(in_file, save_text_to, training_data=False, labels_file=labels_file)
    save_test_to = save_path + "offenseval2019/offenseval_test_preprocessed_with_default_classes.txt"
    preprocess_offenseval(save_text_to, save_test_to)
    
    # Training set of AbusEval
    in_file = "/home/eokpala/antiblack-research/external_datasets/offenseval2019/olid-training-v1.0.tsv"
    labels_file = "/home/eokpala/antiblack-research/external_datasets/abuseval/abuseval_labels/abuseval_offenseval_train.tsv"
    save_text_to = save_path + "abuseval/abuseval_train_with_default_classes.txt"
    handle_abus_eval_with_default_classes(in_file, save_text_to, labels_file)
    save_train_to = save_path + "abuseval/abuseval_train_preprocessed_with_default_classes.txt"
    preprocess_offenseval(save_text_to, save_train_to)
    
    # Test set of AbusEval
    in_file = "/home/eokpala/antiblack-research/external_datasets/offenseval2019/testset-levela.tsv"
    labels_file = "/home/eokpala/antiblack-research/external_datasets/abuseval/abuseval_labels/abuseval_offenseval_test.tsv"
    save_text_to = save_path + "abuseval/abuseval_test_with_default_classes.txt"
    handle_abus_eval_with_default_classes(in_file, save_text_to, labels_file)
    save_test_to = save_path + "abuseval/abuseval_test_preprocessed_with_default_classes.txt"
    preprocess_offenseval(save_text_to, save_test_to)
    print("Completed.")