import time
import random
import json
import csv
import os
import sys
import re
import emoji
import string
import shutil
import logging
import logging.handlers
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import cohen_kappa_score
from pprint import pprint

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


def calc_cohen_kappa_score(train_path, test_path, current_file):
    human_labels, llm_labels = [], []
    
    with open(train_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            llm_label = int(line[3])
            human_labels.append(human_label)
            llm_labels.append(llm_label)
    
    with open(test_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            llm_label = int(line[3])
            human_labels.append(human_label)
            llm_labels.append(llm_label)
            
    cohen_kappa = cohen_kappa_score(human_labels, llm_labels)
    logging.info(f'{current_file} ck score: {cohen_kappa}')


def main():
    
    # CK for GPT annotated datasets
    data_paths_train = [
        "davidson/davidson_with_default_classes_train.txt",
        "waseem/waseem_with_default_classes_train.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_train.txt",
        "founta/founta_with_default_classes_train.txt",
        "golbeck/golbeck_with_default_classes_train.txt",
        "hateval2019/hateval2019_en_train.txt",
        "offenseval2019/offenseval_train.txt",
        "abuseval/abuseval_train_with_default_classes.txt"
    ]
    data_paths_test = [
        "davidson/davidson_with_default_classes_test.txt",
        "waseem/waseem_with_default_classes_test.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_test.txt",
        "founta/founta_with_default_classes_test.txt",
        "golbeck/golbeck_with_default_classes_test.txt",
        "hateval2019/hateval2019_en_test.txt",
        "offenseval2019/offenseval_test.txt",
        "abuseval/abuseval_test_with_default_classes.txt"
    ]
    dataset_names = ['davidson', 'waseem', 'hovy', 'founta', 'golbeck', 'hateval', 'offenseval', 'abuseval']
    annotation_strategies = ["general_prompt_annotation", "few_shot_prompt_annotation", "cot_prompt_annotation"] 
    for strategy in annotation_strategies:
        data_path = f'/project/luofeng/socbd/eokpala/llm_annotation_bias/data/{strategy}/'
        logging.info(f'{strategy}: CK for GPT annotated datasets')
        for i, train_test_paths in enumerate(zip(data_paths_train, data_paths_test)):
            train_path, test_path = train_test_paths
            calc_cohen_kappa_score(data_path + train_path, data_path + test_path, dataset_names[i])
        logging.info(' ')
        
    # CK for GPT annotated datasets with dialect priming annotation
    data_paths_train = [
        "davidson/davidson_with_default_classes_train.txt",
        "founta/founta_with_default_classes_train.txt",
        "golbeck/golbeck_with_default_classes_train.txt",
        "hateval2019/hateval2019_en_train.txt",
        "offenseval2019/offenseval_train.txt",
    ]
    data_paths_test = [
        "davidson/davidson_with_default_classes_test.txt",
        "founta/founta_with_default_classes_test.txt",
        "golbeck/golbeck_with_default_classes_test.txt",
        "hateval2019/hateval2019_en_test.txt",
        "offenseval2019/offenseval_test.txt",
    ]
    dataset_names = ['davidson', 'founta', 'golbeck', 'hateval', 'offenseval']
    annotation_strategies = ["general_prompt_annotation", "few_shot_prompt_annotation", "cot_prompt_annotation"]
    for strategy in annotation_strategies:
        data_path = f'/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/{strategy}/'
        logging.info(f'{strategy}: CK for GPT annotated datasets using dialect priming')
        for i, train_test_paths in enumerate(zip(data_paths_train, data_paths_test)):
            train_path, test_path = train_test_paths
            calc_cohen_kappa_score(data_path + train_path, data_path + test_path, dataset_names[i])
        logging.info(' ')
    
    # CK for LLaMa annotated datasets
    data_paths_train = [
        "davidson/davidson_with_default_classes_train.txt",
        "waseem/waseem_with_default_classes_train.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_train.txt",
        "founta/founta_with_default_classes_train.txt",
        "golbeck/golbeck_with_default_classes_train.txt",
        "hateval2019/hateval2019_en_train.txt",
        "offenseval2019/offenseval_train.txt",
        "abuseval/abuseval_train_with_default_classes.txt"
    ]
    data_paths_test = [
        "davidson/davidson_with_default_classes_test.txt",
        "waseem/waseem_with_default_classes_test.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_test.txt",
        "founta/founta_with_default_classes_test.txt",
        "golbeck/golbeck_with_default_classes_test.txt",
        "hateval2019/hateval2019_en_test.txt",
        "offenseval2019/offenseval_test.txt",
        "abuseval/abuseval_test_with_default_classes.txt"
    ]
    dataset_names = ['davidson', 'waseem', 'hovy', 'founta', 'golbeck', 'hateval', 'offenseval', 'abuseval']
    annotation_strategies = ["general_prompt_annotation"] 
    for strategy in annotation_strategies:
        data_path = f'/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama/{strategy}/'
        logging.info(f'{strategy}: CK for LLaMa annotated datasets')
        for i, train_test_paths in enumerate(zip(data_paths_train, data_paths_test)):
            train_path, test_path = train_test_paths
            calc_cohen_kappa_score(data_path + train_path, data_path + test_path, dataset_names[i])
        logging.info(' ')
    
    # CK for LLaMa annotated datasets using dialect priming
    data_paths_train = [
        "davidson/davidson_with_default_classes_train.txt",
        "founta/founta_with_default_classes_train.txt",
        "golbeck/golbeck_with_default_classes_train.txt",
        "hateval2019/hateval2019_en_train.txt",
        "offenseval2019/offenseval_train.txt",
    ]
    data_paths_test = [
        "davidson/davidson_with_default_classes_test.txt",
        "founta/founta_with_default_classes_test.txt",
        "golbeck/golbeck_with_default_classes_test.txt",
        "hateval2019/hateval2019_en_test.txt",
        "offenseval2019/offenseval_test.txt",
    ]
    dataset_names = ['davidson', 'founta', 'golbeck', 'hateval', 'offenseval']
    annotation_strategies = ["general_prompt_annotation"]
    for strategy in annotation_strategies:
        data_path = f'/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/{strategy}/'
        logging.info(f'{strategy}: CK for LLaMa annotated datasets using dialect priming')
        for i, train_test_paths in enumerate(zip(data_paths_train, data_paths_test)):
            train_path, test_path = train_test_paths
            calc_cohen_kappa_score(data_path + train_path, data_path + test_path, dataset_names[i])
        logging.info(' ')
        
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    
    main()