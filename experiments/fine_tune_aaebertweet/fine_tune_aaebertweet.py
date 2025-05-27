import pickle
import os
import sys
import torch
import time
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import logging
import logging.handlers
from _datetime import datetime as dt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix
from datasets import load_dataset, Features, ClassLabel, Value
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    get_scheduler,
    AutoTokenizer,
    RobertaTokenizer,
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    RobertaForSequenceClassification,
    BertForSequenceClassification,
    AutoModelForSequenceClassification
)
from fine_tuning_module import CustomBertModel
from fine_tuning_utils import CustomTextDataset, flat_accuracy, format_time, train
from bert_utils import (
    PATH,
    SAVE_PATH,
    NUMBER_OF_LABELS,
    hyperparameters,
    custom_bert_parameters
)


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


def main(logger):
    path = PATH
    train_dataset = path + "sampled_black_and_white_aligned_combined_with_dialect_label_train_preprocessed.txt"
    test_dataset = path + "sampled_black_and_white_aligned_combined_with_dialect_label_test_preprocessed.txt"

    # Path where the retrained model and tokenizer can be found
    model_path = "/project/luofeng/socbd/eokpala/new_retrained_aaebert/aae_bertweet/checkpoint-410500"
    tokenizer_path = model_path
    base_model = "bertweet"
    save_path = SAVE_PATH + f"{base_model}/fine_tuned_aae/"

    # Same hyperparameter as HateBERT and BERTweet
    batch_size, learning_rate, epochs = hyperparameters()
    num_labels = NUMBER_OF_LABELS
    
    classifier_criterion = custom_bert_parameters()
    classifier = CustomBertModel(model_path,
                                 num_labels)
    
    tokenizer = classifier.tokenizer
    training_data = CustomTextDataset(tokenizer, train_dataset)
    test_data = CustomTextDataset(tokenizer, test_dataset)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train(
        train_dataloader,
        test_dataloader,
        tokenizer,
        classifier,
        num_labels,
        learning_rate,
        epochs,
        save_path,
        classifier_criterion,
        logger,
        patience=3
    )
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    logger = get_logger(log_dir, get_filename())
    main(logger)
