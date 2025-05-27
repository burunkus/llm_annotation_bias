#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import time
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
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
sys.path.insert(0, '/home/eokpala/llm_annotation_bias/experiments/fine_tune_model')
from fine_tuning_module import CustomBertModel
from fine_tuning_utils import CustomTextDataset, flat_accuracy, format_time, train
from bert_utils import (
    hyperparameters, 
    NUMBER_OF_LABELS_MAP,
    PATH,
    custom_bert_parameters,
    SAVE_PATH,
    TASK,
    PROMPT_TECHNIQUE,
    CLASS_TYPE
)

def main():
    path = PATH
    train_dataset = path + "abuseval/abuseval_train_with_default_classes_preprocessed_dialect_label.txt"
    test_dataset = path + "abuseval/abuseval_test_with_default_classes_preprocessed_dialect_label.txt"

    # Path where the pre-trained model and tokenizer can be found
    model_path = "vinai/bertweet-base"
    tokenizer_path = "vinai/bertweet-base" 
    dataset_name = "abuseval" # Change to the name of the dataset - hateval, waseem etc this is for folder naming purposes
    base_model = "BERTweet" # Change to BERT/RoBERTa if thats the model I am fine-tunning on
    save_path = SAVE_PATH + f"bertweet/{base_model}_fine_tuned_on_{dataset_name}/"

    batch_size, learning_rate, epochs = hyperparameters()
    directory, file_name = os.path.split(train_dataset)
    num_labels = NUMBER_OF_LABELS_MAP[directory.split('/')[-1]]
    
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
        classifier_criterion
    )

if __name__ == "__main__":
    main()
