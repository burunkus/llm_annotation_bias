#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import torch
import time
import numpy as np
import random
import datetime
import csv
import sys
from scipy import stats 
from _datetime import datetime as dt
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
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
from fine_tuning_module import CustomBertModel, OffensiveNetwork
from fine_tuning_utils import CustomTextDataset, flat_accuracy, format_time, train
from bert_utils import (
    hyperparameters, 
    NUMBER_OF_LABELS_MAP,
    PATH,
    custom_bert_parameters,
    SAVE_PATH,
    TASK,
    PROMPT_TECHNIQUE,
    WITH_REGULARIZATION,
    dialect_hyperparameters
)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class CustomTextDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 padding="max_length",
                 truncation=True,
                 max_length=100
                 ):

        """
        Generate a single example and its label from data_path
        Args:
            tokenizer (Object): BERT/RoBERTa tokenization object
            data_path (String): Absolute path to the train/test dataset
            padding (String): How to padding sequences, defaults to "max_lenght"
            truncation (Boolean): Whether to truncate sequences, defaults to True
            max_length (Int): The maximum length of a sequence, sequence longer
            than max_length will be truncated and those shorter will be padded
        Retruns:
            dataset_item (Tuple): A tuple of tensors - tokenized text, attention mask and labels
        """

        if not os.path.exists(data_path):
            raise ValueError(f"Input file {data_path} does not exist")

        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        if file_extension == "txt":
            with open(data_path, encoding="utf-8") as file_handler:
                tweets = []
                for line in file_handler:
                    tweet = line.strip()
                    tweets.append(tweet)

            tokenized_tweet = tokenizer(tweets,
                                        padding=padding,
                                        truncation=truncation,
                                        max_length=max_length)

            self.examples = tokenized_tweet['input_ids']
            self.attention_masks = tokenized_tweet['attention_mask']
            self.token_type_ids = tokenized_tweet['token_type_ids']


    def __len__(self):
        return len(self.examples)

    
    def __getitem__(self, index):
        dataset_item = (torch.tensor(self.examples[index]),
                        torch.tensor(self.attention_masks[index])
                       )
        return dataset_item


def assess_bias(black_aligned_dataloader, 
                white_aligned_dataloader,
                model, 
                num_labels, 
                dataset_name,
                dataset_labels,
                with_regularization=False):
    """
    Determines if BERT models including BlackBERT assigns negative class (hate) 
    to black-aligned tweets than white-aligned tweets
    Args:
        black_aligned_dataloader (Dataset): A PyTorch iterable object through the black_aligned tweets
        white_aligned_dataloader (Dataset): A PyTorch iterable object through the white_aligned tweets
        tokenizer (Object): BERT/RoBERTa tokenization object
        model_path (String): the directory where the model is located 
        num_labels (int): the number of classes 
    Returns:
        None
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    class_probabilities_white_aligned = {}
    for batch in white_aligned_dataloader:
        # Add batch to GPU/CPU
        batch = tuple(pt_tensor.to(device) for pt_tensor in batch)

        # Unpack the inputs from our dataloader
        batch_input_ids, batch_input_mask = batch
        batch_labels = None
        batch = (batch_input_ids, batch_input_mask, batch_labels)

        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits
        
        # Get probabilities
        predictions = nn.functional.softmax(logits, dim=-1)
        
        # Move probabilities to CPU
        predictions = predictions.detach().cpu().numpy()

        # 0 is non-hate, other classes (1, 2) are either hate, offensive or abusive depending on dataset. see rehydrate_with_default_classes.py
        for curr_class in range(num_labels): 
            for example_prediction in predictions:
                if curr_class in class_probabilities_white_aligned:
                    class_probabilities_white_aligned[curr_class].append(example_prediction[curr_class])
                else:
                    class_probabilities_white_aligned[curr_class] = []
                
    
    # Vector of probabilities
    for curr_class, probability_vector in class_probabilities_white_aligned.items(): 
        class_probabilities_white_aligned[curr_class] = np.asarray(probability_vector)
    
    # black-aligned    
    class_probabilities_black_aligned = {}
    for batch in black_aligned_dataloader:
        # Add batch to GPU/CPU
        batch = tuple(pt_tensor.to(device) for pt_tensor in batch)

        # Unpack the inputs from our dataloader
        batch_input_ids, batch_input_mask = batch
        batch_labels = None
        batch = (batch_input_ids, batch_input_mask, batch_labels)

        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits
        
        # Get probabilities
        predictions = nn.functional.softmax(logits, dim=-1)
        
        # Move probabilities to CPU
        predictions = predictions.detach().cpu().numpy()
        
        # 0 is non-hate, other classes (1, 2) are either hate, offensive or abusive depending on dataset
        for curr_class in range(num_labels): 
            for example_prediction in predictions:
                if curr_class in class_probabilities_black_aligned:
                    class_probabilities_black_aligned[curr_class].append(example_prediction[curr_class])
                else:
                    class_probabilities_black_aligned[curr_class] = []
        
    # Vector of probabilities
    for curr_class, probability_vector in class_probabilities_black_aligned.items(): 
        class_probabilities_black_aligned[curr_class] = np.asarray(probability_vector)
    
    for curr_class in range(num_labels):
        # Calculate the proportion of tweets assigned to each class for each group
        if curr_class != 0: # Skip non-hate
            print(f"p_{curr_class}_white_hat: {np.mean(class_probabilities_white_aligned[curr_class]):.3f}")
            print(f"p_{curr_class}_black_hat: {np.mean(class_probabilities_black_aligned[curr_class]):.3f}")
        
            # Calculate racial tendency on each class
            print(f"p_{curr_class}_black_hat/p_{curr_class}_white_hat:\
            {(np.mean(class_probabilities_black_aligned[curr_class])/np.mean(class_probabilities_white_aligned[curr_class])):.3f}")
            
            # Perform t-test and obtain the t and p values
            curr_class_ttest_result = stats.ttest_ind(class_probabilities_black_aligned[curr_class],\
                                                      class_probabilities_white_aligned[curr_class])
            print(f"class {curr_class} t: {(curr_class_ttest_result.statistic):.3f}, class {curr_class} p: {curr_class_ttest_result.pvalue}")
    
    result = {}
    for curr_class in range(num_labels):
        data = []
        if curr_class != 0: # Skip non-hate
            p_i_white_hat = np.mean(class_probabilities_white_aligned[curr_class])
            p_i_black_hat = np.mean(class_probabilities_black_aligned[curr_class])
            racial_tendency = f"{(p_i_black_hat/p_i_white_hat):.3f}"
            curr_class_ttest_result = stats.ttest_ind(class_probabilities_black_aligned[curr_class],\
                                                  class_probabilities_white_aligned[curr_class])
            t = f"{(curr_class_ttest_result.statistic):.3f}"
            p = curr_class_ttest_result.pvalue
            
            if p < 0.001:
                p = "***"
            elif p > 0.05:
                p = ""
            
            p_i_white_hat = f"{(p_i_white_hat):.3f}"
            p_i_black_hat = f"{(p_i_black_hat):.3f}"
            data.append(dataset_name)
            data.append(dataset_labels[dataset_name][curr_class])
            data.append(p_i_black_hat)
            data.append(p_i_white_hat)
            data.append(t)
            data.append(p)
            data.append(racial_tendency)
            result[curr_class] = data
            
    return result


def save_result_to_csv(save_as, results):
    
    with open(save_as + '.csv', mode='w') as file_handle:
        field_names = ['Dataset', 'Class', 'p_iblack ̂', 'p_iwhite ̂', 't', 'p', 'p_iblack ̂/p_iwhite ̂']
        file_handle = csv.writer(file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_handle.writerow(field_names)
        
        datasets = ["abuseval", "offenseval", "hateval", "davidson", "founta", "hovy", "waseem", "golbeck"]
        for dataset in datasets:
            if dataset in results:
                dataset_result = results[dataset]
                for curr_class, data in dataset_result.items():
                    file_handle.writerow(data)
    
    path = "/home/eokpala/llm_annotation_bias/experiments/assess_bias/"
    # Move a copy to SAVE_PATH
    shutil.copy(save_as + '.csv', SAVE_PATH + save_as + '.csv')

    
def main():
    
    black_aligned_dataset = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_preprocessed.txt"
    white_aligned_dataset = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_white_aligned_preprocessed.txt"
    
    classifiers = {"bert": "bert-base-uncased", 
                   "bertweet": "vinai/bertweet-base", 
                   "hate_bert": "../../../../../../project/luofeng/socbd/eokpala/hate_bert"
                  }
    
    path = SAVE_PATH
    num_labels_map = {
            "abuseval": 3,
            "founta": 3,
            "hateval": 2,
            "waseem": 4,
            "davidson": 3,
            "golbeck": 2,
            "offenseval": 2,
            "hovy": 3
        }
    
    dataset_labels = {
        "abuseval": [None, "Explicit", "Implicit"],
        "founta": [None, "Hate", "Abuse"],
        "hateval": [None, "Hate"],
        "waseem": [None, "Racisim", "Sexism", "Racisim & Sexism"],
        "davidson": [None, "Hate", "Offensive"],
        "golbeck": [None, "Harassment"],
        "offenseval": [None, "Offensive"],
        "hovy": [None, "Racism", "Sexism"]
    }
    
    for classifier_name, base_model_config_path in classifiers.items():
        print(f"\n{classifier_name}")
        print("-------------------------------------")
            
        classifier_name = path + classifier_name
        model_name = classifier_name.split('/')[-1]
        model_results = {}
        
        fine_tuned_models = sorted(os.listdir(classifier_name))
        if fine_tuned_models[0] == ".ipynb_checkpoints":
            fine_tuned_models = fine_tuned_models[1:]
        
        # Path where the pre-trained model and tokenizer can be found
        for fine_tuned_model in fine_tuned_models:
            if 'logs' in fine_tuned_model or 'png' in fine_tuned_model:
                continue

            print(f"\nEvaluating {fine_tuned_model}")
            model_path = classifier_name + "/" + fine_tuned_model
            tokenizer_path = base_model_config_path
            dataset_model_name = fine_tuned_model.split('_')[-1]

            batch_size = 32
            num_labels = num_labels_map[fine_tuned_model.split('_')[-1]]
            
            # We only use the CustomBertModel imported from fine_tune_model 
            classifier_criterion = custom_bert_parameters()
            model = CustomBertModel(base_model_config_path,
                                    num_labels)
            
            model_location = model_path + '/' + fine_tuned_model + '.pth'
            model.load_state_dict(torch.load(model_location))
            tokenizer = model.tokenizer

            # tokenize and batch dataset
            black_aligned_data = CustomTextDataset(tokenizer, black_aligned_dataset)
            white_aligned_data = CustomTextDataset(tokenizer, white_aligned_dataset)

            black_aligned_dataloader = DataLoader(black_aligned_data, batch_size=batch_size, shuffle=True)
            white_aligned_dataloader = DataLoader(white_aligned_data, batch_size=batch_size, shuffle=True)

            result = assess_bias(black_aligned_dataloader, 
                                 white_aligned_dataloader,
                                 model, 
                                 num_labels, 
                                 dataset_model_name, 
                                 dataset_labels)
            model_results[dataset_model_name] = result
        save_result_to_csv(model_name, model_results)
        # Skeleton of model_results and result
        '''
        model_results = {
            abuseval: {
                1: [abuseval, explicit, p_i_black_hat, p_i_white_hat, t, p, ratio],
                2: [abuseval, implicit, p_i_black_hat, p_i_white_hat, t, p, ratio]
                },
            ...
        }
        '''

if __name__ == "__main__":
    main()