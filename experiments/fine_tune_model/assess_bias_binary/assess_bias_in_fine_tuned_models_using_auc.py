 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
import torch
import time
import numpy as np
import random
import datetime
import csv
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint
from scipy import stats 
from _datetime import datetime as dt
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_auc_score
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
from collections import defaultdict
sys.path.insert(0, '/home/eokpala/llm_annotation_bias/experiments/fine_tune_model')
from fine_tuning_module import CustomBertModel
from fine_tuning_utils import flat_accuracy, format_time, train
from bert_utils import (
    hyperparameters, 
    NUMBER_OF_LABELS_MAP,
    PATH,
    custom_bert_parameters,
    SAVE_PATH,
    TASK,
    PROMPT_TECHNIQUE,
    CLASS_TYPE,
    WITH_DIALECT_PRIMING,
    WITH_REGULARIZATION,
    dialect_hyperparameters,
    COMBINED_DATA
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

        self.labels = []
        self.dialect_labels = []
        self.uids = []

        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        if file_extension == "txt":
            with open(data_path, encoding="utf-8") as file_handler:
                tweets = []
                for line in file_handler:
                    if PROMPT_TECHNIQUE == None:
                        # We assess bias using human labels and GPT labels. Human label bias assesment must have no prompting technique
                        assert TASK == "human label" and PROMPT_TECHNIQUE == None
                        uid, tweet, human_label, human_label_binary, dialect_label = line.split("\t")
                        human_label = human_label.strip()
                        human_label_binary = human_label_binary.strip()
                        
                    else:
                        uid, tweet, human_label, gpt_label, gpt_label_binary, dialect_label = line.split("\t")
                        gpt_label = gpt_label.strip()
                        gpt_label_binary = gpt_label_binary.strip()
                    
                    tweet = tweet.strip()
                    uid = uid.strip()
                    dialect_label = dialect_label.strip()
                    tweets.append(tweet)
                        
                    if TASK == "human label" and CLASS_TYPE == "multi class":
                        self.labels.append(int(human_label))
                    elif TASK == "human label" and CLASS_TYPE == "binary":
                        self.labels.append(int(human_label_binary))
                    elif TASK == "gpt label" and CLASS_TYPE == "multi class":
                        self.labels.append(int(gpt_label))
                    elif TASK == "gpt label" and CLASS_TYPE == "binary":
                        self.labels.append(int(gpt_label_binary))
                    
                    self.uids.append(uid)
                    self.dialect_labels.append(int(dialect_label))
            print(f'labels: {len(self.labels)}, tweets: {len(tweets)}')
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
                        torch.tensor(self.attention_masks[index]),
                        torch.tensor(self.labels[index]),
                        torch.tensor(self.dialect_labels[index])
                    )
        return dataset_item


def plot_score_per_dialect(final_results):
    """
    Code adapted from https://github.com/hate-alert/HateXplain/blob/master/Bias_Calculation_NB.ipynb
    """
    
    path = "/home/eokpala/llm_annotation_bias/experiments/fine_tune_model/assess_bias_binary/"
    for llm, llm_data in final_results.items():
        metrics_list = ['subgroup', 'bpsn', 'bnsp']
        for metric in metrics_list:
            tuple_dialect = []
            for each_model, model_data in llm_data.items():
                for dialect, dialect_rocauc in model_data[metric].items():
                    tuple_dialect.append((each_model, dialect, dialect_rocauc))
            
            # We have all values for "metric" for all models
            df_dialect_score = pd.DataFrame(tuple_dialect, columns=['Model', 'Dialect', 'AUCROC'])
            # Plot 
            ax = sns.catplot(
                x='Dialect',
                y='AUCROC',
                hue='Model',
                data=df_dialect_score,
                legend=True,
                kind='bar'
            )
            ax.set(ylim=(0.3, 1.0))
            ax.set_xticklabels(rotation=45, size=13, horizontalalignment='right')
            handles = ax._legend_data.values()
            labels = ax._legend_data.keys()
            ax.fig.legend(handles=handles, labels=labels, loc='upper right', ncol=4)
            ax.fig.subplots_adjust(top=0.92)
            ax.set(xlabel="")
            
            # Save image
            name = llm + '_' + metric + '.pdf'
            save_as = path + 'figures/' + name
            plt.savefig(save_as, dpi=300, transparent=True, bbox_inches='tight')
    # Move a copy to SAVE_PATH
    shutil.copytree(path + 'figures/', SAVE_PATH + 'figures/', dirs_exist_ok=True)


def plot_score_by_dialect_by_dataset(final_results, add_bar_labels=False):
    """
    Code adapted from https://github.com/hate-alert/HateXplain/blob/master/Bias_Calculation_NB.ipynb
     
    For each of the dataset and for each AUC metric, plot the AUC metric score of each LLM per dialect. 
    """
    
    path = "/home/eokpala/llm_annotation_bias/experiments/fine_tune_model/assess_bias_binary/"
    if WITH_DIALECT_PRIMING:
        if COMBINED_DATA:
            datasets = ['founta', 'hateval', 'davidson', 'golbeck', 'offenseval', 'data'] # 'data' is for the combined data
        else:
            datasets = ['founta', 'hateval', 'davidson', 'golbeck', 'offenseval']
    else:
        if COMBINED_DATA:
            datasets = ['abuseval', 'founta', 'hateval', 'waseem', 'davidson', 'golbeck', 'offenseval', 'data'] # 'data' is for the combined data
        else:
            datasets = ['abuseval', 'founta', 'hateval', 'waseem', 'davidson', 'golbeck', 'offenseval']
    
    for dataset in datasets:
        metrics = ['subgroup', 'bpsn', 'bnsp']
        for metric in metrics:
            tuple_dialect = []
            for each_llm, model_result in final_results.items():
                if each_llm == 'bert':
                    each_llm = 'BERT'
                elif each_llm == 'bertweet':
                    each_llm = 'BERTweet'
                elif each_llm == 'hate_bert':
                    each_llm = "HateBERT"
                elif each_llm == 'bertweet_regularization':
                    each_llm = "BERTweet+Cos"
                
                if dataset not in model_result:
                    continue 
                
                for each_dialect, auc_value in model_result[dataset][metric].items():
                    tuple_dialect.append((each_llm, each_dialect.upper(), auc_value))
                    
            # Plot the "dataset" "metric"
            df_dialect_score = pd.DataFrame(tuple_dialect, columns=['Model', 'Dialect', 'AUCROC'])
            # Plot 
            ax = sns.catplot(
                x='Dialect',
                y='AUCROC',
                hue='Model',
                data=df_dialect_score,
                kind='bar',
                legend='auto',
                legend_out=False
            )
            ax.set(ylim=(0.3, 1.0))
            ax.set_xticklabels(rotation=45, size=13, horizontalalignment='right')
            handles = ax._legend_data.values()
            labels = ax._legend_data.keys()
            #ax.figure.legend(handles=handles, labels=labels, loc='upper right', ncol=3)
            sns.move_legend(
                ax, "upper right",
                ncol=3, title=None, frameon=False,
            )
            # ax.legend.set_title("Body mass (g)")
            ax.figure.subplots_adjust(top=0.92)
            ax.set(xlabel="")
            
            # Save image
            name = dataset + '_' + metric + '.pdf'
            if add_bar_labels:
                # Add bar labels
                # extract the matplotlib axes_subplot objects from the FacetGrid
                ax = ax.facet_axis(0, 0) 
                # iterate through the axes containers
                for container in ax.containers:
                    labels = [f'{bar.get_height():.2f}' for bar in container]
                    ax.bar_label(container, labels=labels, label_type='edge')
                save_as = path + 'figures_with_bar_labels/' + name
            else:
                save_as = path + 'figures/' + name
            
            plt.savefig(save_as, dpi=300, transparent=True, bbox_inches='tight')
           
            
    # Move a copy to SAVE_PATH
    if add_bar_labels:
        shutil.copytree(path + 'figures_with_bar_labels/', SAVE_PATH + 'figures_with_bar_labels/', dirs_exist_ok=True)
    else:
        shutil.copytree(path + 'figures/', SAVE_PATH + 'figures/', dirs_exist_ok=True)
    

def get_bias_evaluation_output(dataloader,  
                               model,
                               dataset_name,
                               with_regularization=False):
    """
    Determines if BERT models including BlackBERT assigns negative class (hate) 
    to black-aligned tweets than white-aligned tweets
    Args:
        dataloader: The tokenized test set
        model: The fine-tuned model being assessed for bias
        dataset_name: The fine-tuned model name. It is the name of the dataset "model" was fine-tuned on
    Returns:
        None
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Collect global values for calculating roc-auc scores
    global_true_labels = []
    global_predicted_probs = []
    global_predicted_labels = []
    global_dialect_labels = []
    
    for batch in dataloader:
        # Add batch to GPU/CPU
        batch = tuple(pt_tensor.to(device) for pt_tensor in batch)

        # Unpack the inputs from our dataloader
        batch_input_ids, batch_input_mask, batch_input_labels, batch_input_dialect_labels = batch
        batch_token_type_ids = None 
        batch_labels = batch_input_labels
        batch_uids = None
        batch_dialect_labels = batch_input_dialect_labels
        batch = (batch_input_ids, batch_input_mask, None)  

        with torch.no_grad():
            classifier_output = model(batch)

        logits = classifier_output.logits
        
        true_labels = batch_labels.to('cpu').numpy()
        dialect_labels = batch_dialect_labels.to('cpu').numpy()
        
        # Get prediction probabilities for roc-auc cal
        predictions = nn.functional.softmax(logits, dim=-1)
        predictions = predictions.detach().cpu().numpy()
        predicted_labels = np.argmax(predictions, axis=1).flatten() 
        true_labels = true_labels.flatten()
        dialect_labels = dialect_labels.flatten()
        
        global_true_labels.extend(true_labels)
        global_predicted_probs.extend(predictions)
        global_predicted_labels.extend(predicted_labels)
        global_dialect_labels.extend(dialect_labels)
    
    print(f'True labels size: {len(global_true_labels)}, predicted labels size: {len(global_predicted_probs)}, prob[0].size: {len(global_predicted_probs[0])}, dialect labels size: {len(global_dialect_labels)}')
    print(f'dialect labels: {global_dialect_labels}')
    dialect_list = [1, 0] # [AAE, SAE]
    dialect_list_str = {1: 'aae', 0: 'sae'}
    metrics_list = ['subgroup', 'bpsn', 'bnsp']
    
    result = defaultdict(lambda: defaultdict(dict))
    for metric in metrics_list:
        print(f'Processing {metric}')
        for dialect in dialect_list:
            print(f'Processing {dialect_list_str[dialect]}')
            # Store the true and predicted labels for "dialect"
            ground_labels = [] 
            predicted_probs = [] 
            if metric == 'subgroup':
                # Go through the dataset and extract the true label and predicted prob of tweets that are "dialect"
                for index, true_dialect_label in enumerate(global_dialect_labels):
                    if true_dialect_label == dialect:
                        # Extract tweets that are "dialect" for "metric" calculation
                        ground_labels.append(global_true_labels[index])
                        # Use the class probabilities and not the predicted label
                        # Note: From (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - In the binary case, the probability estimates correspond to the probability of the class with the greater label, i.e. estimator.classes_[1]
                        predicted_probs.append(global_predicted_probs[index][1])
                print(f'ground labels: {ground_labels} {len(ground_labels)}, predicted_probs: {predicted_probs} {len(predicted_probs)}')
                print(f'subgroup ground labels: {ground_labels} {len(ground_labels)}')
            elif metric == 'bpsn':
                # Go through the dataset and extract non-hateful tweets that are "dialect" and hateful tweets that are not "dialect"
                for index, true_dialect_label in enumerate(global_dialect_labels):
                    if true_dialect_label == dialect:
                        if global_true_labels[index] == 0: #non-hate
                            ground_labels.append(global_true_labels[index])
                            predicted_probs.append(global_predicted_probs[index][1])
                    else:
                        if global_true_labels[index] == 1: #hate
                            ground_labels.append(global_true_labels[index])
                            predicted_probs.append(global_predicted_probs[index][1])
                print(f'ground labels: {ground_labels} {len(ground_labels)}, predicted_probs: {predicted_probs} {len(predicted_probs)}')
                print(f'bpsn ground labels: {ground_labels} {len(ground_labels)}')
            elif metric == 'bnsp':
                # Go through the dataset and extract hateful tweets that are "dialect" and non-hateful tweets that are not "dialect"
                for index, true_dialect_label in enumerate(global_dialect_labels):
                    if true_dialect_label == dialect:
                        if global_true_labels[index] == 1:
                            ground_labels.append(global_true_labels[index])
                            predicted_probs.append(global_predicted_probs[index][1])
                    else:
                        if global_true_labels[index] == 0:
                            ground_labels.append(global_true_labels[index])
                            predicted_probs.append(global_predicted_probs[index][1])
                print(f'ground labels: {ground_labels} {len(ground_labels)}, predicted_probs: {predicted_probs} {len(predicted_probs)}')
                print(f'bnsp ground labels: {ground_labels} {len(ground_labels)}')
            print('Before roc-auc cal:')
            print(f'ground labels: {ground_labels} {len(ground_labels)}, predicted_probs: {predicted_probs} {len(predicted_probs)}')
            print(f'ground labels: {ground_labels} {len(ground_labels)}')
            roc_output_value = roc_auc_score(ground_labels, predicted_probs)
            result[metric][dialect_list_str[dialect]] = roc_output_value
    
    return result


def process_assess_bias(with_regularization=False):
    classifiers = {f"bert": "bert-base-uncased", 
                    f"bertweet": "vinai/bertweet-base", 
                    f"hate_bert": "../../../../../../project/luofeng/socbd/eokpala/hate_bert"
                    }
    
    path = SAVE_PATH
    num_labels_map = {
            "abuseval": 2,
            "founta": 2,
            "hateval": 2,
            "waseem": 2,
            "davidson": 2,
            "golbeck": 2,
            "offenseval": 2,
            "hovy": 2,
            "data": 2 # The combine data
    }
    
    test_dataset_paths = {
        "abuseval": PATH + "abuseval/abuseval_test_with_default_classes_preprocessed_dialect_label.txt",
        "founta": PATH + "founta/founta_with_default_classes_test_preprocessed_dialect_label.txt",
        "hateval": PATH + "hateval2019/hateval2019_en_test_preprocessed_dialect_label.txt",
        "waseem": PATH + "waseem/waseem_with_default_classes_test_preprocessed_dialect_label.txt",
        "davidson": PATH + "davidson/davidson_with_default_classes_test_preprocessed_dialect_label.txt",
        "golbeck": PATH + "golbeck/golbeck_with_default_classes_test_preprocessed_dialect_label.txt",
        "offenseval": PATH + "offenseval2019/offenseval_test_preprocessed_dialect_label.txt",
        "hovy": PATH + "waseem-and-hovy/waseem_and_hovy_with_default_classes_test_preprocessed_dialect_label.txt",
        "data": PATH + "all_data/all_data_train_preprocessed_dialect_label.txt" # For the combined data
    }
    
    all_results = {}
    # Go through each llm
    for classifier_name, base_model_config_path in classifiers.items():
        print(f"\n{classifier_name}")
        print("-------------------------------------")
        
        classifier_name = path + classifier_name
        model_name = classifier_name.split('/')[-1]
        model_results = {}
        
        # Get the models fine-tuned on the llm "classifier_name"
        fine_tuned_models = sorted(os.listdir(classifier_name))
        if fine_tuned_models[0] == ".ipynb_checkpoints":
            fine_tuned_models = fine_tuned_models[1:]

        # Go through each of the models produced by fine-tuning the llm "classifier_name" on each dataset
        for fine_tuned_model in fine_tuned_models:
            if fine_tuned_model == 'logs':
                continue
                
            # Skip W&H in general prompt analysis
            skip = {'BERT_fine_tuned_on_waseem_and_hovy', 'BERTweet_fine_tuned_on_waseem_and_hovy', 'HateBERT_fine_tuned_on_waseem_and_hovy'}
            if PROMPT_TECHNIQUE == 'general_prompt_annotation' and fine_tuned_model in skip:
                continue
            
            print(f"\nEvaluating {fine_tuned_model}")
            model_path = classifier_name + "/" + fine_tuned_model
            tokenizer_path = base_model_config_path
            dataset_model_name = fine_tuned_model.split('_')[-1]

            batch_size = 32
            num_labels = num_labels_map[fine_tuned_model.split('_')[-1]]

            # Load each pre-trained model
            model = CustomBertModel(base_model_config_path,
                                    num_labels)
            tokenizer = model.tokenizer
                
            model_location = model_path + '/' + fine_tuned_model + '.pth'
            model.load_state_dict(torch.load(model_location))
            
            # Get the test dataset
            test_dataset_path = test_dataset_paths[dataset_model_name]
            data = CustomTextDataset(tokenizer, test_dataset_path)
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

            # Assess bias
            result = get_bias_evaluation_output(dataloader,
                                                model,
                                                dataset_model_name
                                                )

            model_results[dataset_model_name] = result
        
        # Store the result of each llms (BERT/BERTweet/HateBERT)
        all_results[model_name] = model_results
        
    # Combine per-dialect bias aucs
    power_value = -5
    final_results = copy.deepcopy(all_results)
    dialect_list = ['aae', 'sae']
    num_dialects = len(dialect_list)

    for llm, llm_data in all_results.items():
        for each_model, model_data in llm_data.items():
            for each_metric, dialect_data in model_data.items():
                temp_value = []
                for each_dialect, dialect_rocauc in dialect_data.items():
                    temp_value.append(pow(dialect_rocauc, power_value))
                final_results[llm][each_model]['gmb-'+each_metric] = pow(np.sum(temp_value)/num_dialects, 1/power_value)
    
    return final_results
    """ return skeleton of final_results. all_results does not include "gmb-subgroup/bpsn/bnsp"
    {
     "bert": {
            "abuseval": {
                "subgroup": {
                        "AAE": value,
                        "SAE": value
                      },
                "bpsn": {
                        "AAE": value,
                        "SAE": value
                    },
                "bnsp": {
                        "AAE": value,
                        "SAE": value
                    },
                "gmb-subgroup": value,
                "gmb-bpsn": value,
                "gmb-bnsp": value
            },
            "founta": {
                "subgroup": {
                        "AAE": value,
                        "SAE": value
                      },
                "bpsn": {
                        "AAE": value,
                        "SAE": value
                    },
                "bnsp": {
                        "AAE": value,
                        "SAE": value
                    },
                "gmb-subgroup": value,
                "gmb-bpsn": value,
                "gmb-bnsp": value
            },
            ...
        },
    "bertweet": {
            ...
        },
    "hate_bert": {
            ...
        }
    }
    """ 
    
    
def save_result_to_csv(results):
    
    columns = ['sub-aae', 'bpsn-aae', 'bnsp-aae', 'sub-sae', 'bpsn-sae', 'bnsp-sae', 'gmb-sub', 'gmb-bpsn', 'gmb-bnsp']
    all_metrics = ['subgroup', 'bpsn', 'bnsp', 'gmb-subgroup', 'gmb-bpsn', 'gmb-bpsn']
    
    llm_names = ['bert', 'bertweet', 'hate_bert']
    for llm in llm_names:
        save_as = f'{llm}_auc_metrics'
        with open(save_as + '.csv', mode='w') as file_handle:
            file_handle = csv.writer(file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for each_fine_tuned_model, model_result in results[llm].items():
                field_names = [each_fine_tuned_model.upper()]
                file_handle.writerow(field_names)
                file_handle.writerow(columns)

                row = []
                row.append(model_result['subgroup']['aae'])
                row.append(model_result['bpsn']['aae'])
                row.append(model_result['bnsp']['aae'])
                row.append(model_result['subgroup']['sae'])
                row.append(model_result['bpsn']['sae'])
                row.append(model_result['bnsp']['sae'])
                row.append(model_result['gmb-subgroup'])
                row.append(model_result['gmb-bpsn'])
                row.append(model_result['gmb-bpsn'])
                file_handle.writerow(row)

                # Give some space
                file_handle.writerow([])
    
        path = "/home/eokpala/llm_annotation_bias/experiments/fine_tune_model/assess_bias_binary/"
        # Move a copy to SAVE_PATH
        shutil.copy(path + save_as + '.csv', SAVE_PATH + save_as + '.csv')


def save_result_to_csv_by_dataset(final_results):
    
    columns = ['sub-aae', 'bpsn-aae', 'bnsp-aae', 'sub-sae', 'bpsn-sae', 'bnsp-sae', 'gmb-sub', 'gmb-bpsn', 'gmb-bnsp']
    
    # Save by dataset
    if WITH_DIALECT_PRIMING:
        datasets = ['founta', 'hateval', 'davidson', 'golbeck', 'offenseval']
    else:
        datasets = ['abuseval', 'founta', 'hateval', 'waseem', 'davidson', 'golbeck', 'offenseval']
        
    for dataset in datasets:
        save_as = f'{dataset}_auc_metrics_by_model'
        with open(save_as + '.csv', mode='w') as file_handle:
            file_handle = csv.writer(file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            for each_llm, model_result in final_results.items():
                if dataset in model_result:
                    field_name = [each_llm.upper()]
                    file_handle.writerow(field_name)
                    file_handle.writerow(columns)
                
                    row = []
                    row.append(round(model_result[dataset]['subgroup']['aae'], 3))
                    row.append(round(model_result[dataset]['bpsn']['aae'], 3))
                    row.append(round(model_result[dataset]['bnsp']['aae'], 3))
                    row.append(round(model_result[dataset]['subgroup']['sae'], 3))
                    row.append(round(model_result[dataset]['bpsn']['sae'], 3))
                    row.append(round(model_result[dataset]['bnsp']['sae'], 3))
                    row.append(round(model_result[dataset]['gmb-subgroup'], 3))
                    row.append(round(model_result[dataset]['gmb-bpsn'], 3))
                    row.append(round(model_result[dataset]['gmb-bpsn'], 3))
                    file_handle.writerow(row)

                    # Give some space
                    file_handle.writerow([])
                
        path = "/home/eokpala/llm_annotation_bias/experiments/fine_tune_model/assess_bias_binary/"
        # Move a copy to SAVE_PATH
        shutil.copy(path + save_as + '.csv', SAVE_PATH + save_as + '.csv')
    
    
if __name__ == "__main__":
    # final_results = process_assess_bias()
    final_results = process_assess_bias(with_regularization=WITH_REGULARIZATION)
    save_result_to_csv_by_dataset(final_results)
    plot_score_by_dialect_by_dataset(final_results, add_bar_labels=False)
    plot_score_by_dialect_by_dataset(final_results, add_bar_labels=True)
        
    
