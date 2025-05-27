import numpy as np
import pickle
import json
import os
import csv
from pprint import pprint
import torch
import time
import datetime
import shutil
from bert_utils import SAVE_PATH  
        
def write_evaluation_results_to_csv(path, save_as, with_regularization=False):
    if with_regularization:
        names = {
            'bert_with_default_classes': 'BERT',
            'bertweet_with_default_classes': 'BERTweet',
            'hatebert_with_default_classes': 'HateBERT',
            'bertweet_reg_with_default_classes': 'BERTweet+Cos'
        }
    else:
        # we consider base models
        # names = {
        #     'bert_with_default_classes': 'BERT',
        #     'bertweet_with_default_classes': 'BERTweet',
        #     'hatebert_with_default_classes': 'HateBERT'
        # }
    
        # For use when processing the combined data/model trained on combined data (ICWSM revision)
        names = {
            'bert_with_default_classes_all_data': 'BERT', 
            'bertweet_with_default_classes_all_data': 'BERTweet',
            'hatebert_with_default_classes_all_data': 'HateBERT'
        }
        
    macro_results, micro_results = {}, {}
    
    files = os.listdir(path)
    for file in files:
        if '.' in file or file == "logs" or file == '__pycache__':
            continue
        
        if file not in names:
            continue
            
        files_in_folder = os.listdir(path + file)
        for child_file in files_in_folder:
            if child_file == "logs":
                continue
            dataset_name_array = child_file.split('.')
            dataset_name = dataset_name_array[0]
            extension = dataset_name_array[-1]
            if extension != "pickle":
                continue
                
            if dataset_name not in macro_results and dataset_name not in micro_results:
                macro_results[dataset_name] = {}
                micro_results[dataset_name] = {}
                
            with open(path + file + '/' + child_file, 'rb') as file_handle:
                unserialized_result_dict = pickle.load(file_handle)
            
            for key, value in unserialized_result_dict.items():
                result = []
                result.append(value['f1'])
                result.append(value['precision'])
                result.append(value['recall'])
                if key == 'macro':
                    macro_results[dataset_name][names[file]] = result
                else:
                    micro_results[dataset_name][names[file]] = result
                
    with open(save_as, mode='w') as file_handle:
        macro_column_name = ['Macro']
        micro_column_name = ['Micro']
        field_names = ['Dataset', 'Model', 'F1', 'Precision', 'Recall']
        file_handle = csv.writer(file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        datasets = macro_results.keys()
        # desired order of output
        
        if with_regularization:
            model_names = ['BERT', 
                           'BERTweet',
                           'HateBERT',
                           'BERTweet+Cos']
        else:
            model_names = ['BERT', 
                           'BERTweet',
                           'HateBERT']
        #model_names = ['BERT']
        for dataset in datasets:
            file_handle.writerow(macro_column_name)
            file_handle.writerow(field_names)
            for model in model_names:
                data = [dataset, model]
                file_handle.writerow(data + macro_results[dataset][model])
                
            file_handle.writerow(micro_column_name)

            for model in model_names:
                data = [dataset, model]
                file_handle.writerow(data + micro_results[dataset][model])
                
            # Give some space
            file_handle.writerow([])
            file_handle.writerow([])
    
    # Move logs to SAVE_PATH/...
    if with_regularization:
        names = {
            'bert_with_default_classes': 'bert',
            'bertweet_with_default_classes': 'bertweet',
            'hatebert_with_default_classes': 'hate_bert',
            'bertweet_reg_with_default_classes': 'bertweet_regularized'
        }
    else:
        # names = {
        #     'bert_with_default_classes': 'bert',
        #     'bertweet_with_default_classes': 'bertweet',
        #     'hatebert_with_default_classes': 'hate_bert'
        # }
        
        # Combined data
        names = {
            'bert_with_default_classes_all_data': 'bert',
            'bertweet_with_default_classes_all_data': 'bertweet',
            'hatebert_with_default_classes_all_data': 'hate_bert'
        }
  
    for folder, base in names.items():
        src_dir = path + folder + '/logs'

        # path to destination directory
        dest_dir = SAVE_PATH + base + '/logs/'
        
        # getting all the files in the source directory
        files = os.listdir(src_dir)
        
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
    
    # move result to SAVE_PATH
    shutil.copy(path + save_as, SAVE_PATH + save_as)

def main():
    path = "/home/eokpala/llm_annotation_bias/experiments/fine_tune_model/"
    save_as = "perf_result.csv"
    write_evaluation_results_to_csv(path, save_as)
    # write_evaluation_results_to_csv(path, save_as, with_regularization=True)
    
if __name__ == "__main__":
    main()
