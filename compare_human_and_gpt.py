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
from pprint import pprint

def get_filename():
    current_file_name = os.path.basename(__file__).split('.')[0]
    log_name = current_file_name
    return log_name


def get_logger(log_folder,log_filename):
    if os.path.exists(log_folder) == False:
        os.makedirs(log_folder)
    
    logger = logging.getLogger(os.path.join(log_folder, log_filename))

    log_handler = logging.FileHandler(os.path.join(log_folder, log_filename+'.log'), mode='w')
    logger.addHandler(log_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    
    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]:  %(message)s', datefmt="%m-%d-%Y %H:%M:%S")
    log_handler.setFormatter(formatter)
    
    logger.setLevel(logging.INFO)
    return logger

label_map = {
        "waseem": {1: "Racism", 2: "Sexism", 3: "Both", 0: "Neither"},
        "hovy": {1: "Racism", 2: "Sexism", 0: "None"},
        "founta": {1: "Hateful", 2: "Abusive", 0: "Normal"},
        "davidson": {1: "Hate", 2: "Offensive", 0: "Normal"},
        "golbeck": {1: "Harassment", 0: "Non-harassment"},
        "hateval": {1: "Hate", 0: "Non-hate"},
        "offenseval": {1: "Offensive", 0: "Non-offensive"},
        "abuseval": {1: "Explicit", 2: "Implicit", 0: "Neither"}
    }


def performance_metrics(data_path_with_dialect, dataset_name, logger):
    
    dataset_label_map = {
            "hovy": ["neither", "racism", "sexism"],
            "waseem": ["neither", "racism", "sexism", "racism and sexism"],
            "founta": ["normal", "hateful", "abusive"],
            "davidson": ["normal", "hate", "offensive"],
            "golbeck": ["not harassment", "harassment"],
            "hateval": ["not hate", "hate"],
            "offenseval": ["not offensive",  "offensive"],
            "abuseval": ["not abusive", "explicit abuse", "implicit abuse"]
        }
    
    human_labels_aae = []
    human_labels_sae = []
    gpt_labels_aae = []
    gpt_labels_sae = []
    
    with open(data_path_with_dialect) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            dialect_label = int(line[5])
            
            if dialect_label == 1:
                human_labels_aae.append(human_label)
                gpt_labels_aae.append(gpt_label)
            else:
                human_labels_sae.append(human_label)
                gpt_labels_sae.append(gpt_label)
    
    dataset_classes = dataset_label_map[dataset_name]
    if dataset_name != 'hovy':
        logger.info(f'Performance of AAE:')
        logger.info(f"{classification_report(human_labels_aae, gpt_labels_aae, digits=4, target_names=dataset_classes)}")
        precision, recall, f1_score, _ = precision_recall_fscore_support(human_labels_aae, gpt_labels_aae, average='macro')
        accuracy = accuracy_score(human_labels_aae, gpt_labels_aae)
        logger.info(f'Macro Precision: {precision}, Recall: {recall}, and F1: {f1_score}. Accuracy: {accuracy}')

        logger.info(f'Performance of SAE:')
        logger.info(f"{classification_report(human_labels_sae, gpt_labels_sae, digits=4, target_names=dataset_classes)}")
        precision, recall, f1_score, _ = precision_recall_fscore_support(human_labels_sae, gpt_labels_sae, average='macro')
        accuracy = accuracy_score(human_labels_sae, gpt_labels_sae)
        logger.info(f'Macro Precision: {precision}, Recall: {recall}, and F1: {f1_score}. Accuracy: {accuracy}')
    
    
def plot_confusion_matrix(data_path, data_path_with_dialect, dataset_name):
    
    labels = list(label_map[dataset_name].items())
    labels = sorted(labels)
    labels = [label for label_index, label in labels]
    
    human_labels = []
    gpt_labels = []
    
    with open(data_path) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            human_labels.append(human_label)
            gpt_labels.append(gpt_label)
    
    for norm in ['all', None]:
        disp = ConfusionMatrixDisplay.from_predictions(human_labels,
                                                       gpt_labels,
                                                       display_labels=labels,
                                                       normalize=norm,
                                                       colorbar=False,
                                                       cmap=plt.cm.Greys)
        disp.plot()
        disp.ax_.set(xlabel='GPT', ylabel='Human')
        disp.ax_.set_title(f'{dataset_name.title()}')
        if norm == None:
            save_as = 'confusion_matrix_figures/' + dataset_name + '_cm.pdf'
        else:
            save_as = 'confusion_matrix_figures_normalized/' + dataset_name + '_cm.pdf'
        plt.savefig(save_as, dpi=300, transparent=True, bbox_inches='tight', format='pdf')
    
    
    human_labels_aae, human_labels_sae = [], []
    gpt_labels_aae, gpt_labels_sae = [], []
    
    with open(data_path_with_dialect) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            dialect_label = int(line[5])

            if dialect_label == 1:
                human_labels_aae.append(human_label)
                gpt_labels_aae.append(gpt_label)
            else:
                human_labels_sae.append(human_label)
                gpt_labels_sae.append(gpt_label)
    
    # Don't plot confusion matrix for Waseem and Hovy because the tweets belonging to the racism class and sexism class are all SAE
    if dataset_name != 'hovy':
        # AAE
        for norm in ['all', None]:
            disp = ConfusionMatrixDisplay.from_predictions(human_labels_aae,
                                                           gpt_labels_aae,
                                                           display_labels=labels,
                                                           normalize=norm,
                                                           colorbar=False,
                                                           cmap=plt.cm.Greys)
            disp.plot()
            disp.ax_.set(xlabel='GPT', ylabel='Human')
            disp.ax_.set_title(f'{dataset_name.title() + " with AAE label"}')
            if norm == None:
                save_as = 'confusion_matrix_figures/' + dataset_name + '_cm_aae.pdf'
            else:
                save_as = 'confusion_matrix_figures_normalized/' + dataset_name + '_cm_aae.pdf'
            plt.savefig(save_as, dpi=300, transparent=True, bbox_inches='tight', format='pdf')       

        # SAE
        for norm in ['all', None]:
            disp = ConfusionMatrixDisplay.from_predictions(human_labels_sae,
                                                           gpt_labels_sae,
                                                           display_labels=labels,
                                                           normalize=norm,
                                                           colorbar=False,
                                                           cmap=plt.cm.Greys)
            disp.plot()
            disp.ax_.set(xlabel='GPT', ylabel='Human')
            disp.ax_.set_title(f'{dataset_name.title() + " with SAE label"}')
            if norm == None:
                save_as = 'confusion_matrix_figures/' + dataset_name + '_cm_sae.pdf'
            else:
                save_as = 'confusion_matrix_figures_normalized/' + dataset_name + '_cm_sae.pdf'
            plt.savefig(save_as, dpi=300, transparent=True, bbox_inches='tight', format='pdf')    


def plot_confusion_matrix_of_full_data(data_path_train, 
                                       data_path_test,
                                       data_path_with_dialect_train,
                                       data_path_with_dialect_test,
                                       dataset_name,
                                       annotation_strategy='general_prompt_annotation'):
    
    labels = list(label_map[dataset_name].items())
    labels = sorted(labels)
    labels = [label for label_index, label in labels]
    
    human_labels = []
    gpt_labels = []
    
    with open(data_path_train) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            human_labels.append(human_label)
            gpt_labels.append(gpt_label)
    
    with open(data_path_test) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            human_labels.append(human_label)
            gpt_labels.append(gpt_label)
    
    if annotation_strategy == 'general_prompt_annotation':
        custom_name = ''
    elif annotation_strategy == 'few_shot_prompt_annotation':
        custom_name = 'few_shot'
    elif annotation_strategy == 'cot_prompt_annotation':
        custom_name = 'chain_of_thought'
            
    for norm in ['all', None]:
        disp = ConfusionMatrixDisplay.from_predictions(human_labels,
                                                       gpt_labels,
                                                       display_labels=labels,
                                                       normalize=norm,
                                                       colorbar=False,
                                                       cmap=plt.cm.Greys)
        disp.plot()
        disp.ax_.set(xlabel='GPT', ylabel='Human')
        #disp.ax_.set_title(f'{dataset_name.title()}')
        if norm == None:
            save_as = f"confusion_matrix_figures_full_data/{dataset_name}_cm_{custom_name}.pdf"
        else:
            # save_as = 'confusion_matrix_figures_normalized_full_data/' + dataset_name + '_cm.pdf'
            save_as = f"confusion_matrix_figures_normalized_full_data/{dataset_name}_cm_{custom_name}.pdf"
        plt.savefig(save_as, dpi=300, transparent=True, bbox_inches='tight', format='pdf')
    
    human_labels_aae, human_labels_sae = [], []
    gpt_labels_aae, gpt_labels_sae = [], []
    
    with open(data_path_with_dialect_train) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            dialect_label = int(line[5])

            if dialect_label == 1:
                human_labels_aae.append(human_label)
                gpt_labels_aae.append(gpt_label)
            else:
                human_labels_sae.append(human_label)
                gpt_labels_sae.append(gpt_label)
    
    with open(data_path_with_dialect_test) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            dialect_label = int(line[5])

            if dialect_label == 1:
                human_labels_aae.append(human_label)
                gpt_labels_aae.append(gpt_label)
            else:
                human_labels_sae.append(human_label)
                gpt_labels_sae.append(gpt_label)
                
    # Don't plot confusion matrix for Waseem and Hovy because the tweets belonging to the racism class and sexism class are all SAE
    if dataset_name != 'hovy':
        # AAE
        for norm in ['all', None]:
            disp = ConfusionMatrixDisplay.from_predictions(human_labels_aae,
                                                           gpt_labels_aae,
                                                           display_labels=labels,
                                                           normalize=norm,
                                                           colorbar=False,
                                                           cmap=plt.cm.Greys)
            disp.plot()
            disp.ax_.set(xlabel='GPT', ylabel='Human')
            #disp.ax_.set_title(f'{dataset_name.title() + " with AAE label"}')
            if norm == None:
                save_as = f"confusion_matrix_figures_full_data/{dataset_name}_cm_aae_{custom_name}.pdf"
            else:
                save_as = f"confusion_matrix_figures_normalized_full_data/{dataset_name}_cm_aae_{custom_name}.pdf"
            plt.savefig(save_as, dpi=300, transparent=True, bbox_inches='tight', format='pdf')       

        # SAE
        for norm in ['all', None]:
            disp = ConfusionMatrixDisplay.from_predictions(human_labels_sae,
                                                           gpt_labels_sae,
                                                           display_labels=labels,
                                                           normalize=norm,
                                                           colorbar=False,
                                                           cmap=plt.cm.Greys)
            disp.plot()
            disp.ax_.set(xlabel='GPT', ylabel='Human')
            #disp.ax_.set_title(f'{dataset_name.title() + " with SAE label"}')
            if norm == None:
                save_as = f"confusion_matrix_figures_full_data/{dataset_name}_cm_sae_{custom_name}.pdf"
            else:
                save_as = f"confusion_matrix_figures_normalized_full_data/{dataset_name}_cm_sae_{custom_name}.pdf"
            plt.savefig(save_as, dpi=300, transparent=True, bbox_inches='tight', format='pdf')
            

def performance_metrics_full_data(data_path_train, 
                                  data_path_test, 
                                  data_path_with_dialect_train, 
                                  data_path_with_dialect_test, 
                                  dataset_name, 
                                  logger):
    
    dataset_label_map = {
            "hovy": ["neither", "racism", "sexism"],
            "waseem": ["neither", "racism", "sexism", "racism and sexism"],
            "founta": ["normal", "hateful", "abusive"],
            "davidson": ["normal", "hate", "offensive"],
            "golbeck": ["not harassment", "harassment"],
            "hateval": ["not hate", "hate"],
            "offenseval": ["not offensive",  "offensive"],
            "abuseval": ["not abusive", "explicit abuse", "implicit abuse"]
        }
    
    human_labels = []
    gpt_labels = []
    
    with open(data_path_train) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            human_labels.append(human_label)
            gpt_labels.append(gpt_label)
            
    with open(data_path_test) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            human_labels.append(human_label)
            gpt_labels.append(gpt_label)
    
    human_labels_aae = []
    human_labels_sae = []
    gpt_labels_aae = []
    gpt_labels_sae = []
    
    with open(data_path_with_dialect_train) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            dialect_label = int(line[5])
            
            if dialect_label == 1:
                human_labels_aae.append(human_label)
                gpt_labels_aae.append(gpt_label)
            else:
                human_labels_sae.append(human_label)
                gpt_labels_sae.append(gpt_label)
    
    with open(data_path_with_dialect_test) as file_handle:
        for i, row in enumerate(file_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = int(line[2])
            gpt_label = int(line[3])
            dialect_label = int(line[5])
            
            if dialect_label == 1:
                human_labels_aae.append(human_label)
                gpt_labels_aae.append(gpt_label)
            else:
                human_labels_sae.append(human_label)
                gpt_labels_sae.append(gpt_label)
                
    dataset_classes = dataset_label_map[dataset_name]
    if dataset_name != 'hovy':
        logger.info(f'Performance of full dataset:')
        logger.info(f"{classification_report(human_labels, gpt_labels, digits=4, target_names=dataset_classes)}")
        precision, recall, f1_score, _ = precision_recall_fscore_support(human_labels, gpt_labels, average='macro')
        accuracy = accuracy_score(human_labels, gpt_labels)
        logger.info(f'Macro Precision: {precision}, Recall: {recall}, and F1: {f1_score}. Accuracy: {accuracy}')
        
        logger.info(f'Performance of AAE:')
        logger.info(f"{classification_report(human_labels_aae, gpt_labels_aae, digits=4, target_names=dataset_classes)}")
        precision, recall, f1_score, _ = precision_recall_fscore_support(human_labels_aae, gpt_labels_aae, average='macro')
        accuracy = accuracy_score(human_labels_aae, gpt_labels_aae)
        logger.info(f'Macro Precision: {precision}, Recall: {recall}, and F1: {f1_score}. Accuracy: {accuracy}')
        
        logger.info(f'Performance of SAE:')
        logger.info(f"{classification_report(human_labels_sae, gpt_labels_sae, digits=4, target_names=dataset_classes)}")
        precision, recall, f1_score, _ = precision_recall_fscore_support(human_labels_sae, gpt_labels_sae, average='macro')
        accuracy = accuracy_score(human_labels_sae, gpt_labels_sae)
        logger.info(f'Macro Precision: {precision}, Recall: {recall}, and F1: {f1_score}. Accuracy: {accuracy}')
        
        
def test_logger(logger, dataset_name):
    
    logger.info(dataset_name)
    
    
if __name__ == "__main__":
    log_dir ='./log_folder_cm'
    
    data_paths = [
        "davidson/davidson_with_default_classes_train.txt",
        "waseem/waseem_with_default_classes_train.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_train.txt",
        "founta/founta_with_default_classes_train.txt",
        "golbeck/golbeck_with_default_classes_train.txt",
        "hateval2019/hateval2019_en_train.txt",
        "offenseval2019/offenseval_train.txt",
        "abuseval/abuseval_train_with_default_classes.txt"
    ]
    dialect_data_paths = [
        "davidson/davidson_with_default_classes_train_preprocessed_dialect_label.txt",
        "waseem/waseem_with_default_classes_train_preprocessed_dialect_label.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_train_preprocessed_dialect_label.txt",
        "founta/founta_with_default_classes_train_preprocessed_dialect_label.txt",
        "golbeck/golbeck_with_default_classes_train_preprocessed_dialect_label.txt",
        "hateval2019/hateval2019_en_train_preprocessed_dialect_label.txt",
        "offenseval2019/offenseval_train_preprocessed_dialect_label.txt",
        "abuseval/abuseval_train_with_default_classes_preprocessed_dialect_label.txt"
    ]
    dataset_names = ['davidson', 'waseem', 'hovy', 'founta', 'golbeck', 'hateval', 'offenseval', 'abuseval']
    annotation_strategy = "cot_prompt_annotation" # general_prompt_annotation, few_shot_prompt_annotation, or cot_prompt_annotation
    data_path = f'/project/luofeng/socbd/eokpala/llm_annotation_bias/data/{annotation_strategy}/'
    
    # Get confusion matrix on only train set
    # for i, dataset in enumerate(data_paths):
    #     logger = get_logger(log_dir, get_filename() + f'_new_{dataset_names[i]}')
    #     plot_confusion_matrix(data_path + dataset, data_path + dialect_data_paths[i], dataset_names[i])
    
    # Get confusion matrix on full data
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
    
    dialect_data_paths_test = [
        "davidson/davidson_with_default_classes_test_preprocessed_dialect_label.txt",
        "waseem/waseem_with_default_classes_test_preprocessed_dialect_label.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_test_preprocessed_dialect_label.txt",
        "founta/founta_with_default_classes_test_preprocessed_dialect_label.txt",
        "golbeck/golbeck_with_default_classes_test_preprocessed_dialect_label.txt",
        "hateval2019/hateval2019_en_test_preprocessed_dialect_label.txt",
        "offenseval2019/offenseval_test_preprocessed_dialect_label.txt",
        "abuseval/abuseval_test_with_default_classes_preprocessed_dialect_label.txt"
    ]
    
    for i, dataset in enumerate(data_paths):
        logger = get_logger(log_dir, get_filename() + f'_{dataset_names[i]}')
        performance_metrics_full_data(data_path + dataset, 
                                      data_path + data_paths_test[i],
                                      data_path + dialect_data_paths[i],
                                      data_path + dialect_data_paths_test[i],
                                      dataset_names[i], 
                                      logger)
        plot_confusion_matrix_of_full_data(data_path + dataset, 
                                           data_path + data_paths_test[i],
                                           data_path + dialect_data_paths[i],
                                           data_path + dialect_data_paths_test[i],
                                           dataset_names[i],
                                           annotation_strategy=annotation_strategy)