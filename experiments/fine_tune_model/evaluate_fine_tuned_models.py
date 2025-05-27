import os
import sys
import torch
import time
import numpy as np
import random
import datetime
import pickle
import matplotlib.pyplot as plt
import logging
import logging.handlers
from _datetime import datetime as dt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix
from datasets import load_dataset, Features, ClassLabel, Value
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch import nn
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

def get_filename():
    #ct = datetime.datetime.now()
    #log_name = f"{ct.year}-{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}:{ct.minute:02d}:{ct.second:02d}"
    current_file_name = os.path.basename(__file__).split('.')[0]
    log_name = current_file_name
    return log_name


def get_logger(log_folder, log_filename):
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


def evaluate_model(model, test_dataloader, save_path, save_as):
    
    dataset_label_map = {
        "waseem_and_hovy": ["neither", "racism", "sexism"],
        "waseem": ["neither", "racism", "sexism", "racism and sexism"],
        "founta": ["normal", "hateful", "abusive"],
        "davidson": ["normal", "hate", "offensive"],
        "golbeck": ["not harassment", "harassment"],
        "hateval": ["not hate", "hate"],
        "offenseval": ["not offensive",  "offensive"],
        "abuseval": ["not abusive", "explicit abuse", "implicit abuse"]
    }
    
    logging.info('Evaluating trained model ................')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    gold_labels = []
    predicted_labels = []
    
    for batch in test_dataloader:
        # Add batch to GPU/CPU
        batch_input_ids = batch[0].to(device)
        batch_input_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)
        batch = (batch_input_ids, batch_input_mask, None)

        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits  # CustomBertModel

        predictions = nn.functional.softmax(logits, dim=-1)
        predictions = predictions.detach().cpu().numpy()
        pred_flat = np.argmax(predictions, axis=1).flatten()
        label_ids = batch_labels.to('cpu').numpy()
        labels_flat = label_ids.flatten()

        # Store gold labels single list
        gold_labels.extend(labels_flat)
        # Store predicted labels single list
        predicted_labels.extend(pred_flat)
    
    class_names = dataset_label_map[save_as]
    logging.info(f"{classification_report(gold_labels, predicted_labels, digits=4, target_names=class_names)}")  
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    
    logging.info(f"\nMacro F1: {macro_f1}")
    logging.info(f"Macro Precision: {macro_precision}")
    logging.info(f"Macro Recall: {macro_recall}")
    logging.info(f"Micro F1: {micro_f1}")
    logging.info(f"Micro Precision: {micro_precision}")
    logging.info(f"Micro Recall: {micro_recall}")
    
    result = {
        'macro': {
            'f1': macro_f1,
            'precision': macro_precision,
            'recall': macro_recall
        },
        'micro': {
            'f1': micro_f1,
            'precision': micro_precision,
            'recall': micro_recall
        }
    }
    
    save_as = save_as + '.pickle'
    with open(save_path + '/' + save_as, 'wb') as file_handle:
        pickle.dump(result, file_handle)
    
    
def main():
    path = PATH
    
    datasets = {
        "davidson": path + "davidson/davidson_with_default_classes_test_preprocessed_dialect_label.txt",
        "founta": path + "founta/founta_with_default_classes_test_preprocessed_dialect_label.txt",
        "golbeck": path + "golbeck/golbeck_with_default_classes_test_preprocessed_dialect_label.txt",
        "hateval2019": path + "hateval2019/hateval2019_en_test_preprocessed_dialect_label.txt",
        "offenseval2019": path + "offenseval2019/offenseval_test_preprocessed_dialect_label.txt"
    }
    
    classifiers = {
        "bert": "bert-base-uncased", 
        "bertweet": "vinai/bertweet-base", 
        "hate_bert": "../../../../../../project/luofeng/socbd/eokpala/hate_bert"
    }
    
    for classifier_name, base_model_config_path in classifiers.items():
        classifier_path = SAVE_PATH + classifier_name
        
        fine_tuned_models = sorted(os.listdir(classifier_path))
        # Path where the pre-trained model and tokenizer can be found
        for fine_tuned_model in fine_tuned_models:
            if 'logs' in fine_tuned_model or 'png' in fine_tuned_model:
                continue
            
            logging.info(f"Evaluating {classifier_path + '/' + fine_tuned_model}")
            model_path = classifier_path + "/" + fine_tuned_model
            tokenizer_path = base_model_config_path
            dataset_model_name = fine_tuned_model.split('_')[-1]
            if dataset_model_name == 'hateval':
                dataset = 'hateval2019'
            elif dataset_model_name == 'offenseval':
                dataset = 'offenseval2019'
            else:
                dataset = dataset_model_name
            
            if dataset in datasets:
                batch_size, _, _ = hyperparameters()
                num_labels = NUMBER_OF_LABELS_MAP[dataset]

                classifier_criterion = custom_bert_parameters()
                model = CustomBertModel(base_model_config_path,
                                        num_labels)

                model_location = model_path + '/' + fine_tuned_model + '.pth'
                model.load_state_dict(torch.load(model_location))
                tokenizer = model.tokenizer
            
                test_data = CustomTextDataset(tokenizer, datasets[dataset])
                test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
                if classifier_name == 'hate_bert':
                    save_path = f'hatebert_with_default_classes'
                else:
                    save_path = f'{classifier_name}_with_default_classes'
                evaluate_model(model, test_dataloader, save_path, dataset_model_name)
    
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    curr_evaluation = "_multi_class"
    _ = get_logger(log_dir, get_filename() + curr_evaluation)
    main()
