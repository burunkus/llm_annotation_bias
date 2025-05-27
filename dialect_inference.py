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
import logging
import logging.handlers
from sklearn.model_selection import train_test_split
from _datetime import datetime as dt
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from datasets import load_dataset, Features, ClassLabel, Value
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    get_scheduler,
    AutoModel,
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    RobertaForSequenceClassification,
    BertForSequenceClassification
)


def get_filename():
    #ct = datetime.datetime.now()
    #log_name = f"{ct.year}-{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}:{ct.minute:02d}:{ct.second:02d}"
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


class CustomBertModel(nn.Module):

    def __init__(self, 
                 model_path, 
                 num_labels):
        super(CustomBertModel, self).__init__()
        self.model_path = model_path
        self.num_labels = num_labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(self.model_path,
                                                                       num_labels=self.num_labels,
                                                                       output_attentions=False,
                                                                       output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        
    def forward(self, batch):
        '''
        Args:
            batch (Tuple): a tuple of tokenized batch of examples containing the batch's
            input ids, token type ids, input mask and labels. 
        Return: 
            outputs (Tuple): A tuple of SequenceClassifierOutput and representations of size [batch size, embedding size]
        '''

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_input_ids, batch_input_mask, batch_labels = batch
        outputs = self.bert(batch_input_ids,
                            token_type_ids=None,
                            attention_mask=batch_input_mask,
                            labels=batch_labels
                            ) # If we pass the labels, loss will be computed for us automatically
        hidden_states = outputs.hidden_states
        last_hidden_layer = hidden_states[-1] #layer 12 Of shape (# batch size, # tokens, hidden size)
        representations = last_hidden_layer[:,0,:] # Take each example, select the cls token (at index 0) of each example, take the full length i.e the dimension (hidden size) producing a tensor of shape (# batch size, hidden size)

        return outputs 
    

class CustomTextDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 padding="max_length",
                 truncation=True,
                 max_length=100,
                 with_gpt_label=True
                 ):

        """
        Generate a single example and its label from data_path
        Args:
            tokenizer (Object): BERT/RoBERTa tokenization object
            data_path (String): Absolute path to the train/test dataset. 
            Each line is of the form tweetID \t userID \t tweet
            padding (String): How to padding sequences, defaults to "max_lenght"
            truncation (Boolean): Whether to truncate sequences, defaults to True
            max_length (Int): The maximum length of a sequence, sequence longer
            than max_length will be truncated and those shorter will be padded
        Retruns:
            dataset_item (Tuple): A tuple of tensors - tokenized text, attention mask and labels
        """

        if not os.path.exists(data_path):
            raise ValueError(f"Input file {data_path} does not exist")
        
        self.with_gpt_label = with_gpt_label
        self.uids = []
        self.labels = []
        self.gpt_labels = []
        self.binary_labels = []
        self.raw_tweets = []

        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        if file_extension == "txt":
            with open(data_path, encoding="utf-8") as file_handler:
                for line in file_handler:
                    if with_gpt_label:
                        uid, tweet, label, gpt_label, binary_label = line.split("\t")
                        self.gpt_labels.append(int(gpt_label.strip()))
                    else:
                        uid, tweet, label, binary_label = line.split("\t")
                    
                    self.uids.append(uid.strip())
                    self.labels.append(int(label.strip()))
                    self.binary_labels.append(int(binary_label.strip()))
                    self.raw_tweets.append(tweet.strip())

            tokenized_tweet = tokenizer(self.raw_tweets,
                                        padding=padding,
                                        truncation=truncation,
                                        max_length=max_length)

            self.examples = tokenized_tweet['input_ids']
            self.attention_masks = tokenized_tweet['attention_mask']
            self.token_type_ids = tokenized_tweet['token_type_ids']


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.with_gpt_label:
            dataset_item = (torch.tensor(self.examples[index]),
                            torch.tensor(self.attention_masks[index]),
                            torch.tensor(self.token_type_ids[index]),
                            torch.tensor(self.labels[index]),
                            torch.tensor(self.gpt_labels[index]),
                            torch.tensor(self.binary_labels[index]),
                            self.raw_tweets[index],
                            self.uids[index]
                        )
        else:
            dataset_item = (torch.tensor(self.examples[index]),
                            torch.tensor(self.attention_masks[index]),
                            torch.tensor(self.token_type_ids[index]),
                            torch.tensor(self.labels[index]),
                            torch.tensor(self.binary_labels[index]),
                            self.raw_tweets[index],
                            self.uids[index]
                        )
        return dataset_item


def classify(dataloader,
             base_model_path,
             fine_tuned_model_path, 
             num_labels, 
             save_as,
             with_gpt_label=True):
    '''
    Classify the sampled white/black aligned tweets
    Args:
        train_dataloader (Iterable): A PyTorch iterable object through the train set
        test_dataloader (Iterable): A PyTorch iterable object through the test set
        model_path (String): The location (absolute) of the pre-trained model
        num_labels (Int): The number of classes in our task
        save_train_as (String): Absolute path where the classified training tweets will be saved
        save_test_as (String): Absolute path where the classified testing tweets will be saved
    Returns:
        None
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    logging.info("Loading model")
    
    # Initailize the model architecture using the retrained aaebertweet
    model = CustomBertModel(base_model_path,
                            num_labels)
    
    # Load the fine-tuned model weights
    model.load_state_dict(torch.load(fine_tuned_model_path))
    
    model.to(device)
    logging.info("Model loaded!")

    model.eval()
    
    with open(save_as, 'w') as save_file_handle:
        for batch in dataloader:
            if with_gpt_label:
                # Unpack the inputs from our dataloader
                batch_input_ids, batch_input_mask, batch_token_type_ids, batch_labels, batch_gpt_labels, batch_binary_labels, batch_raw_tweets, batch_uids = batch
            else:
                # Unpack the inputs from our dataloader
                batch_input_ids, batch_input_mask, batch_token_type_ids, batch_labels, batch_binary_labels, batch_raw_tweets, batch_uids = batch

            batch_input_ids = batch_input_ids.to(device)
            batch_input_mask = batch_input_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch = (batch_input_ids, batch_input_mask, None)

            with torch.no_grad():
                outputs = model(batch)

            logits = outputs.logits
            hidden_states = outputs.hidden_states
            last_hidden_layer = hidden_states[-1] 
            sequence_repr = last_hidden_layer[:, 0, :]

            # Get probabilities
            predictions = nn.functional.softmax(logits, dim=-1)
            # Move tensors to CPU and convert to numpy
            predictions = predictions.detach().cpu().numpy()

            #print(f"predictions: {predictions}, size: {predictions.shape}")
            batch_labels = batch_labels.to('cpu').numpy()
            batch_binary_labels = batch_binary_labels.to('cpu').numpy()
            batch_uids = np.asarray(list(batch_uids))
            batch_raw_tweets = np.asarray(list(batch_raw_tweets))

            if with_gpt_label:
                batch_gpt_labels = batch_gpt_labels.to('cpu').numpy()

            # Flatten 
            predictions_flat = np.argmax(predictions, axis=1).flatten()

            for i, predicted_dialect in enumerate(predictions_flat):
                if with_gpt_label:
                    save_file_handle.write(f"{batch_uids[i]}\t{batch_raw_tweets[i]}\t{batch_labels[i]}\t{batch_gpt_labels[i]}\t{batch_binary_labels[i]}\t{predicted_dialect}\n")
                else:
                    save_file_handle.write(f"{batch_uids[i]}\t{batch_raw_tweets[i]}\t{batch_labels[i]}\t{batch_binary_labels[i]}\t{predicted_dialect}\n")
    
           
            
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    
    batch_size = 32
    num_labels = 2
    base_model_path = "/project/luofeng/socbd/eokpala/new_retrained_aaebert/aae_bertweet/checkpoint-410500"
    fine_tuned_model_path = "/project/luofeng/socbd/eokpala/bert_experiments/new_experiments_with_baseline_comparison/fine_tune_aaebert/bertweet/fine_tuned_aae/fine_tuned_aae.pth"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    data_paths = [
        "waseem/waseem_with_default_classes_train_preprocessed.txt",
        "waseem/waseem_with_default_classes_test_preprocessed.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_train_preprocessed.txt",
        "waseem-and-hovy/waseem_and_hovy_with_default_classes_test_preprocessed.txt",
        "founta/founta_with_default_classes_train_preprocessed.txt",
        "founta/founta_with_default_classes_test_preprocessed.txt",
        "davidson/davidson_with_default_classes_train_preprocessed.txt",
        "davidson/davidson_with_default_classes_test_preprocessed.txt",
        "golbeck/golbeck_with_default_classes_train_preprocessed.txt",
        "golbeck/golbeck_with_default_classes_test_preprocessed.txt",
        "hateval2019/hateval2019_en_train_preprocessed.txt",
        "hateval2019/hateval2019_en_test_preprocessed.txt",
        "hateval2019/hateval2019_en_dev_preprocessed.txt",
        "offenseval2019/offenseval_train_preprocessed.txt",
        "offenseval2019/offenseval_test_preprocessed.txt",
        "abuseval/abuseval_train_with_default_classes_preprocessed.txt",
        "abuseval/abuseval_test_with_default_classes_preprocessed.txt"
    ]
    
    paths = [
        "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/general_prompt_annotation/",
        "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/few_shot_prompt_annotation/",    
        "/project/luofeng/socbd/eokpala/llm_annotation_bias/data/cot_prompt_annotation/"
    ]
    
    logging.info(f'Dialect inference for GPT annotated data.')
    for path in paths:
        logging.info(f'Processing {path}')
        for data_path in data_paths:
            data = CustomTextDataset(tokenizer, path + data_path, with_gpt_label=True)
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    
            current_data = data_path.split('/')
            dataset_name = current_data[0]
            dataset_file = current_data[1]
            save_as = path + dataset_name + '/' + dataset_file.split('.')[0] + '_dialect_label.txt'
            logging.info(f"Infering dialect for {data_path}")
            classify(dataloader, base_model_path, fine_tuned_model_path, num_labels, save_as, with_gpt_label=True)
    
    logging.info(f'Dialect inference for human annotated data.')
    paths = ["/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"]
    for path in paths:
        logging.info(f'Processing {path}')
        for data_path in data_paths:
            data = CustomTextDataset(tokenizer, path + data_path, with_gpt_label=False)
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    
            current_data = data_path.split('/')
            dataset_name = current_data[0]
            dataset_file = current_data[1]
            save_as = path + dataset_name + '/' + dataset_file.split('.')[0] + '_dialect_label.txt'
            logging.info(f"Infering dialect for {data_path}")
            classify(dataloader, base_model_path, fine_tuned_model_path, num_labels, save_as, with_gpt_label=False)
    
    logging.info("Completed.")