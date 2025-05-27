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
                            ) 
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
        
        self.ids = []
        self.labels = []
        self.raw_tweets = []

        with open(data_path) as input_file_handle:
            csv_reader = csv.reader(input_file_handle, delimiter=',')
            for i, line in enumerate(csv_reader, 1):
                tweet_id = line[0]
                author_id = line[1]
                tweet = line[2]
                label = line[3]
                
                self.ids.append(tweet_id.strip())
                self.labels.append(int(label.strip()))
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
        dataset_item = (torch.tensor(self.examples[index]),
                        torch.tensor(self.attention_masks[index]),
                        torch.tensor(self.token_type_ids[index]),
                        torch.tensor(self.labels[index])
                    )
        return dataset_item


def evaluate(dataloader,
             base_model_path,
             fine_tuned_model_path, 
             num_labels):
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
    
    # Initailize the model architecture using the retrained aaebert
    model = CustomBertModel(base_model_path,
                            num_labels)
    
    # Load the fine-tuned model weights
    model.load_state_dict(torch.load(fine_tuned_model_path))
    
    model.to(device)
    logging.info("Model loaded!")

    model.eval()
    
    gold_labels = []
    predicted_labels = []
    
    for batch in dataloader:
        # Unpack the inputs from our dataloader
        batch_input_ids, batch_input_mask, batch_token_type_ids, batch_labels = batch

        batch_input_ids = batch_input_ids.to(device)
        batch_input_mask = batch_input_mask.to(device)
        batch_token_type_ids = batch_token_type_ids.to(device)
        batch = (batch_input_ids, batch_input_mask, None)

        with torch.no_grad():
            outputs = model(batch)

        logits = outputs.logits

        # Get probabilities
        predictions = nn.functional.softmax(logits, dim=-1)
        # Move tensors to CPU and convert to numpy
        predictions = predictions.detach().cpu().numpy()
        pred_flat = np.argmax(predictions, axis=1).flatten()

        batch_labels = batch_labels.to('cpu').numpy()
        labels_flat = batch_labels.flatten()
        
        # Store gold labels single list
        gold_labels.extend(labels_flat)
        # Store predicted labels single list
        predicted_labels.extend(pred_flat)
        
    class_names = ['white', 'black']
    logging.info(f"{classification_report(gold_labels, predicted_labels, digits=4, target_names=class_names)}")  
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    
    logging.info(f"\nMacro F1: {macro_f1}")
    logging.info(f"Macro Precision: {macro_precision}")
    logging.info(f"Macro Recall: {macro_recall}")
    logging.info(f"Micro F1: {micro_f1}")
    logging.info(f"Micro Precision: {micro_precision}")
    logging.info(f"Micro Recall: {micro_recall}")

    
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    
    batch_size = 32
    num_labels = 2
    base_model_path = "/project/luofeng/socbd/eokpala/new_retrained_aaebert/aae_bertweet/checkpoint-410500"
    fine_tuned_model_path = "/project/luofeng/socbd/eokpala/bert_experiments/new_experiments_with_baseline_comparison/fine_tune_aaebert/bertweet/fine_tuned_aae/fine_tuned_aae.pth"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    data_path = "self_reported_user_data/self_reported_user_tweets_preprocessed.csv"
    data = CustomTextDataset(tokenizer, data_path)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    logging.info(f"Evaluating dialect model on self reported user tweets ...")
    evaluate(dataloader, base_model_path, fine_tuned_model_path, num_labels)
    logging.info('Evaluation completed.')
    