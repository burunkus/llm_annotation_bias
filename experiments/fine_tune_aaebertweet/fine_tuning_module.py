import math
import pickle
import os
import torch
import time
import numpy as np
import random
import datetime
import csv
from scipy import stats 
from _datetime import datetime as dt
from torch import nn
import torch.nn.functional as F
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
    AutoModelForSequenceClassification,
    AutoModel
)

seed_val = 23
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


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
    
