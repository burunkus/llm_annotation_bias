import os
import sys
import torch
import time
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import pickle
from _datetime import datetime as dt
from torch import nn
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
        
        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        if file_extension == "txt":
            with open(data_path, encoding="utf-8") as file_handler:
                tweets = []
                for line in file_handler:
                    tweet, label = line.split("\t")
                    tweet = tweet.strip()
                    label = label.strip()
                    tweets.append(tweet)
                    self.labels.append(int(label))

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
                        torch.tensor(self.labels[index])
                    )
        return dataset_item
    

class EarlyStopping:
    """
    Stops the training if validation loss doesn't improve after a given patience.
    Credit: https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
def flat_accuracy(preds, labels):
    """
    Calculate the accuracy using the predicted values and the true labels
    Note: Code adapted from https://osf.io/qkjuv/
    Args:
        preds: ndarray of model predictions
        labels: ndarray of true labels
    Returns:
        accuracy (ndarray): accuracy of the current batch
    """

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    Note: Code adapted from https://osf.io/qkjuv/
    '''

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate_model(model, test_dataloader, save_path, logging):
    
    logging.info('Evaluating trained model ................')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    gold_labels = []
    predicted_labels = []
    
    for batch in test_dataloader:
        # Add batch to GPU/CPU
        batch_input_ids = batch[0].to(device)
        batch_input_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)
        batch = (batch_input_ids, batch_input_mask, None)

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits
            outputs = model(batch)
            logits = outputs.logits

        predictions = nn.functional.softmax(logits, dim=-1)
        predictions = predictions.detach().cpu().numpy()
        pred_flat = np.argmax(predictions, axis=1).flatten()
        label_ids = batch_labels.to('cpu').numpy()
        labels_flat = label_ids.flatten()

        # Store gold labels single list
        gold_labels.extend(labels_flat)
        # Store predicted labels single list
        predicted_labels.extend(pred_flat)
            
    logging.info(f"{classification_report(gold_labels, predicted_labels, digits=4)}")  
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    
    logging.info(f"Macro F1: {macro_f1}")
    logging.info(f"Macro Precision: {macro_precision}")
    logging.info(f"Macro Recall: {macro_recall}")
    logging.info(f"Micro F1: {micro_f1}")
    logging.info(f"Micro Precision: {micro_precision}")
    logging.info(f"Micro Recall: {micro_recall}")
    
    """
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
    
    save_as = save_path.split('/')[-2].split('_')[-1] + '.pickle'
    with open(save_as, 'wb') as file_handle:
        pickle.dump(result, file_handle)
    """
    
        
def train(train_dataloader,
          test_dataloader,
          tokenizer,
          model,
          num_labels,
          learning_rate,
          epochs,
          save_path,
          classifier_criterion,
          logging,
          patience=None):

    """
    Fine-tune a new model using the pre-trained model and save the new model in
    save_path
    Note: Most of the code has been adapted from the HateBERT training implementation
    located at: https://osf.io/qkjuv/

    Args:
        train_dataloader (Object): A PyTorch iterable object through the train set
        test_dataloader (Object): A PyTorch iterable object through the test set
        model_path (String): The location (absolute) of the pre-trained model
        num_labels (Int): The number of classes in our task
        learning_rate (Float): Learning rate for the optimizer
        epochs(Int): The number of times to go through the entire dataset
        save_path (String): Absolute path where the fine-tuned model will be saved
    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model.to(device)
    print("Model loaded!")

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=1e-8)
    num_training_steps = len(train_dataloader) * epochs
    learning_rate_scheduler = get_scheduler("linear",
                                            optimizer=optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=num_training_steps)
    # path to save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    name = save_path.split('/')[-2] + '.pth'
    
    # intialize early stopping object if early stopping is being used (i.e when patience is provided)
    if patience is not None:
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path+name, trace_func=logging.info)
        
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    eval_loss_values = []
    
    # For each epoch...
    for epoch in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Store true lables for global eval
        gold_labels = []

        # Store predicted labels for global eval
        predicted_labels = []

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0.0
        total_train_accuracy = 0.0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Unpack this training batch from our dataloader.
            # `batch` contains three pytorch tensors:
            # [0]: input ids
            # [1]: attention masks
            # [2]: labels
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            
            batch = (batch_input_ids, batch_input_mask, batch_labels)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return logits from the classifier before softmax/sigmoid
            outputs = model(batch)
            loss = classifier_criterion(outputs.logits, batch_labels)
            
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            learning_rate_scheduler.step()
            
            # Get predictions and move predictions and labels to CPU
            logits = outputs.logits
            predictions = nn.functional.softmax(logits, dim=-1)
            predictions = predictions.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            train_accuracy = flat_accuracy(predictions, label_ids)
            # Accumulate the total accuracy.
            total_train_accuracy += train_accuracy
            

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        
        # Get average train accuracy
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        
        elapsed_train_time = format_time(time.time() - t0)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_loss, total_eval_accuracy = 0.0, 0.0
        num_eval_steps, num_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU/CPU
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            batch = (batch_input_ids, batch_input_mask, batch_labels)

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits
                outputs = model(batch)
                logits = outputs.logits
            
            # Calculate validation loss
            val_loss = classifier_criterion(logits, batch_labels)
            
            # Accumulate validation loss
            total_eval_loss += val_loss
            
            predictions = nn.functional.softmax(logits, dim=-1)
            # Move predictions and labels to CPU
            predictions = predictions.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_accuracy = flat_accuracy(predictions, label_ids)

            # Accumulate the total accuracy.
            total_eval_accuracy += eval_accuracy

            # Track the number of batches
            num_eval_steps += 1

            predictions_flat = np.argmax(predictions, axis=1).flatten()
            labels_flat = label_ids.flatten()

            # Store gold labels single list
            gold_labels.extend(labels_flat)
            # Store predicted labels single list
            predicted_labels.extend(predictions_flat)
        
        elapsed_eval_time = format_time(time.time() - t0)
        avg_eval_loss = total_eval_loss / num_eval_steps
        eval_loss_values.append(avg_eval_loss)
        avg_eval_accuracy = total_eval_accuracy / num_eval_steps

        # Report the final accuracy for this validation run.
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train loss: {avg_train_loss:.5f} ' +
                     f'valid loss: {avg_eval_loss:.5f} ' +
                     f'train acc: {avg_train_accuracy:.5f} ' +
                     f'valid acc: {avg_eval_accuracy:.5f} ' +
                     f'train time: {elapsed_train_time} ' +
                     f'valid time: {elapsed_eval_time}')
        
        # Report the statistics for this epoch's validation run.
        logging.info(print_msg)
        
        # Check if the validation loss is no longer improving(i.e decreasing) after "patience" consecutive times
        if patience is not None:
            early_stopping(avg_eval_loss, model)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break
                
        
    # Evaluate model
    evaluate_model(model, test_dataloader, save_path, logging)
    # Save model
    torch.save(model.state_dict(), save_path + name)
    logging.info(f"model {name} saved at {save_path}")

    
def main():
    pass

if __name__ == "__main__":
    main()
