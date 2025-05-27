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
from bert_utils import TASK, PROMPT_TECHNIQUE, CLASS_TYPE

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
                    tweets.append(tweet)
                    uid = uid.strip()
                    self.uids.append(uid)
                    dialect_label = dialect_label.strip()
                    self.dialect_labels.append(int(dialect_label))
                    
                    if TASK == "human label" and CLASS_TYPE == "multi class":
                        self.labels.append(int(human_label))
                    elif TASK == "human label" and CLASS_TYPE == "binary":
                        self.labels.append(int(human_label_binary))
                    elif TASK == "gpt label" and CLASS_TYPE == "multi class":
                        self.labels.append(int(gpt_label))
                    elif TASK == "gpt label" and CLASS_TYPE == "binary":
                        self.labels.append(int(gpt_label_binary))

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
    

def flat_accuracy(preds, labels):
    """
    Calculate the accuracy using the predicted values and the true labels
    Code adapted from https://osf.io/qkjuv/
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
    Code adapted from https://osf.io/qkjuv/
    '''

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate_model(model, test_dataloader, save_path, with_regularization=False):
    
    if CLASS_TYPE == "multi class":
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
    else:
        dataset_label_map = {
            "hovy": ["normal", "offensive"],
            "waseem": ["normal", "offensive"],
            "founta": ["normal", "offensive"],
            "davidson": ["normal", "offensive"],
            "golbeck": ["normal", "offensive"],
            "hateval": ["normal", "offensive"],
            "offenseval": ["normal", "offensive"],
            "abuseval": ["normal", "offensive"],
            "data": ["normal", "offensive"] # Combined data is binary
        }
    
    class_labels = {
        "hovy": [0, 1, 2],
        "waseem": [0, 1, 2, 3],
        "founta": [0, 1, 2],
        "davidson": [0, 1, 2],
        "golbeck": [0, 1],
        "hateval": [0, 1],
        "offenseval": [0,  1],
        "abuseval": [0, 1, 2],
        "data": [0, 1] # Combined data is binary
    }
    
    print('Evaluating trained model ................')
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

        with torch.no_grad():
            if with_regularization:
                outputs, _ = model(batch)
                logits = outputs.logits  
            else:
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
    
    dataset_name = save_path.split('/')[-2].split('_')[-1]
    dataset_classes = dataset_label_map[dataset_name]
    print(f"{classification_report(gold_labels, predicted_labels, digits=4, target_names=dataset_classes, labels=class_labels[dataset_name])}")  
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    
    print(f"\nMacro F1: {macro_f1}")
    print(f"Macro Precision: {macro_precision}")
    print(f"Macro Recall: {macro_recall}")
    print(f"Micro F1: {micro_f1}")
    print(f"Micro Precision: {micro_precision}")
    print(f"Micro Recall: {micro_recall}")
    
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
        
        
def train(train_dataloader,
          test_dataloader,
          tokenizer,
          model,
          num_labels,
          learning_rate,
          epochs,
          save_path,
          classifier_criterion):

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

    # Store the average loss after each epoch so we can plot the learning curve 
    avg_train_losses = []
    avg_valid_losses = []
    
    # For each epoch...
    for epoch in range(epochs):
        # Store true lables for global eval
        gold_labels = []

        # Store predicted labels for global eval
        predicted_labels = []

        # Perform one full pass over the training set.
        print(f'Epoch {epoch + 1} / {epochs}')
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0.0
        total_train_accuracy = 0.0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print(f'Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}.')

            # Unpack this training batch from our dataloader
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            
            batch = (batch_input_ids, batch_input_mask, batch_labels)

            model.zero_grad()

            # Perform a forward pass
            outputs = model(batch)
            logits = outputs.logits
            loss = classifier_criterion(logits, batch_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            learning_rate_scheduler.step()
            
            # Get predictions and move predictions and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            train_accuracy = flat_accuracy(logits, label_ids)
            # Accumulate the total accuracy.
            total_train_accuracy += train_accuracy
            
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        # Store the loss value for plotting the learning curve.
        avg_train_losses.append(avg_train_loss)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        elapsed_train_time = format_time(time.time() - t0)

        # After the completion of each training epoch, measure our performance on
        # our validation set.

        t0 = time.time()

        # Put the model in evaluation mode
        model.eval()

        # Tracking variables
        eval_accuracy = 0
        num_eval_steps = 0
        running_val_loss = 0.0
        
        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU/CPU
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            batch = (batch_input_ids, batch_input_mask, None)

            with torch.no_grad():
                outputs = model(batch)
                logits = outputs.logits 
            
                probs = nn.functional.softmax(logits, dim=-1)
            
            # Calculate the loss
            val_loss = classifier_criterion(logits, batch_labels)
            # Accumulate validation loss
            running_val_loss += val_loss
            
            # Move probabilities to CPU
            probs = probs.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(probs, label_ids)
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            num_eval_steps += 1
            
            predictions = np.argmax(probs, axis=1).flatten()
            label_ids = label_ids.flatten()
            
            # Store gold labels single list
            gold_labels.extend(label_ids)
            # Store predicted labels single list
            predicted_labels.extend(predictions)
        
        elapsed_valid_time = format_time(time.time() - t0)
        avg_val_loss = running_val_loss / num_eval_steps
        avg_valid_losses.append(avg_val_loss)
        avg_valid_accuracy = eval_accuracy / num_eval_steps
        
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train loss: {avg_train_loss:.5f} ' +
                     f'valid loss: {avg_val_loss:.5f} ' +
                     f'train acc: {avg_train_accuracy:.5f} ' +
                     f'valid acc: {avg_valid_accuracy:.5f} ' +
                     f'train time: {elapsed_train_time} ' +
                     f'valid time: {elapsed_valid_time}')
        
        # Report the statistics for this epoch's validation run.
        print(print_msg)
        
    # Evaluate model
    evaluate_model(model, test_dataloader, save_path)
    
    # Make dir for model serializations
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    name = save_path.split('/')[-2] + '.pth'
    torch.save(model.state_dict(), save_path + name)
    print(f"model {name} saved at {save_path}")
    
    
def main():
    pass

if __name__ == "__main__":
    main()
