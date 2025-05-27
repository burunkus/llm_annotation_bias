import time
import random
import json
import csv
import os
import sys
import logging
import logging.handlers
import transformers
from transformers import pipeline
from huggingface_hub import login
import torch

# Ensure you 1) have access to the model (request access on huggingface) 2) create a token on huggingface 3) add token by running huggingface-cli login on terminal

#login(token = "enter your token")

print(f"Transformers version: {transformers.__version__}")

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SEED = 23
MAX_TOKEN = 512
TEMPERATURE = 0.2
KEY = "enter your token"

dataset_label_map = {
    "waseem_and_hovy": {"racism": 1, "sexism": 2, "neither": 0},
    "waseem": {"racism": 1, "sexism": 2, "racism and sexism": 3, "neither": 0},
    "founta": {"hateful": 1, "abusive": 2, "normal": 0},
    "davidson": {"hate": 1, "offensive": 2, "normal": 0}, #Note that in the process of extracting Davidson, the labels were changed e.g hate (0) -> (1)
    "golbeck": {"harassment": 1, "not harassment": 0},
    "hateval": {"hate": 1, "not hate": 0},
    "offenseval": {"offensive": 1, "not offensive": 0},
    "abuseval": {"explicit abuse": 1, "implicit abuse": 2, "not abusive": 0}
}

dataset_label_int_to_str_map = {
    "waseem_and_hovy": {1: "racism", 2: "sexism", 0: "neither"},
    "waseem": {1: "racism", 2: "sexism", 3: "racism and sexism", 0: "neither"},
    "founta": {1: "hateful", 2: "abusive", 0: "normal"},
    "davidson": {1: "hate", 2: "offensive", 0: "normal"}, 
    "golbeck": {1: "harassment", 0: "not harassment"},
    "hateval": {1: "hate", 0: "not hate"},
    "offenseval": {1: "offensive", 0: "not offensive"},
    "abuseval": {1: "explicit abuse", 2: "implicit abuse", 0: "not abusive"}
}

classes_map = {
    "waseem_and_hovy": "racism or sexism or neither",
    "waseem": "racism or sexism or racism and sexism or neither",
    "founta": "hateful or abusive or normal",
    "davidson": "hate or offensive or normal",
    "golbeck": "harassment or not harassment",
    "hateval": "hate or not hate",
    "offenseval": "offensive or not offensive",
    "abuseval": "explicit abuse or implicit abuse or not abusive"
}

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


def general_prompt_annotation(data_path, save_path, dataset):
    
    absolute_path, file_name = os.path.split(save_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
    
    unannotated_tweets = []
    
    with open(data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    # Save memory
    data = None
    
    pipe = pipeline(
        "text-generation",
        model=MODEL,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Write annotation to save_path 
    with open(save_path, 'w') as save_file_handle:
        # Annotate each tweet in data_path
        with open(data_path) as file_handle:
            for i, row in enumerate(file_handle):
                logging.info(f"Annotating [{i + 1}/{dataset_size}]")
                
                line = row.split('\t')
                tweet_id = line[0]
                tweet = line[1]
                human_label = line[2].strip()
                dialect_label = line[3].strip()
                
                # Insert tweet in prompt
                dialect = None
                if dialect_label == '1':
                    dialect = "African American English"
                else:
                    dialect == "Standard American English"
                    
                prompt = f'Given the tweet in triple quotes: """{tweet}""" likely written in {dialect}. Do you think the tweet is {classes_map[dataset]}? Only answer with one of the following: {classes_map[dataset]}. Do not provide an explanation for your answer.'
                messages = [{"role": "user", "content": prompt}]
                
                try:
                    response = pipe(
                        messages,
                        max_new_tokens=MAX_TOKEN,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=0.9,
                    )
                    response_content = response[0]["generated_text"][-1]["content"].lower()
                    # Write response to file
                    save_file_handle.write(f"{tweet_id}\t{tweet}\t{human_label}\t{dataset_label_map[dataset][response_content]}\t{dialect_label}\n")
                except KeyError as err:
                    logging.info(f'Caught exception {err} during annotation.')
                    # Track unannotated tweet
                    unannotated_tweets.append(row)
    
    # If some tweets were annotated due to Key error because LLaMa didn't follow the instruction of producing only one class from the given classes
    if len(unannotated_tweets) > 0:
        logging.info(f"Unannotated tweets due to error: {len(unannotated_tweets)}")
        name = save_path.split('.txt')[0] + '_unannotated.txt'
        with open(name, 'w') as save_file_handle:
            for line in unannotated_tweets:
                save_file_handle.write(line)
                
                
def re_annotate(unannotated_data_path, annotated_data_path, dataset):
    
    unannotated_tweets = []
    
    with open(unannotated_data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    # Save memory
    data = None
    
    pipe = pipeline(
        "text-generation",
        model=MODEL,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Open the annotated data in append mode 
    with open(annotated_data_path, 'a') as save_file_handle:
        # re-annotate each tweet in unannotated_data_path
        with open(unannotated_data_path) as file_handle:
            for i, row in enumerate(file_handle):
                logging.info(f"Annotating [{i + 1}/{dataset_size}]")
                
                line = row.split('\t')
                tweet_id = line[0]
                tweet = line[1]
                human_label = line[2].strip()
                dialect_label = line[3].strip()
                
                # Insert tweet in prompt
                dialect = None
                if dialect_label == '1':
                    dialect = "African American English"
                else:
                    dialect == "Standard American English"
                    
                prompt = f'Given the tweet in triple quotes: """{tweet}""" likely written in {dialect}. Do you think the tweet is {classes_map[dataset]}? Only answer with one of the following: {classes_map[dataset]}. Do not provide an explanation for your answer.'
                messages = [{"role": "user", "content": prompt}]
                
                try:
                    response = pipe(
                        messages,
                        max_new_tokens=MAX_TOKEN,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=0.9,
                    )
                    response_content = response[0]["generated_text"][-1]["content"].lower()
                    if response_content == "sexism and racism":
                        response_content = "racism and sexism"
                        
                    # Write response to file
                    save_file_handle.write(f"{tweet_id}\t{tweet}\t{human_label}\t{dataset_label_map[dataset][response_content]}\t{dialect_label}\n")
                except KeyError as err:
                    logging.info(f'Caught exception {err} during annotation.')
                    # Track unannotated tweet
                    unannotated_tweets.append(row)
    
    # If some tweets were annotated due to Key error because LLaMa didn't follow the instruction of producing only one class from the given classes
    if len(unannotated_tweets) > 0:
        logging.info(f"Unannotated tweets due to error: {len(unannotated_tweets)}")
        
        absolute_path, file_name = os.path.split(unannotated_data_path)
        name = ""
        try:
            unannotated_count = int(file_name.split("_")[-1].split(".")[0])
            file_name = "_".join(file_name.split('_')[:-1])
            name = absolute_path + '/' + file_name + f'_{str(unannotated_count + 1)}.txt'
        except ValueError as err:
            name = unannotated_data_path.split('.txt')[0] + '_1.txt'
            
        with open(name, 'w') as save_file_handle:
            for line in unannotated_tweets:
                save_file_handle.write(line)


def process_reannotation():
    
    PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/general_prompt_annotation/"
    unannotated_data_path = PATH + "waseem/waseem_with_default_classes_train_unannotated.txt"
    annotated_data_path = PATH + "waseem/waseem_with_default_classes_train.txt"
    dataset = "waseem"
    re_annotate(unannotated_data_path, annotated_data_path, dataset)
    
    
def main():
    DATASET_PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data_for_dialect_priming/"
    SAVE_PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/general_prompt_annotation/"
    data_paths = [
        ("founta/founta_with_default_classes_train.txt", 'founta'),
        ("founta/founta_with_default_classes_test.txt", 'founta'),
        ("davidson/davidson_with_default_classes_train.txt", 'davidson'),
        ("davidson/davidson_with_default_classes_test.txt", 'davidson'),
        ("golbeck/golbeck_with_default_classes_train.txt", 'golbeck'),
        ("golbeck/golbeck_with_default_classes_test.txt", 'golbeck'),
        ("hateval2019/hateval2019_en_train.txt", 'hateval'),
        ("hateval2019/hateval2019_en_test.txt", 'hateval'),
        ("offenseval2019/offenseval_train.txt", 'offenseval'),
        ("offenseval2019/offenseval_test.txt", 'offenseval'),
    ]
    
    for file_name, dataset in data_paths:
        data_path = DATASET_PATH + file_name
        save_path = SAVE_PATH + file_name
        general_prompt_annotation(data_path, save_path, dataset)
    
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
   
    # LLaMa 3 annotation needs transformers 4.48.0, torch 2.5.1, huggingface hub 0.27.1
    main() 
    #process_reannotation()
