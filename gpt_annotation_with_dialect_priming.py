import time
import random
import json
import csv
import os
import sys
import logging
import logging.handlers
from openai import OpenAI

MODEL = "gpt-4"
SEED = 23
MAX_TOKEN = 512
TEMPERATURE = 0.2
KEY = "Enter your OpenAI key here"
CLIENT = OpenAI(api_key=KEY)

dataset_label_map = {
    "waseem_and_hovy": {"racism": 1, "sexism": 2, "neither": 0},
    "waseem": {"racism": 1, "sexism": 2, "racism and sexism": 3, "neither": 0},
    "founta": {"hateful": 1, "abusive": 2, "normal": 0},
    "davidson": {"hate": 1, "offensive": 2, "normal": 0},
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


def general_prompt_annotation(data_path, save_path, dataset, max_retries=3, delay=1):
    
    absolute_path, file_name = os.path.split(save_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        
    dataset_label_map = {
        "waseem_and_hovy": {"racism": 1, "sexism": 2, "neither": 0},
        "waseem": {"racism": 1, "sexism": 2, "racism and sexism": 3, "neither": 0},
        "founta": {"hateful": 1, "abusive": 2, "normal": 0},
        "davidson": {"hate": 1, "offensive": 2, "normal": 0},
        "golbeck": {"harassment": 1, "not harassment": 0},
        "hateval": {"hate": 1, "not hate": 0},
        "offenseval": {"offensive": 1, "not offensive": 0},
        "abuseval": {"explicit abuse": 1, "implicit abuse": 2, "not abusive": 0}
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

    unannotated_tweets = []
    
    with open(data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    # Save memory
    data = None
    
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

                # Exponential backoff to control rate limit
                num_retries = 0
                backoff_delay = delay
                exponential_base = 2
                jitter = True

                while True:
                    try:
                        # Progressively add delay between request
                        time.sleep(delay)

                        # Make a request 
                        response = CLIENT.chat.completions.create(
                            model=MODEL,
                            seed=SEED,
                            max_tokens=MAX_TOKEN,
                            temperature=TEMPERATURE,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response_content = response.choices[0].message.content.lower()
                        
                        # Write response to file
                        save_file_handle.write(f"{tweet_id}\t{tweet}\t{human_label}\t{dataset_label_map[dataset][response_content]}\t{dialect_label}\n")

                        break
                    except Exception as e:
                        logging.info(f'Caught exception {e} during annotation.')
                        num_retries += 1
                        if num_retries > max_retries:
                            logging.info(f"Too many consecutive retries: {num_retries}, stopping.")
                            # Track unannotated tweet
                            unannotated_tweets.append(row)
                            break
                        else:
                            backoff_delay *= exponential_base * (1 + jitter * random.random())
                            time.sleep(backoff_delay)
          
    # If some tweets were annotated due to execeeding maximum number of retries, write them to file for reannotation
    if len(unannotated_tweets) > 0:
        logging.info(f"Unannotated tweets due to error: {len(unannotated_tweets)}")
        name = save_path.split('.txt')[0] + '_unannotated.txt'
        with open(name, 'w') as save_file_handle:
            for line in unannotated_tweets:
                save_file_handle.write(line)


def few_shot_prompt_annotation(data_path, save_path, dataset, few_shot_examples_path, max_retries=3, delay=1.5):
    
    absolute_path, file_name = os.path.split(save_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    with open(data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    # Save memory
    data = None
    
    logging.info(f"Creating prompt with few shot examples ....")
    few_shot_examplars = []
    with open(few_shot_examples_path, 'r') as few_shot_handle:
        for i, row in enumerate(few_shot_handle):
            line = row.split('\t')
            tweet_id = line[0]
            tweet = line[1]
            human_label = dataset_label_int_to_str_map[dataset][int(line[2].strip())]
            dialect_label = line[3].strip()
            
            dialect = None
            if dialect_label == '1':
                dialect = "African American English"
            else:
                dialect == "Standard American English"
                    
            question_answer = f'Question: is the tweet in triple quotes """{tweet}""" likely written in {dialect} {classes_map[dataset]}? Only respond with {classes_map[dataset]}. Answer: {human_label}.'
            few_shot_examplars.append(question_answer)
            
    logging.info(f'Examplars: {" ".join(few_shot_examplars)}')
    
    unannotated_tweets = []
    
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
                
                dialect = None
                if dialect_label == '1':
                    dialect = "African American English"
                else:
                    dialect == "Standard American English"
                    
                # Insert tweet in prompt
                question = f'Question: is the tweet in triple quotes """{tweet}""" likely written in {dialect} {classes_map[dataset]}? Only respond with {classes_map[dataset]}. Answer:'
                prompt = " ".join(few_shot_examplars) + " " + question

                # Exponential backoff to control rate limit
                num_retries = 0
                backoff_delay = delay
                exponential_base = 2
                jitter = True

                while True:
                    try:
                        # Progressively add delay between request
                        time.sleep(delay)

                        # Make a request 
                        response = CLIENT.chat.completions.create(
                            model=MODEL,
                            seed=SEED,
                            max_tokens=MAX_TOKEN,
                            temperature=TEMPERATURE,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response_content = response.choices[0].message.content.lower()
                        # remove the dot ('.') at the end of answer
                        if response_content[-1] == '.':
                            response_content = response_content[:-1]
                        
                        # Write response to file
                        save_file_handle.write(f"{tweet_id}\t{tweet}\t{human_label}\t{dataset_label_map[dataset][response_content]}\t{dialect_label}\n")

                        break
                    except Exception as e:
                        logging.info(f'Caught exception {e} during annotation.')
                        num_retries += 1
                        if num_retries > max_retries:
                            logging.info(f"Too many consecutive retries: {num_retries}, stopping.")
                            # Track unannotated tweet
                            unannotated_tweets.append(row)
                            break
                        else:
                            backoff_delay *= exponential_base * (1 + jitter * random.random())
                            time.sleep(backoff_delay)

    # If some tweets were annotated due to execeeding maximum number of retries, write them to file for reannotation
    if len(unannotated_tweets) > 0:
        logging.info(f"Unannotated tweets due to error: {len(unannotated_tweets)}")
        name = save_path.split('.txt')[0] + '_unannotated.txt'
        with open(name, 'w') as save_file_handle:
            for line in unannotated_tweets:
                save_file_handle.write(line)
    

def chain_of_thought_prompt_annotation(data_path, save_path, dataset, max_retries=3, delay=1.5, write_or_append='w'):
    
    absolute_path, file_name = os.path.split(save_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    with open(data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    # Save memory
    data = None
    
    few_shot_example_normal1 = '@USER True Liberals do'
    few_shot_example_normal2 = "@USER I know! She is still alive....but wondering if her career was railroaded? Odd! Loved her...especially that day!"
    few_shot_example_hate1 = "@USER @USER Bitch I wanna know"
    few_shot_example_hate2 = "#ThursdayThoughts- FUCK liberals. Forever."

    cot_example_normal1 = f'Question: Is the tweet in triple quotes """{few_shot_example_normal1}""" likely written in African American English {classes_map[dataset]}? Answer with only one of the following classes: {classes_map[dataset]}. The answer should be in the last words of your final sentence after your explanation. Answer: This tweet, likely written in African American English, is not offensive because it does not contain profanity. So, the answer is not offensive.'
    cot_example_normal2 = f'Question: Is the tweet in triple quotes """{few_shot_example_normal2}""" likely written in Standard American English {classes_map[dataset]}? Answer with only one of the following classes: {classes_map[dataset]}. The answer should be in the last words of your final sentence after your explanation. Answer: This tweet, likely written in Standard American English, is not offensive because it does not contain profanity. So, the answer is not offensive.'
    cot_example_hate1 = f'Question: Is the tweet in triple quotes """{few_shot_example_hate1}""" likely written in African American English {classes_map[dataset]}? Answer with only one of the following classes: {classes_map[dataset]}. The answer should be in the last words of your final sentence after your explanation. Answer: This tweet, likely written in African American English, is offensive because it contains an unacceptable language, a profanity, Bitch directed at targets. So, the answer is offensive.'
    cot_example_hate2 = f'Question: Is the tweet in triple quotes """{few_shot_example_hate2}""" likely written in Standard American English {classes_map[dataset]}? Answer with only one of the following classes: {classes_map[dataset]}. The answer should be in the last words of your final sentence after your explanation. Answer: This tweet, likely written in Standard American English, is offensive because it contains an unacceptable language, a swear word, Fuck. So, the answer is offensive.'

    cot_exemplars = [cot_example_normal1, 
                     cot_example_normal2,
                     cot_example_hate1,
                     cot_example_hate2
                    ]
    
    unannotated_tweets = []
    unannotated_tweets_response = []
    
    # Write annotation to save_path 
    with open(save_path, write_or_append) as save_file_handle:
        # Annotate each tweet in data_path
        with open(data_path) as file_handle:
            for i, row in enumerate(file_handle):
                logging.info(f"Annotating [{i + 1}/{dataset_size}]")
                
                line = row.split('\t')
                tweet_id = line[0]
                tweet = line[1]
                human_label = line[2].strip()
                dialect_label = line[3].strip()
                
                dialect = None
                if dialect_label == '1':
                    dialect = "African American English"
                else:
                    dialect == "Standard American English"
                    
                # Insert tweet in prompt
                question = f'Question: Is the tweet in triple quotes """{tweet}""" likely written in {dialect} {classes_map[dataset]}? Answer with only one of the following classes: {classes_map[dataset]}. The answer should be in the last words of your final sentence after your explanation. Answer:'
                prompt = " ".join(cot_exemplars) + " " + question

                # Exponential backoff to control rate limit
                num_retries = 0
                backoff_delay = delay
                exponential_base = 2
                jitter = True

                while True:
                    try:
                        # Progressively add delay between request
                        time.sleep(delay)

                        # Make a request 
                        response = CLIENT.chat.completions.create(
                            model=MODEL,
                            seed=SEED,
                            max_tokens=MAX_TOKEN,
                            temperature=TEMPERATURE,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response_content = response.choices[0].message.content.lower()
                        # remove the dot ('.') at the end of answer
                        if response_content[-1] == '.':
                            response_content = response_content[:-1]
                        
                        answer_start_index = response_content.find('the answer is')
                        if answer_start_index > 0:
                            answer_sentence = response_content[answer_start_index:]
                            answer_sentence_list = answer_sentence.split()
                            label_start_index = answer_sentence_list.index('is')

                            label_list = answer_sentence_list[label_start_index + 1:]
                            if len(label_list) == 1:
                                gpt_label = label_list[0]

                                if gpt_label == "undeterminable" and dataset == "hateval":
                                    gpt_label = "not hate"
                            else:
                                gpt_label = " ".join(label_list)
                            
                        # Write response to file
                        save_file_handle.write(f"{tweet_id}\t{tweet}\t{human_label}\t{dataset_label_map[dataset][gpt_label]}\t{dialect_label}\n")

                        break
                    except Exception as e:
                        logging.info(f'Caught exception {e} during annotation.')
                        num_retries += 1
                        if num_retries > max_retries:
                            logging.info(f"Too many consecutive retries: {num_retries}, stopping.")
                            # Track unannotated tweet
                            unannotated_tweets.append(row)
                            unannotated_tweets_response(row.strip() + '\t' + " ".join(response_content))
                            break
                        else:
                            backoff_delay *= exponential_base * (1 + jitter * random.random())
                            time.sleep(backoff_delay)

    # If some tweets were annotated but wasn't written to file due to the GPT label not being at the end of the last sentence by reaching execeeding maximum number of retries, write them to file for reannotation
    if len(unannotated_tweets) > 0:
        logging.info(f"Unannotated tweets due to error: {len(unannotated_tweets)}")
        name = save_path.split('.txt')[0] + '_unannotated.txt'
        with open(name, 'w') as save_file_handle:
            for line in unannotated_tweets:
                save_file_handle.write(line)
                
    # If some tweets were annotated but wasn't written to file due to the GPT label not being at the end of the last sentence, write them to file for manual extraction of the annotation
    if len(unannotated_tweets_response) > 0:
        logging.info(f"Unannotated tweets due to error2: {len(unannotated_tweets_response)}")
        name = save_path.split('.txt')[0] + '_unannotated_for_manual.txt'
        with open(name, 'w') as save_file_handle:
            for line in unannotated_tweets_response:
                save_file_handle.write(f'{line}\n')
                
                
def re_annotate(unannotated_data_path, annotated_data_path, dataset, max_retries=3, delay=1):
    
    classes_map = {
    "waseem_and_hovy": "racism or sexism or neither",
    "waseem": "racism or sexism or racism and sexism or neither",
    "founta": "hateful or abusive or normal",
    "davidson": "hate, offensive or normal",
    "golbeck": "harassment or not harassment",
    "hateval": "hate or not hate",
    "offenseval": "offensive or not offensive",
    "abuseval": "explicit abuse or implicit abuse or not abusive"
    }
    
    unannotated_tweets = []
    
    with open(unannotated_data_path, 'r') as file_handle:
        data = file_handle.readlines()
    dataset_size = len(data)
    data = None

    # Write annotation to save_path 
    with open(annotated_data_path, 'a') as save_file_handle:
        # Annotate each tweet in data_path
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

                # Exponential backoff to control rate limit
                num_retries = 0
                backoff_delay = delay
                exponential_base = 2
                jitter = True

                while True:
                    try:
                        # Progressively add delay between request
                        time.sleep(delay)

                        # Make a request 
                        response = CLIENT.chat.completions.create(
                            model=MODEL,
                            seed=SEED,
                            max_tokens=MAX_TOKEN,
                            temperature=TEMPERATURE,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response_content = response.choices[0].message.content.lower()
                        
                        # Write response to file
                        save_file_handle.write(f"{tweet_id}\t{tweet}\t{human_label}\t{dataset_label_map[dataset][response_content]}\t{dialect_label}\n")

                        break
                    except Exception as e:
                        logging.info(f'Caught exception {e} during annotation.')
                        num_retries += 1
                        if num_retries > max_retries:
                            logging.info(f"Too many consecutive retries: {num_retries}, stopping.")
                            # Track unannotated tweet
                            unannotated_tweets.append(row)
                            break
                        else:
                            backoff_delay *= exponential_base * (1 + jitter * random.random())
                            time.sleep(backoff_delay)
               
    # If some tweets were annotated due to execeeding maximum number of retries, write them to file for reannotation
    if len(unannotated_tweets) > 0:
        logging.info(f"Unannotated tweets due to error: {len(unannotated_tweets)}")
        name = unannotated_data_path.split('.txt')[0] + '_unannotated.txt'
        with open(name, 'w') as save_file_handle:
            for line in unannotated_tweets:
                save_file_handle.write(line)
    

def process_reannotation():
    
    PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/general_prompt_annotation/"
    unannotated_data_path = PATH + "davidson/davidson_with_default_classes_train_unannotated.txt"
    annotated_data_path = PATH + "davidson/davidson_with_default_classes_train.txt"
    dataset = "davidson"
    re_annotate(unannotated_data_path, annotated_data_path, dataset, max_retries=3, delay=1)

    
def process_cot_reannotation():
    
    PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/cot_prompt_annotation/"
    unannotated_data_path = PATH + "davidson/davidson_with_default_classes_train_unannotated.txt"
    annotated_data_path = PATH + "davidson/davidson_with_default_classes_train.txt"
    dataset = "davidson"
    chain_of_thought_prompt_annotation(unannotated_data_path, annotated_data_path, dataset, max_retries=3, delay=1.5, write_or_append='a')
    

def main():
    
    prompt_type = "cot_prompt_annotation"
    DATASET_PATH = "/project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data_for_dialect_priming/"
    SAVE_PATH = f"/project/luofeng/socbd/eokpala/llm_annotation_bias/data_dialect_priming/{prompt_type}/"
    data_path = DATASET_PATH + "offenseval2019/offenseval_train.txt"
    save_path = SAVE_PATH + "offenseval2019/offenseval_train.txt"
    dataset = "offenseval"
    # Each tweet in save_path will be of the form id \t tweet \t label \t gpt-label \t dialect-label
    #general_prompt_annotation(data_path, save_path, dataset, max_retries=3, delay=1)
    #few_shot_examples_path = SAVE_PATH + "offenseval2019/offenseval_train_exemplars_2_samples_balanced_dialect.txt"
    #few_shot_prompt_annotation(data_path, save_path, dataset, few_shot_examples_path, max_retries=3, delay=1.5)
    chain_of_thought_prompt_annotation(data_path, save_path, dataset, max_retries=4, delay=1.5)


if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
   
    main()
    #process_reannotation()
    #process_cot_reannotation()
    