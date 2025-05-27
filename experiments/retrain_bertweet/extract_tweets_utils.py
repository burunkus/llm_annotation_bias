import re
import random 
import emoji
import string
from sklearn.model_selection import train_test_split


def count_tweets(file_name):
    with open(file_name, 'r') as file:
        for i, line in enumerate(file, 1):
            continue
        print(f"{file_name}: {i}\n")


def count_dialects(file_name):
    num_aae, num_none_aae = 0, 0
    with open(file_name) as file_handler:
        for i, line in enumerate(file_handler, 1):
            tweet, label = line.split("\t")
            tweet = tweet.strip()
            label = label.strip()

            if label == '1':
                num_aae += 1
            elif label == '0':
                num_none_aae += 1
        
        print(f"{file_name}: {i}")
        print(f"AAE: {num_aae}, None-AAE: {num_none_aae}\n")
            

def sample_tweets(file, save_sampled_as, k=1000):
    """
    Randomly sample k tweets from file and exclude the sampled tweets
    Args:
        file: String. The file (path + file name) to sample tweets from. Each line is of
        the form - tweetID authorID tweet
        save_sampled_as: String. The file (path + file name) name to save the sampled tweets
        save_as: String. The file (path + file name) name to save tweets not sampled i.e dataset - sampled
    """
    with open(file, 'r') as race_aligned_file:
        all_tweets = race_aligned_file.readlines()
    
    seen = set()
    unique_all_tweets = []
    # Ensure we are sampling from unique tweets
    for tweet in all_tweets:
        if tweet not in seen:
            unique_all_tweets.append(tweet)
            seen.add(tweet)
            
    seed = 5
    random.seed(seed)
    sampled_tweets = random.sample(unique_all_tweets, k=k)
    # save sampled tweets 
    with open(save_sampled_as, 'w') as sample_file:
        for tweet in sampled_tweets:
            sample_file.write(tweet)


def sample_from_black_aligned(data_file, save_sampled_as, save_remaining_as, k=1000):
    """
    Randomly sample k tweets from black aligned dataset and save the remaining tweets as training set
    Args:
        data_file: String. The file (path + file name) to sample tweets from. Each line is of
        the form separated by tab - tweetID authorID tweet
        save_sampled_as: String. The file (path + file name) name to save the sampled tweets
        save_as: String. The file (path + file name) name to save tweets not sampled i.e dataset - sampled
    """
    
    with open(data_file, 'r') as file_handler:
        all_tweets = file_handler.readlines()
    
    # Ensure we are sampling from unique tweets
    seen = set()
    unique_all_tweets = []
    for tweet in all_tweets:
        if tweet not in seen:
            unique_all_tweets.append(tweet)
            seen.add(tweet)
            
    seed = 5
    random.seed(seed)
    sampled_tweets = random.sample(unique_all_tweets, k=k)
    # save sampled tweets 
    with open(save_sampled_as, 'w') as sample_file:
        for tweet in sampled_tweets:
            sample_file.write(tweet)
    
    # Get the tweets in all_tweets that are not in sampled_tweets i.e the remaining tweets not among the sampled
    sampled_tweets = set(sampled_tweets)
    remaining_tweets = set(unique_all_tweets) - sampled_tweets
    
    # Save tweets not in sampled tweets
    with open(save_remaining_as, 'w') as remaining_tweet_file:
        for tweet in remaining_tweets:
            remaining_tweet_file.write(tweet)        


def split_dataset(data_path, save_train_as, save_test_as):
    """
    Splits a dataset into train and test set. This function assumes there is
    a file containing text and label in each line separated by tab
    Args:
        data_path (String): path where the full dataset test set is contained
        save_train_as (String): absolute path to the training set
        save_test_as (String): absolute path to the test set
    Returns:
        None
    """

    data, labels = [], []
    seed = 42

    with open(data_path, 'r') as file_handler:
        for i, line in enumerate(file_handler):
            line = line.split("\t")
            tweet = line[0].strip()
            label = line[1].strip()
            data.append(tweet)
            labels.append(label)
    
    # Create test set from train
    train_data, test_data, train_label, test_label = train_test_split(
            data, labels, train_size=0.80, random_state=seed
        )

    # Write train set
    with open(save_train_as, "w") as file_handler:
        for i, tweet in enumerate(train_data):
            label = train_label[i]
            file_handler.write(f"{tweet}\t{label}\n")

    # Write test set
    with open(save_test_as, "w") as file_handler:
        for i, tweet in enumerate(test_data):
            label = test_label[i]
            file_handler.write(f"{tweet}\t{label}\n")
            
            
def combine_black_and_white_aligned_with_labels(black_aligned_path, white_aligned_path, save_as):
    
    with open(save_as, 'w') as new_file_handle:
        with open(black_aligned_path, 'r') as black_aligned_handle:
            for tweet in black_aligned_handle:
                tweet = tweet.strip().strip("\n")
                new_file_handle.write(f"{tweet}\t{1}\n")
                
        with open(white_aligned_path, 'r') as white_aligned_handle:
            for tweet in white_aligned_handle:
                tweet = tweet.strip().strip("\n")
                new_file_handle.write(f"{tweet}\t{0}\n")


def preprocess_race_dataset(file_path, save_as):
    """
    Proprocess tweets by lower casing, normalize by converting
    user mentions to @USER, url to HTTPURL, and number to NUMBER 
    Convert emoji to text string and remove duplicate tweets
    Args:
        file_path(String): location of file to preprocess
        save_as(String): name to save the preproced tweet in
    Return:
        None
    """
    punct_chars = list((set(string.punctuation) | {
    "’", "‘", "–", "—", "~", "|", "“", "”", "…", "'", "`", "_",
    "“"
    }) - set(["#", "@"]))
    punct_chars.sort()
    punctuation = "".join(punct_chars)
    replace = re.compile("[%s]" % re.escape(punctuation))

    with open(save_as, 'w') as new_file:
        with open(file_path, 'r') as old_file:
            for i, line in enumerate(old_file, 1):
                tweet = line.strip()
                tweet = tweet.lower()
                # remove stock market tickers like $GE
                tweet = re.sub(r'\$\w*', '', tweet)
                # remove old style retweet text "RT"
                tweet = re.sub(r'^rt[\s]+', '', tweet)
                # replace hyperlinks with URL
                tweet = re.sub(r'(https?:\/\/[a-zA-Z0-9]+\.[^\s]{2,})', 'HTTPURL', tweet)
                # remove hashtags - only removing the hash # sign from the word
                tweet = re.sub(r'#', '', tweet)
                # replace emojis with emoji text
                tweet = emoji.demojize(tweet, delimiters=("", ""))
                tweet = re.sub(r'^\d+$', 'NUMBER', tweet)
                # replace handles with @USER
                tweet = re.sub(r'@\w+', '@USER', tweet)
                tweet = replace.sub(" ", tweet)
                # replace all whitespace with a single space
                tweet = re.sub(r"\s+", " ", tweet)
                # strip off spaces on either end
                tweet = tweet.strip()
                # remove tweets with less than 4 words
                word_list = tweet.split()
                if len(word_list) > 3:
                    new_file.write(f"{tweet}\n")


if __name__ == "__main__":
    
    # preprocess the white-aligned tweets
    file_path = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/white_aligned.txt"
    save_preprocessed_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/white_aligned_preprocessed.txt"
    preprocess_race_dataset(file_path, save_preprocessed_as)
    count_tweets(save_preprocessed_as)
    
    # Sample 1000 tweets from the white aligned tweets 
    save_sampled_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_white_aligned_preprocessed.txt"
    sample_tweets(save_preprocessed_as, save_sampled_as, k=1000)
    count_tweets(save_sampled_as)

    # preprocess the black aligned tweets
    file_path = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/black_aligned.txt"
    save_preprocessed_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/black_aligned_preprocessed.txt"
    preprocess_race_dataset(file_path, save_preprocessed_as)
    count_tweets(save_preprocessed_as)
    
    # Sample 1000 tweets from the white aligned tweets
    save_sampled_black_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_preprocessed.txt"
    save_train_set_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_train_set_preprocessed.txt"
    sample_from_black_aligned(save_preprocessed_as, save_sampled_black_as, save_train_set_as, k=1000)
    count_tweets(save_sampled_black_as)
    count_tweets(save_train_set_as)
    
    # combine sampled black and white aligned tweets for fine-tuning AAEBERTweet for dialect prediction
    black_aligned_path = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_preprocessed.txt"
    white_aligned_path = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_white_aligned_preprocessed.txt"
    save_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_and_white_aligned_combined_with_dialect_label_preprocessed.txt"
    combine_black_and_white_aligned_with_labels(black_aligned_path, white_aligned_path, save_as)
    count_tweets(save_as)
    
    # Split combined into train and test set
    file_path = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_and_white_aligned_combined_with_dialect_label_preprocessed.txt"
    save_train_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_and_white_aligned_combined_with_dialect_label_train_preprocessed.txt"
    save_test_as = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_and_white_aligned_combined_with_dialect_label_test_preprocessed.txt"
    split_dataset(file_path, save_train_as, save_test_as)
    count_dialects(save_train_as)
    count_dialects(save_test_as)