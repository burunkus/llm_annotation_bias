import re
import math

def extract_tweet(path, data_file, save_black_aligned, save_white_aligned):
    """
    Extract tweet id, user id and tweet text from data_file (AAE dataset)
    Args:
        path: the location where file is located
        data_file: name of file
        save_as: name of file to save ids to
        black: Boolean, true if demographic is Black, False if White
    Returns:
        None
    """
    
    with (
        open(save_black_aligned, 'w') as black_aligned_file,
        open(save_white_aligned, 'w') as white_aligned_file,
    ):
        with open(path + data_file) as all_file:
            for i, line in enumerate(all_file, 1):
                line = line.split('\t')
                tweet_id = line[0]
                user_id = line[2]
                tweet = line[5]
                african_american = line[6]
                hispanic = line[7]
                other = line[8]
                white = line[9]
                
                # Check if any posterior is 'nan'
                if math.isnan(float(african_american)):
                    african_american = 0.0
                if math.isnan(float(hispanic)):
                    hispanic = 0.0
                if math.isnan(float(other)):
                    other = 0.0
                if math.isnan(float(white)):
                    white = 0.0

                if float(african_american) > 0.8:
                    black_aligned_file.write(f"{tweet}\n")
                elif float(white) > 0.8:
                     white_aligned_file.write(f"{tweet}\n")

def count_tweets(file_name):
    with open(file_name) as file:
        for i, line in enumerate(file, 1):
            continue
    print(f"File: {file_name}, Number of tweets: {i} \n")
    
if __name__ == '__main__':
    path = "../../../project/luofeng/socbd/eokpala/TwitterAAE-full-v1/"
    
    # Extract black-aligned tweets
    data_file = 'twitteraae_all'
    save_black_aligned_as = '../../../project/luofeng/socbd/eokpala/new_aaebert_experiment_data/black_aligned.txt'
    save_white_aligned_as = '../../../project/luofeng/socbd/eokpala/new_aaebert_experiment_data/white_aligned.txt'
    extract_tweet(path, data_file, save_black_aligned_as, save_white_aligned_as)
    
    #Count black/white aligned tweets
    count_tweets(save_black_aligned_as)
    count_tweets(save_white_aligned_as)
    