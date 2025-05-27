#!/bin/bash

#SBATCH --job-name extract_tweets
#SBATCH --output /home/eokpala/llm_annotation_bias/experiments/retrain_bertweet/logs/extract_tweets_out.txt
#SBATCH --error /home/eokpala/llm_annotation_bias/experiments/retrain_bertweet/logs/extract_tweets_out.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 125gb
#SBATCH --time 72:00:00
#SBATCH --constraint interconnect_hdr
#SBATCH --mail-user=eokpala@clemson.edu
#SBATCH --mail-type=FAIL,BEGIN,END

module add cuda/12.3.0
module add anaconda3/2023.09-0

source activate pytorch_env

cd /home/eokpala/experiments/retrain_bertweet/
srun python extract_tweets.py
