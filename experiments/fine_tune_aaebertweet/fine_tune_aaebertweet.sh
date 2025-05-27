#!/bin/bash

#SBATCH --job-name aaebertweet
#SBATCH --output /home/eokpala/llm_annotation_bias/experiments/fine_tune_aaebertweet/logs/aaebertweet_out.txt
#SBATCH --error /home/eokpala/llm_annotation_bias/experiments/fine_tune_aaebertweet/logs/aaebertweet_err.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 20gb
#SBATCH --time 72:00:00
#SBATCH --constraint interconnect_hdr
#SBATCH --mail-user=eokpala@clemson.edu
#SBATCH --mail-type=FAIL,BEGIN,END

module add cuda/12.3.0
module add anaconda3/2023.09-0

source activate pytorch_env

cd /home/eokpala/experiments/fine_tune_aaebertweet/
srun python fine_tune_aaebertweet.py
