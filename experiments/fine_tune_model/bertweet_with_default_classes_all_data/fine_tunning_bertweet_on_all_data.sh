#!/bin/bash

#SBATCH --job-name bt_on_all_data
#SBATCH --output /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/bertweet_with_default_classes_all_data/logs/bt_on_all_data_out.txt
#SBATCH --error /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/bertweet_with_default_classes_all_data/logs/bt_on_all_data_err.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 10gb
#SBATCH --time 02:00:00
#SBATCH --constraint interconnect_hdr
#SBATCH --mail-user=eokpala@clemson.edu
#SBATCH --mail-type=FAIL,BEGIN,END

module add cuda/12.3.0
module add anaconda3/2023.09-0

source activate pytorch_env_new3

cd /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/bertweet_with_default_classes_all_data/
srun python fine_tunning_bertweet_on_all_data.py

