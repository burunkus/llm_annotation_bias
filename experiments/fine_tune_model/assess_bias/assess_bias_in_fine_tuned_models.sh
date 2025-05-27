#!/bin/bash

#SBATCH --job-name bias_boostrap
#SBATCH --output /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/assess_bias/logs/assess_bias_in_fine_tuned_models_out.txt
#SBATCH --error /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/assess_bias/logs/assess_bias_in_fine_tuned_models_err.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 20gb
#SBATCH --time 1:00:00
#SBATCH --constraint interconnect_hdr
#SBATCH --mail-user=eokpala@clemson.edu
#SBATCH --mail-type=FAIL,BEGIN,END

module add cuda/12.3.0
module add anaconda3/2023.09-0

source activate pytorch_env_new3

cd /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/assess_bias/
srun python assess_bias_in_fine_tuned_models.py

