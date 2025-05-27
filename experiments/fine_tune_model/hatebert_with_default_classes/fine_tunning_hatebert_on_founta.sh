#!/bin/bash

#SBATCH --job-name hb_on_founta
#SBATCH --output /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/hatebert_with_default_classes/logs/hb_on_founta_out.txt
#SBATCH --error /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/hatebert_with_default_classes/logs/hb_on_founta_err.txt
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

cd /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/hatebert_with_default_classes/
srun python fine_tunning_hatebert_on_founta.py

