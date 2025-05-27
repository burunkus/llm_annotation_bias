#!/bin/bash

#SBATCH --job-name utils
#SBATCH --output /home/eokpala/llm_annotation_bias/logs/data_stats_dialect_inference_gpt_annotated_out.txt
#SBATCH --error /home/eokpala/llm_annotation_bias/logs/data_stats_dialect_inference_gpt_annotated_err.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
##SBATCH --gpus-per-node v100:1
#SBATCH --mem 30gb
#SBATCH --time 02:00:00
#SBATCH --constraint interconnect_hdr
#SBATCH --mail-user=eokpala@clemson.edu
#SBATCH --mail-type=FAIL,BEGIN,END

module add cuda/11.8.0 
module add anaconda3/2023.09-0

source activate pt_cuda_env

cd /home/eokpala/llm_annotation_bias/
srun python utils.py

