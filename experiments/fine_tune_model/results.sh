#!/bin/bash

#SBATCH --job-name perf_results
#SBATCH --output /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/logs/results_out.txt
#SBATCH --error /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/logs/results_err.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
##SBATCH --gpus-per-node v100:1
#SBATCH --mem 10gb
#SBATCH --time 1:00:00
#SBATCH --constraint interconnect_hdr
#SBATCH --mail-user=eokpala@clemson.edu
#SBATCH --mail-type=FAIL,BEGIN,END

module add cuda/12.3.0
module add anaconda3/2023.09-0

source activate pytorch_env

cd /home/eokpala/llm_annotation_bias/experiments/fine_tune_model/
srun python results.py
