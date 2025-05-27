#PBS -N retrain_bertweet_on_aae
#PBS -o /home/eokpala/llm_annotation_bias/experiments/retrain_bertweet/logs/retrain_bertweet_on_aae_out.txt
#PBS -e /home/eokpala/llm_annotation_bias/experiments/retrain_bertweet/logs/retrain_bertweet_on_aae_err.txt
#PBS -l select=1:ncpus=16:ngpus=1:gpu_model=v100:mem=350gb:interconnect=hdr,walltime=72:00:00
#PBS -m abe
#PBS -M eokpala@clemson.edu

module add cuda/11.6.2-gcc/9.5.0 
module add cudnn/8.1.0.77-11.2-gcc/9.5.0 
module add anaconda3/2022.05-gcc/9.5.0

source activate pt_cuda_env

cd /home/eokpala/llm_annotation_bias/experiments/retrain_bertweet/
python run_mlm.py --output_dir=../../../../project/luofeng/socbd/eokpala/new_retrained_aaebert/aae_bertweet --model_type=bert --model_name_or_path=vinai/bertweet-base --do_train --train_file=../../../../project/luofeng/socbd/eokpala/new_aaebert_experiment_data/sampled_black_aligned_train_set_preprocessed.txt --line_by_line --per_device_train_batch_size 64 --max_seq_length 100 --per_gpu_train_batch_size 64 --mlm_probability 0.15 --gradient_accumulation_steps 4 --num_train_epochs 100.0 --save_total_limit 1 --load_best_model_at_end --evaluation_strategy=steps
