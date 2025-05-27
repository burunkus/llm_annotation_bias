directory1="assess_bias_binary"
(cd "$directory1" && sbatch assess_bias_in_fine_tuned_models_using_auc.sh)
(cd "$directory1" && sbatch assess_bias_in_fine_tuned_models.sh)
sbatch results.sh
