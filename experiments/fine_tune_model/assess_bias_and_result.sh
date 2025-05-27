directory1="assess_bias"
(cd "$directory1" && sbatch assess_bias_in_fine_tuned_models.sh)
sbatch results.sh
