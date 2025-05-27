#!/bin/bash
## declare an array variable
## Dialect priming
##declare -a arr=("fine_tunning_bertweet_on_davidson.sh" "fine_tunning_bertweet_on_founta.sh" "fine_tunning_bertweet_on_golbeck.sh" "fine_tunning_bertweet_on_hateval.sh" "fine_tunning_bertweet_on_offenseval.sh")

declare -a arr=("fine_tunning_bertweet_on_abuseval.sh" "fine_tunning_bertweet_on_davidson.sh" "fine_tunning_bertweet_on_founta.sh" "fine_tunning_bertweet_on_golbeck.sh" "fine_tunning_bertweet_on_hateval.sh" "fine_tunning_bertweet_on_offenseval.sh" "fine_tunning_bertweet_on_waseem_and_hovy.sh" "fine_tunning_bertweet_on_waseem.sh")

##for file in *.pbs
for file in "${arr[@]}"
do
    sbatch $file
    sleep 1s
done