#!/bin/bash
## declare an array variable
## Dialect priming 
##declare -a arr=("fine_tunning_bertweet_on_all_data.sh")

declare -a arr=("fine_tunning_bertweet_on_all_data.sh")

##for file in *.pbs
for file in "${arr[@]}"
do
    sbatch $file
    sleep 1s
done