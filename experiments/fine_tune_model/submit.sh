#Only consider base models
directories="bert_with_default_classes bertweet_with_default_classes hatebert_with_default_classes"

# Consider base models on combined data (ICWSM revision)
directories="bert_with_default_classes_all_data bertweet_with_default_classes_all_data hatebert_with_default_classes_all_data"

for directory in $directories
do
    (cd "$directory" && ./submit_jobs.sh)
    sleep 1s
done
