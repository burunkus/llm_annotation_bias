import torch

PATH = "/project/luofeng/socbd/eokpala/new_aaebert_experiment_data/"
SAVE_PATH = "/project/luofeng/socbd/eokpala/bert_experiments/new_experiments_with_baseline_comparison/fine_tune_aaebert/"
NUMBER_OF_LABELS = 2 

def custom_bert_parameters():
    classifier_criterion = torch.nn.CrossEntropyLoss()
    return classifier_criterion

      
def hyperparameters():
    batch_size = 32
    learning_rate = 1e-5
    epochs = 20
    return batch_size, learning_rate, epochs

