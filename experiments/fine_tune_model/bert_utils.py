import torch
# These experiments were performed using transformers 4.11.3, python 3.9.x, and latest version of pytorch
TASK = "gpt label" # Can either be 'human label' or 'gpt label.' # After ICWSM review, new LLaMa3 experiements are to be performed so just maintain "gpt" instances in the code even though the data being passed to these instances is LLaMa. This saves the time of having to edit the code to be generalizable to other LLMs. 
CLASS_TYPE = "binary" # Can either be 'multi class' or 'binary'. Binary is used to assess bias using AUC metric. When datasets are combined, class type is binary.
PROMPT_TECHNIQUE = "general_prompt_annotation" # Can either be 'None', 'general_prompt_annotation', 'few_shot_prompt_annotation', or 'cot_prompt_annotation.' For combined datasets, 'general_prompt_annotation_combined_data'
WITH_DIALECT_PRIMING = False
WITH_REGULARIZATION = False
COMBINED_DATA = False

if WITH_DIALECT_PRIMING:
    PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/{PROMPT_TECHNIQUE}/"
    if CLASS_TYPE == "multi class":
        SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models_llama_dialect_priming/gpt_labels/{PROMPT_TECHNIQUE}/"
    else:
        SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models_llama_dialect_priming/gpt_labels_binary/{PROMPT_TECHNIQUE}/"
else:
    if PROMPT_TECHNIQUE == None:
        PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
        if CLASS_TYPE == "multi class":
            SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models/human_labels/"
        else:
            SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models/human_labels_binary/"
    else:
        PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama/{PROMPT_TECHNIQUE}/"
        if CLASS_TYPE == "multi class":
            SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models_llama/llama_labels/{PROMPT_TECHNIQUE}/"
        else:
            if WITH_REGULARIZATION:
                # Save the regularized model in a different path
                SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models/gpt_labels_binary/{PROMPT_TECHNIQUE}_regularization/"
            else:
                SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models_llama/llama_labels_binary/{PROMPT_TECHNIQUE}/"
"""
# For Combined data (COMBINED_DATA = True)
if WITH_DIALECT_PRIMING:
    PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama_dialect_priming/{PROMPT_TECHNIQUE}/"
    SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models_all_data_dialect_priming/llama_labels_binary/{PROMPT_TECHNIQUE}/"
else:
    if PROMPT_TECHNIQUE == None:
        PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/sampled_original_data/"
        SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models_all_data/human_labels_binary/"
    else:
        PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/data_llama/{PROMPT_TECHNIQUE}/"
        SAVE_PATH = f"../../../../../../project/luofeng/socbd/eokpala/llm_annotation_bias/models_all_data/llama_labels_binary/{PROMPT_TECHNIQUE}/"
"""

if CLASS_TYPE == "multi class":
    NUMBER_OF_LABELS_MAP = {
        'abuseval': 3,
        'davidson': 3,
        'founta': 3,
        'golbeck': 2,
        'hateval2019': 2,
        'offenseval2019': 2,
        'waseem-and-hovy': 3,
        'waseem': 4
    }
else:
    NUMBER_OF_LABELS_MAP = {
        'abuseval': 2,
        'davidson': 2,
        'founta': 2,
        'golbeck': 2,
        'hateval2019': 2,
        'offenseval2019': 2,
        'waseem-and-hovy': 2,
        'waseem': 2,
        'all_data': 2 # For combined data
    }

def custom_bert_parameters():
    classifier_criterion = torch.nn.CrossEntropyLoss()
    return classifier_criterion


def hyperparameters():
    batch_size = 32
    learning_rate = 1e-5
    epochs = 5
    return batch_size, learning_rate, epochs


def dialect_hyperparameters():
    alpha = 1.0
    number_labels = 2
    use_mean_representation = False
    use_last_four_representation = True
    experiment_name = f"alpha{str(alpha)}_use_mean{str(use_mean_representation)}_use_last_four{str(use_last_four_representation)}"
    return number_labels, alpha, use_mean_representation, use_last_four_representation, experiment_name