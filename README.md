# Large Language Model Annotation Bias in Hate Speech Detection
Ebuka Okpala and Long Cheng

Accepted at the International Conference on Web and Social Media (ICWSM'25), June 23 - 26, 2025, Copenaghen, Denmark

# Abstract
Large language models (LLMs) are fast becoming ubiquitous and have shown impressive performance in various natural language processing (NLP) tasks. Annotating data for downstream applications is a resource-intensive task in NLP. Recently, the use of LLMs as a cost-effective data annotator for annotating data used to train other models or as an assistive tool has been explored. Yet, little is known regarding the societal implications of using LLMs for data annotation. In this work, focusing on hate speech detection, we investigate how using LLMs such as GPT-4 and Llama-3 for hate speech detection can lead to different performances for different text dialects and racial bias in online hate detection classifiers. We used
LLMs to predict hate speech in seven hate speech datasets and trained classifiers on the LLM annotations of each dataset. Using tweets written in African-American English (AAE) and Standard American English (SAE), we show that classifiers trained on
LLM annotations assign tweets written in AAE to negative classes (e.g., hate, offensive, abuse, racism, etc.) at a higher rate than tweets written in SAE and that the classifiers have a higher false positive rate towards AAE tweets. We explore the effect of incorporating dialect priming in the prompting techniques used in prediction, showing that introducing dialect increases the rate at which AAE tweets are assigned to negative classes. 

This repository contains code used for the analysis in the paper mentioned above. 
