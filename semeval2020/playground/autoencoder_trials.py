from semeval2020.model.bertwrapper import BertWrapper
from semeval2020.data_loader.senseval_sentence_loader import SensEvalLoader
from semeval2020.model import preprocessing

import torch
import os.path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import tqdm
from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory

import os.path
import shutil
import itertools
import warnings
import pprint
import numpy as np
from numba import NumbaPerformanceWarning
from itertools import compress
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, normalized_mutual_info_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

################################################
# Pipeline architecture ########################
################################################

data_load = "embedding_loader"
model_name = "BIRCH"
preprocessing_method = "UMAP"
preprocessing_method2 = "AutoEncoder"

################################################
# Configs ######################################
################################################

config_paths = "ProjectPaths"

################################################
# Task Specifications ##########################
################################################

tasks = ("task1", "task2")
languages = ("english",)
corpora = ("corpus1", "corpus2")

########################################
#  Config Parameter ####################
########################################

language = 'english'
corpus = "corpus1"

base_path = "../../trial_data_public/"
model_string = "bert-base-cased"

output_path = "../../data/embedding_data/"
sentence_out_path = "../../data/sentence_data/"

max_length_sentence = 200
padding_length = 256

num_words = 10

########################################
#  Code ################################
########################################

base_out_path = f"{output_path}{language}/{corpus}/"
sentence_output_final_path = f"{sentence_out_path}{language}/{corpus}/"
os.makedirs(base_out_path, exist_ok=True)

data_loader = SensEvalLoader(base_path, language=language)
bert_model = BertWrapper(model_string=model_string)

target_words, word_to_sentences_senses = data_loader.load(num_words=num_words)

target_word_array = [word for word in target_words for s in word_to_sentences_senses[word]['sentences']]
sentences = [s for word in target_words for s in word_to_sentences_senses[word]['sentences']]

sentences = list(preprocessing.sanitized_sentences(sentences, max_len=max_length_sentence))
tokenized_target_sentences = bert_model.tokenize_sentences_direct_mapping(sentences, target_word_array, target_words)
target_embeddings_list = []
target_embedding_dict = {w: [] for w in target_words}

bert_model.enter_eval_mode()

for tokenized_sentence, target_word_idx_dict in tqdm.tqdm(tokenized_target_sentences):
    input_ids = bert_model.get_tokenized_input_ids(tokenized_sentence, padding_length=padding_length)
    attention_mask = bert_model.get_attention_mask(input_ids)

    input_id_tensor = torch.tensor([input_ids])
    attention_mask_tensor = torch.tensor([attention_mask])
    target_embeddings = bert_model.compute_embeddings(input_id_tensor, attention_mask_tensor, target_word_idx_dict)
    for target, embeddings in target_embeddings.items():
        if not len(embeddings):
            print(target)
            print(tokenized_sentence)
        target_embeddings_list.append(embeddings[0])
        target_embedding_dict[target].append(embeddings[0])

# now compute clustering
for fig_idx, word in enumerate(target_words):
    embs = target_embedding_dict[word]
    x_data = np.asarray(embs)

    ae_preprocessor = preprocessor_factory.create_preprocessor(preprocessing_method2, input_size=len(x_data[0]),
                                                               **config_factory.get_config(preprocessing_method2))

    pp_data = ae_preprocessor.fit_transform(x_data)

    pp_data = np.asarray(pp_data)

    preprocessor = preprocessor_factory.create_preprocessor(preprocessing_method,
                                                            **config_factory.get_config(preprocessing_method))
    preprocessed_data = preprocessor.fit_transform(pp_data)

    model = model_factory.create_model(model_name, **config_factory.get_config(model_name))
    labels = model.fit_predict_labeling(preprocessed_data)

    noise = False
    if -1 in labels:
        noise = True
        print(f"Noise points in the first sense for word {word}")

    sense_set = set([sense for senses in word_to_sentences_senses[word]['senses'] for sense in senses])
    word_senses = word_to_sentences_senses[word]['senses']
    sense_to_num = {s: idx for idx, s in enumerate(sense_set)}

    label_to_sense = {}
    for lab in set(labels):
        index_filter = [lab == l for l in labels]
        filtered_senses = [s[0] for s in list(compress(word_senses, index_filter))]
        label_to_sense[lab] = max(set(filtered_senses), key=filtered_senses.count)

    predicted_labels = [sense_to_num[label_to_sense[label]] for label in labels]
    true_labels = [sense_to_num[senses_[0]] for senses_ in word_senses]

    # Plots ##################################################################################

    cluster_n = len(set(labels))
    num_senses = len(sense_set)
    plt.figure(fig_idx)
    sns.set(style='white', context='poster')
    _, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(preprocessed_data[:, 0], preprocessed_data[:, 1], c=labels, cmap='Spectral', alpha=1.0)
    if cluster_n > 1:
        color_bar = plt.colorbar(boundaries=np.arange(cluster_n + 1) - 0.5)
        color_bar.set_ticks(np.arange(cluster_n))
        color_bar.set_ticklabels([f"Sense {sense_number}" for sense_number in range(cluster_n)])
    plt.title(f"{preprocessing_method2} embedded {model_name} clustered {word} Estimated Senses")

    plt.figure(fig_idx + len(target_words) + 1)
    sns.set(style='white', context='poster')
    _, ax = plt.subplots(1, figsize=(14, 10))
    sense_labels = [sense_to_num[sense[0]] for sense in word_senses]
    plt.scatter(preprocessed_data[:, 0], preprocessed_data[:, 1], c=sense_labels, cmap='Spectral', alpha=1.0)
    if num_senses > 1:
        color_bar = plt.colorbar(boundaries=np.arange(num_senses + 1) - 0.5)
        color_bar.set_ticks(np.arange(num_senses))
        color_bar.set_ticklabels([f"Sense {sense_number}" for sense_number in range(num_senses)])
    plt.title(f"{preprocessing_method2} embedded {model_name} clustered {word} True Senses")

    print(f"Evaluation for word {word}")
    print(f'Number Senses: {len(sense_set)} | Number Clusters: {cluster_n} | '
          f'Distinct Sense Clusters {len(set(predicted_labels))}')
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
    print(f"Normalized Mutual Information Score {normalized_mutual_info_score(true_labels, labels)}")
    # if len(set(predicted_labels)) > 1:
    #     print(f"Roc Auc Score: {roc_auc_score(true_labels, predicted_labels)}")
    #     print(f"F1 Score: {f1_score(true_labels, predicted_labels)}")
    print(f"Confusion matrix: ")
    print(confusion_matrix(true_labels, predicted_labels))
    print('')

plt.show()



