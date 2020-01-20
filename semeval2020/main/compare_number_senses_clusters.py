from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory
from semeval2020.util import util

import numpy as np

import matplotlib.pyplot as plt
import os.path
import pandas as pd
import itertools
from nltk.corpus import wordnet as wn
from scipy.stats import spearmanr

import warnings
from numba import NumbaPerformanceWarning


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Because having multiple scatters in one plot is too much work as an official API
def mscatter(x,y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = plt.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


################################################
# Pipeline architecture ########################
################################################

data_load = "embedding_loader"
model_name = "BIRCH"
preprocessing_method = "UMAP"

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

# 'german', "swedish", "latin", "english"

base_path = "../../data/embedding_data/"

########################################
#  Code ################################
########################################

paths = config_factory.get_config(config_paths)

corpora_to_load = list(itertools.product(languages, corpora))
corpora_embeddings = {f"{language}_{corpus}":
                      data_loader_factory.create_data_loader(data_load, base_path=paths["embedding_data_path"],
                                                             language=language, corpus=corpus)
                      for language, corpus in corpora_to_load}

label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}
marker_encoding = {0: '+', 1: "*"}
colors = ['black', 'red', 'orange', 'blue', 'gray', 'salmon', 'wheat', 'navy']
target_colors = {target: colors[idx] for idx, target in enumerate(corpora)}
predicted_no_senses = []
real_no_senses = []
mapping_dict = {model_name: {}}

for lang_idx, language in enumerate(languages):
    emb_loaders = [emb_loader for emb_loader in corpora_embeddings.values() if emb_loader.language == language]
    target_words = emb_loaders[0].target_words
    mapping_dict[model_name][language] = {}

    for fig_idx, word in enumerate(target_words):
        word_embeddings = []
        embeddings_label_encoded = []
        for emb_loader in emb_loaders:
            embedding = np.asarray(emb_loader[word])
            word_embeddings.append(embedding)
            embeddings_label_encoded.extend([label_encoding[emb_loader.corpus]] * len(embedding))

        x_data = np.vstack(word_embeddings)
        preprocessor = preprocessor_factory.create_preprocessor(preprocessing_method,
                                                                **config_factory.get_config(preprocessing_method))
        preprocessed_data = preprocessor.fit_transform(x_data)

        n_clusters_list = []
        for lab_idx in range(20):
            model = model_factory.create_model(model_name, **config_factory.get_config(model_name))
            labels = model.fit_predict_labeling(preprocessed_data)
            if -1 in labels:
                print(f"Noise points for word {word}")
                n_clusters = len(set(labels)) - 1
            else:
                n_clusters = len(set(labels))
            n_clusters_list.append(n_clusters)
        n_clusters_average = np.mean(n_clusters_list)
        n_senses = len(wn.synsets(word))
        predicted_no_senses.append(n_clusters_average)
        real_no_senses.append(n_senses)

        print(f"Number Average Clusters predicted by {model_name}: {n_clusters_average} Number Senses: {n_senses} for word {word} \n"
              f"Support was {len(x_data)} data points\n")


print(f"Spearman correlation: {spearmanr(real_no_senses, predicted_no_senses)}")
