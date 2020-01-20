from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory
from semeval2020.util import util

import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from semeval2020.model.embeddingloader import EmbeddingLoader
from sklearn.cluster import Birch

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


for lang_idx, language in enumerate(languages):
    emb_loaders = [emb_loader for emb_loader in corpora_embeddings.values() if emb_loader.language == language]
    target_words = emb_loaders[0].target_words

    for fig_idx, word in enumerate(target_words):
        word_embeddings = []
        embeddings_label_encoded = []
        for emb_loader in emb_loaders:
            embedding = np.asarray(emb_loader[word])
            word_embeddings.append(embedding)
            embeddings_label_encoded.extend([label_encoding[emb_loader.corpus]] * len(embedding))
        print(word)

        x_data = np.vstack(word_embeddings)
        preprocessor = preprocessor_factory.create_preprocessor(preprocessing_method,
                                                                **config_factory.get_config(preprocessing_method))
        preprocessed_data = preprocessor.fit_transform(x_data)

        model = model_factory.create_model(model_name, **config_factory.get_config(model_name))
        labels = model.fit_predict_labeling(preprocessed_data)
        noise = False
        if -1 in labels:
            noise = True
            print(f"Noise points in the first sense for word {word}")

        cluster_n = len(set(labels))
        plt.figure(fig_idx + len(languages) * lang_idx + len(languages) * len(target_words))
        sns.set(style='white', context='poster')
        _, ax = plt.subplots(1, figsize=(14, 10))
        markers = [marker_encoding[lab] for lab in embeddings_label_encoded]
        mscatter(preprocessed_data[:, 0], preprocessed_data[:, 1], ax=ax, m=markers, c=labels, cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        if cluster_n > 1:
            color_bar = plt.colorbar(boundaries=np.arange(cluster_n + 1) - 0.5)
            color_bar.set_ticks(np.arange(cluster_n))
            color_bar.set_ticklabels([f"Sense {sense_number}" for sense_number in range(cluster_n)])
        plt.title(f"{preprocessing_method} embedded {model_name} clustered {word}")

        plt.figure(fig_idx + len(languages) * lang_idx)
        _, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(preprocessed_data[:, 0], preprocessed_data[:, 1], c=embeddings_label_encoded,
                    cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        color_bar = plt.colorbar(boundaries=np.arange(len(corpora) + 1) - 0.5)
        color_bar.set_ticks(np.arange(len(corpora)))
        color_bar.set_ticklabels(corpora)
        plt.title(f"{preprocessing_method} embedded {word}")

plt.show()


