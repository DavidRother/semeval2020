from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory
from semeval2020.util import util
import matplotlib.lines as mlines
import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
from os import listdir
import os.path
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
model_name = "HDBSCAN"
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

########################################
#  Code ################################
########################################

paths = config_factory.get_config(config_paths)

label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}
marker_encoding = {0: '*', 1: "^"}
colors = ['black', 'red', 'orange', 'blue', 'gray', 'salmon', 'wheat', 'navy']
target_colors = {target: colors[idx] for idx, target in enumerate(corpora)}
np.random.seed(42)

for lang_idx, language in enumerate(languages):
    target_dir = f"{paths['embedding_data_path_main']}{language}/corpus1/"
    filenames = listdir(target_dir)
    csv_filenames = [filename for filename in filenames if filename.endswith(".csv")]
    target_words = [os.path.splitext(filename)[0] for filename in csv_filenames]

    for fig_idx, word in enumerate(target_words):
        if word != "chairman":
            continue
        file1 = f"{paths['auto_embedding_data_path_main']}{language}/corpus1/{word}.npy"
        auto_embedded_data1 = np.load(file1)
        file2 = f"{paths['auto_embedding_data_path_main']}{language}/corpus2/{word}.npy"
        auto_embedded_data2 = np.load(file2)
        embeddings_label_encoded = []
        embeddings_label_encoded.extend([0] * len(auto_embedded_data1))
        embeddings_label_encoded.extend([1] * len(auto_embedded_data2))
        print(word)

        x_data = np.vstack([auto_embedded_data1, auto_embedded_data2])
        preprocessor = preprocessor_factory.create_preprocessor("UMAP", **config_factory.get_config("UMAP_AE_Language")
                                                                [language])
        preprocessed_data = preprocessor.fit_transform(x_data)

        model = model_factory.create_model(model_name, **config_factory.get_config(model_name)[language])
        labels = model.fit_predict_labeling(preprocessed_data)
        # noise = False
        if -1 in labels:
            # noise = True
            labels = labels + 1
            print(f"Noise points in the first sense for word {word}")

        cluster_n = len(set(labels))
        plt.figure(fig_idx + len(languages) * lang_idx + len(languages) * len(target_words))
        sns.set(style='white', context='poster')
        _, ax = plt.subplots(1, figsize=(14, 10))
        markers = [marker_encoding[lab] for lab in embeddings_label_encoded]

        if cluster_n == 2:
            print("yes")
        marks = ['*' if lab % 2 else '^' for lab in labels]
        mscatter(preprocessed_data[:, 0], preprocessed_data[:, 1], ax=ax, m=marks, c=labels, cmap='Spectral',
                 alpha=1.0)
        sen1_handle = mlines.Line2D([0], [0], color='w', marker='*', markersize=20, label='Sense 1',
                                    markerfacecolor='#a61e3b')
        sen2_handle = mlines.Line2D([0], [0], color='w', marker='^', markersize=20, label='Sense 2',
                                    markerfacecolor='slateblue')

        ax.legend(handles=[sen1_handle, sen2_handle])
        plt.setp(ax, xticks=[], yticks=[])
        if cluster_n > 200:
            print(f"wtf {word}")
        plt.title(f'{model_name} clustered word "{word}"')

        plt.figure(fig_idx + len(languages) * lang_idx)
        sns.set(style='white', context='poster')
        _, ax = plt.subplots(1, figsize=(14, 10))
        mscatter(preprocessed_data[:, 0], preprocessed_data[:, 1], ax=ax, m=markers, c=embeddings_label_encoded, s=80,
                 cmap='Spectral', alpha=1.0)
        corp1_handle = mlines.Line2D([0], [0], color='w', marker='*', markersize=20, label='Corpus 1',
                                     markerfacecolor='#a61e3b')
        corp2_handle = mlines.Line2D([0], [0], color='w', marker='^', markersize=20, label='Corpus 2',
                                     markerfacecolor='slateblue')

        ax.legend(handles=[corp1_handle, corp2_handle])
        plt.setp(ax, xticks=[], yticks=[])
        plt.title(f'{preprocessing_method} embedded word "{word}"')

plt.show()


