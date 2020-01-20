from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory
from semeval2020.util import util

import os.path
import shutil
import itertools
import warnings
import pprint
import pandas as pd
import numpy as np
from numba import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

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
preprocessing_config = "UMAP"
model_config = "BIRCH"

################################################
# Task Specifications ##########################
################################################

tasks = ("task1", "task2")
languages = ("english",)
corpora = ("corpus1", "corpus2")

################################################
#  Code ########################################
################################################

paths = config_factory.get_config(config_paths)

corpora_to_load = list(itertools.product(languages, corpora))
corpora_embeddings = {f"{language}_{corpus}":
                      data_loader_factory.create_data_loader(data_load, base_path=paths["embedding_data_path"],
                                                             language=language, corpus=corpus)
                      for language, corpus in corpora_to_load}

label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}
mapping_dict = {model_name: {}}

# Compute the answers

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
                                                                **config_factory.get_config(preprocessing_config))
        preprocessed_data = preprocessor.fit_transform(x_data)

        model = model_factory.create_model(model_name, **config_factory.get_config(model_config))
        labels = model.fit_predict_labeling(preprocessed_data)

        label_sentences = util.map_labels_to_sentences(labels, word, language, corpora)
        label_sentence_dict = {ul: [sentence for label, sentence in label_sentences if label == ul]
                               for ul in set(labels)}
        mapping_dict[model_name][language][word] = label_sentence_dict


for language in mapping_dict[model_name]:
    for word in mapping_dict[model_name][language]:
        for sense in mapping_dict[model_name][language][word]:
            if sense == -1:
                sense_label = "Noise"
            else:
                sense_label = f"Sense_{sense}"
            task_path = f"{paths['sense_sentence_mappings']}{model_name}/{language}/{word}/"
            os.makedirs(task_path, exist_ok=True)

            sentences = mapping_dict[model_name][language][word][sense]
            df = pd.DataFrame(sentences)
            df.to_csv(f"{task_path}{sense_label}.csv")


print("done")
