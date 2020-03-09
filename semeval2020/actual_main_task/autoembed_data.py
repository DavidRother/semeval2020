from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory

import os.path
import shutil
import itertools
import warnings
import pprint
import numpy as np
import tqdm
from numba import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

################################################
# Pipeline architecture ########################
################################################

data_load = "lazy_embedding_loader"
model_name = "DBSCAN_BIRCH"
preprocessing_method = "UMAP"

################################################
# Configs ######################################
################################################

config_paths = "ProjectPaths"

################################################
# Task Specifications ##########################
################################################

tasks = ("task1", "task2")
languages = ("latin", "german", "swedish", "english")
corpora = ["corpus1", "corpus2"]

################################################
#  Code ########################################
################################################

paths = config_factory.get_config(config_paths)

corpora_to_load = list(itertools.product(languages, corpora))
corpora_embeddings = {f"{language}_{corpus}":
                      data_loader_factory.create_data_loader(data_load, base_path=paths["embedding_data_path_main"],
                                                             language=language, corpus=corpus)
                      for language, corpus in corpora_to_load}

answer_dict = {"task1": {}, "task2": {}}
label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}

# Compute the answers
for lang_idx, language in enumerate(languages):
    print(f"Computing Auto Embeddings for {language}")
    emb_loaders = [emb_loader for emb_loader in corpora_embeddings.values() if emb_loader.language == language]
    target_words = emb_loaders[0].target_words
    answer_dict["task1"][language] = {}
    answer_dict["task2"][language] = {}

    for fig_idx, word in tqdm.tqdm(enumerate(target_words)):
        word_embeddings = []
        embeddings_len = []
        for emb_loader in emb_loaders:
            embedding = np.asarray(emb_loader.lazy_load_embeddings(word), dtype=np.float32)
            embedding = embedding[:, 1:]
            word_embeddings.append(embedding)
            embeddings_len.append(len(embedding))

        x_data = np.vstack(word_embeddings)

        preprocessor = preprocessor_factory.create_preprocessor("AutoEncoder",
                                                                **config_factory.get_config("AutoEncoder"))
        preprocessed_data = preprocessor.fit_transform(x_data)

        task_path = f"{paths['auto_embedding_data_path_main']}/{language}/corpus1/"
        os.makedirs(task_path, exist_ok=True)

        np.save(task_path + word, preprocessed_data[:embeddings_len[0]])

        task_path = f"{paths['auto_embedding_data_path_main']}/{language}/corpus2/"
        os.makedirs(task_path, exist_ok=True)

        np.save(task_path + word, preprocessed_data[embeddings_len[0]:])

print("done")
