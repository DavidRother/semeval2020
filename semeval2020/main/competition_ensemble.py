from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory

import os.path
import shutil
import itertools
import warnings
import pprint
import numpy as np
from itertools import groupby as g
from numba import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

################################################
# Pipeline architecture ########################
################################################

data_load = "embedding_loader"
models = ["DBSCAN", "BIRCH", "NaiveGMM"]
preprocessing_queues = [[("UMAP", "UMAP")], [("AutoEncoder", "AutoEncoder"), ("UMAP", "UMAP_AE")],
                        [("AutoEncoder", "AutoEncoder")], [("TSNE", "TSNE")],
                        [("AutoEncoder", "AutoEncoder"), ("TSNE", "TSNE_AE")]]

################################################
# Configs ######################################
################################################

config_paths = "ProjectPaths"

################################################
# Task Specifications ##########################
################################################

tasks = ("task1", "task2")
languages = ("english", "german", "swedish", "latin")
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

answer_dict = {"task1": {}, "task2": {}}
label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}

# Compute the answers

for lang_idx, language in enumerate(languages):
    emb_loaders = [emb_loader for emb_loader in corpora_embeddings.values() if emb_loader.language == language]
    target_file = f"{paths['trial_data_path']}targets/{language}.txt"
    with open(target_file, 'r', encoding='utf-8') as f_in:
        target_words = [line.strip().split('\t')[0] for line in f_in]
    answer_dict["task1"][language] = {}
    answer_dict["task2"][language] = {}

    for fig_idx, word in enumerate(target_words):
        word_embeddings = []
        embeddings_label_encoded = []
        for emb_loader in emb_loaders:
            embedding = np.asarray(emb_loader[word])
            word_embeddings.append(embedding)
            embeddings_label_encoded.extend([label_encoding[emb_loader.corpus]] * len(embedding))

        data = np.vstack(word_embeddings)

        task_1_stack = []
        task_2_stack = []

        for preprocessing_queue in preprocessing_queues:

            for preprocessing_method, pp_config in preprocessing_queue:
                preprocessor = preprocessor_factory.create_preprocessor(preprocessing_method,
                                                                        **config_factory.get_config(pp_config))
                data = preprocessor.fit_transform(data)

            for model_name in models:
                model = model_factory.create_model(model_name, **config_factory.get_config(model_name))
                task_1_answer, task_2_answer = model.fit_predict(data,
                                                                 embedding_epochs_labeled=embeddings_label_encoded)

                task_1_stack.append(task_1_answer)
                task_2_stack.append(task_2_answer)

        answer_dict["task1"][language][word] = max(g(sorted(task_1_stack)),
                                                   key=lambda x, v: (len(list(v)), -task_1_stack.index(x)))[0]
        answer_dict["task2"][language][word] = np.mean(task_2_stack)

##############################################
# Evaluate the answer and save it ############
##############################################

truth_data_loader = data_loader_factory.create_data_loader("truth_data_loader",
                                                           base_path=paths['truth_trial_data_path'])
truth_data = truth_data_loader.load()

evaluation_suite = evaluation_suite_factory.create_evaluation_suite("semeval_2020_trial_evaluation")
eval_results = evaluation_suite.evaluate(predictions=answer_dict, truth=truth_data)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(answer_dict)
pp.pprint(eval_results)

for task in answer_dict:
    for language in answer_dict[task]:

        task_path = f"{paths['answer_path']}{task}/"
        os.makedirs(task_path, exist_ok=True)

        with open(f"{task_path}{language}.txt", 'w', encoding='utf-8') as f_out:
            for word in answer_dict[task][language]:
                answer = int(answer_dict[task][language][word]) if task == "task1" else \
                    float(answer_dict[task][language][word])
                f_out.write('\t'.join((word, str(answer) + '\n')))

shutil.make_archive(paths['out_zip_path'], 'zip', paths['in_zip_path'])

print("done")
