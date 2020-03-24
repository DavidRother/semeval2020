from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory
from semeval2020.model import preprocessing


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
languages = ("english", "latin", "german", "swedish")
corpora = ("corpus1", "corpus2")

################################################
#  Code ########################################
################################################

base_path = "../../data/main_task_data/"
target_file = f"{base_path}targets/english.txt"
with open(target_file, 'r', encoding='utf-8') as f_in:
    english_target_words = [line.strip().split('\t')[0] for line in f_in]

paths = config_factory.get_config(config_paths)
task_params = config_factory.get_config("TaskParameter")

answer_dict = {"task1": {}, "task2": {}}
label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}

# Compute the answers

for lang_idx, language in enumerate(languages):
    target_file = f"{base_path}targets/{language}.txt"
    with open(target_file, 'r', encoding='utf-8') as f_in:
        target_words = [line.strip().split('\t')[0] for line in f_in]
        if language == "english":
            target_words = [preprocessing.remove_pos_tagging_word(word) for word in target_words]

    answer_dict["task1"][language] = {}
    answer_dict["task2"][language] = {}
    k = task_params[language]["k"]
    n = task_params[language]["n"]
    n_ones = 0

    for fig_idx, word in tqdm.tqdm(enumerate(target_words)):
        file1 = f"{paths['auto_embedding_data_path_main']}{language}/corpus1/{word}.npy"
        auto_embedded_data1 = np.load(file1)
        file2 = f"{paths['auto_embedding_data_path_main']}{language}/corpus2/{word}.npy"
        auto_embedded_data2 = np.load(file2)
        embeddings_label_encoded = []
        embeddings_label_encoded.extend([0] * len(auto_embedded_data1))
        embeddings_label_encoded.extend([1] * len(auto_embedded_data2))

        x_data = np.vstack([auto_embedded_data1, auto_embedded_data2])

        preprocessor = preprocessor_factory.create_preprocessor("UMAP", **config_factory.get_config("UMAP_AE_Language")[
                                                                language])
        x_data = preprocessor.fit_transform(x_data)

        model = model_factory.create_model(model_name, **config_factory.get_config(model_name)[language])
        task_1_answer, task_2_answer = model.fit_predict(x_data, embedding_epochs_labeled=embeddings_label_encoded,
                                                         k=k, n=n)

        n_ones = n_ones + task_1_answer
        answer_dict["task1"][language][word] = task_1_answer
        answer_dict["task2"][language][word] = task_2_answer

    print(n_ones)

##############################################
# Evaluate the answer and save it ############
##############################################

# truth_data_loader = data_loader_factory.create_data_loader("truth_data_loader",
#                                                            base_path=paths['truth_trial_data_path'])
# truth_data = truth_data_loader.load()
#
# evaluation_suite = evaluation_suite_factory.create_evaluation_suite("semeval_2020_trial_evaluation")
# eval_results = evaluation_suite.evaluate(predictions=answer_dict, truth=truth_data)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(answer_dict)
# pp.pprint(eval_results)

for task in answer_dict:
    for language in answer_dict[task]:

        task_path = f"{paths['answer_path_main']}{task}/"
        os.makedirs(task_path, exist_ok=True)

        with open(f"{task_path}{language}.txt", 'w', encoding='utf-8') as f_out:
            for word in answer_dict[task][language]:
                answer = int(answer_dict[task][language][word]) if task == "task1" else \
                    float(answer_dict[task][language][word])
                if language == "english":
                    for t_word in english_target_words:
                        if word == t_word[:-3]:
                            word = t_word
                f_out.write('\t'.join((word, str(answer) + '\n')))

shutil.make_archive(paths['out_zip_path_main'], 'zip', paths['in_zip_path_main'])

print("done")
