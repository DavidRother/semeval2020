from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory, model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory
from semeval2020.model import preprocessing
from semeval2020.configs import constants

from numba import NumbaPerformanceWarning

import tqdm
import evalpy
import warnings
import numpy as np


# PARAMS ###########################

experiment_name = "Hyperparameter_Influence"

###############################

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

################################################
# Pipeline architecture ########################
################################################

data_load = "embedding_loader"
model_names = ["HDBSCANLanguage"]
preprocessing_methods = ["UMAP_AE_Language"]

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

num_trials_per_config = 5

################################################
# Hyperparameter Configs #######################
################################################

model_changes = {"HDBSCANLanguage": {"min_ratio": [0.02, 0.01]}}

preprocessor_changes = {"UMAP_AE_Language": {"n_neighbors": [5, 10]}}

################################################
#  Code ########################################
################################################

base_path = "../../data/main_task_data/"
target_file = f"{base_path}targets/english.txt"
with open(target_file, 'r', encoding='utf-8') as f_in:
    english_target_words = [line.strip().split('\t')[0] for line in f_in]

paths = config_factory.get_config(config_paths)
task_params = config_factory.get_config("TaskParameter")
project_root = paths["evalpy_path"]
evalpy.set_project(project_root, "Semeval 2020 Task 1")

for model_name in model_names:
    for m_param, values in model_changes[model_name].items():
        for value in values:
            for preprocessing_method in preprocessing_methods:
                for p_param, p_values in preprocessor_changes[preprocessing_method].items():
                    for p_value in p_values:
                        for seed in tqdm.tqdm(constants.RANDOM_SEEDS[:num_trials_per_config]):
                            with evalpy.start_run(experiment_name + ' ' + model_name):
                                np.random.seed(seed)
                                evalpy.log_run_entries({"model": model_name, "preprocessing": preprocessing_method})
                                answer_dict = {"task1": {}, "task2": {}}
                                label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}

                                language_preprocessing_config = config_factory.get_config(preprocessing_method)["german"]
                                language_preprocessing_config[p_param] = p_value

                                model_config = config_factory.get_config(model_name)["german"]
                                model_config[m_param] = value
                                model_config_log = {k + '_' + model_name: v for k, v in model_config.items()}

                                evalpy.log_run_entries(language_preprocessing_config)
                                evalpy.log_run_entries(model_config_log)
                                evalpy.log_run_entries({"Auto Encoder": True})

                                all_labels = []

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

                                    language_labels = []

                                    for fig_idx, word in enumerate(target_words):
                                        # try:
                                        file1 = f"{paths['auto_embedding_xlmr_data_path_main']}{language}/corpus1/{word}.npy"
                                        auto_embedded_data1 = np.load(file1)
                                        file2 = f"{paths['auto_embedding_xlmr_data_path_main']}{language}/corpus2/{word}.npy"
                                        auto_embedded_data2 = np.load(file2)
                                        embeddings_label_encoded = []
                                        embeddings_label_encoded.extend([0] * len(auto_embedded_data1))
                                        embeddings_label_encoded.extend([1] * len(auto_embedded_data2))

                                        x_data = np.vstack([auto_embedded_data1, auto_embedded_data2])

                                        preprocessor = preprocessor_factory.create_preprocessor(preprocessing_method,
                                                                                                **language_preprocessing_config)
                                        x_data = preprocessor.fit_transform(x_data)

                                        model = model_factory.create_model(model_name, **model_config)
                                        task_1_answer, task_2_answer, labels = model.predict_with_extra_return(x_data,
                                                                                                               embedding_epochs_labeled=embeddings_label_encoded,
                                                                                                               k=k, n=n)

                                        answer_dict["task1"][language][word] = task_1_answer
                                        answer_dict["task2"][language][word] = task_2_answer
                                        word_noise = len([l for l in labels if l == -1])/len(labels)
                                        language_labels.extend(labels)
                                        # log everything for the step
                                        evalpy.log_run_step({"word": word, "Word noise ratio": word_noise,
                                                            "language": language}, step_forward=True)
                                    # except:
                                        #     evalpy.log_run_step(
                                        #         {"word": word, "Word noise ratio": -1, "language": language},
                                        #         step_forward=True)
                                    if len(language_labels) == 0:
                                        language_noise = -1
                                    else:
                                        language_noise = len([l for l in language_labels if l == -1])/len(language_labels)
                                    evalpy.log_run_entries({f"{language} noise ratio": language_noise})
                                    all_labels.extend(language_labels)

                                if len(all_labels) == 0:
                                    evalpy.log_run_entries({"Noise ratio ALL": -1, "random seed": seed,
                                                            "Embeddings": "XLMR"})
                                    continue
                                truth_data_loader = data_loader_factory.create_data_loader("truth_data_loader",
                                                                                           base_path=paths["truth_test_data_path"])
                                truth_data = truth_data_loader.load()

                                evaluation_suite = evaluation_suite_factory.create_evaluation_suite("semeval_2020_trial_evaluation")
                                t1_results, t2_results = evaluation_suite.evaluate(predictions=answer_dict, truth=truth_data)

                                # log everything in evalpy for the whole run
                                if len(all_labels) == 0:
                                    noise = -1
                                else:
                                    noise = len([l for l in all_labels if l == -1]) / len(all_labels)
                                evalpy.log_run_entries(t1_results)
                                evalpy.log_run_entries(t2_results)
                                evalpy.log_run_entries({"Noise ratio ALL": noise, "random seed": seed,
                                                        "Embeddings": "XLMR"})




