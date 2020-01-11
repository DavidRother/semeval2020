import umap
import numpy as np
import os.path

import itertools
import shutil
from semeval2020.model.embeddingloader import EmbeddingLoader
from scipy.spatial import distance
from semeval2020.model import clustering
from semeval2020.factory_hub import data_loader_factory, evaluation_suite_factory

########################################
#  Config Parameter ####################
########################################

languages = ['german', 'latin', 'swedish', 'english']
corpora = ["corpus1", "corpus2"]

base_path = "../../semeval2020/embedding_data/"
truth_path = "../../trial_data_public/truth/"
answer_path = "../../my_results/answer/answer/"
out_zip_path = "../../my_results/answer"
in_zip_path = "../../my_results/answer/"

########################################
#  Code ################################
########################################

corpora_to_load = list(itertools.product(languages, corpora))
corpora_embeddings = {f"{language}_{corpus}": EmbeddingLoader(base_path, language=language, corpus=corpus)
                      for language, corpus in corpora_to_load}
label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}

answer_dict = {}
answer_dict["task1"] = {}
answer_dict["task2"] = {}
for lang_idx, language in enumerate(languages):
    emb_loaders = [emb_loader for emb_loader in corpora_embeddings.values() if emb_loader.language == language]
    target_words = emb_loaders[0].target_words
    answer_dict["task1"][language] = {}
    answer_dict["task2"][language] = {}

    for fig_idx, word in enumerate(target_words):
        word_embeddings = []
        embeddings_label_encoded = []
        for emb_loader in emb_loaders:
            embedding = np.asarray(emb_loader[word])
            word_embeddings.append(embedding)
            embeddings_label_encoded.extend([label_encoding[emb_loader.corpus]] * len(embedding))

        x_data = np.vstack(word_embeddings)
        umap_embedded_data = None
        for idx in range(10, 2, -1):
            try:
                umap_instance = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='cosine')
                umap_embedded_data = umap_instance.fit_transform(x_data)
                break
            except TypeError:
                continue

        gmmm_clustering = clustering.find_gmm_clustering(umap_embedded_data)
        sense_frequencies = clustering.compute_cluster_sense_frequency(gmmm_clustering, embeddings_label_encoded,
                                                                       list(range(len(corpora))))

        task_1_answer = int(any([True for sd in sense_frequencies if 0 in sense_frequencies[sd]]))
        task_2_answer = distance.jensenshannon(sense_frequencies[0], sense_frequencies[1], 2.0)
        answer_dict["task1"][language][word] = task_1_answer
        answer_dict["task2"][language][word] = task_2_answer

truth_data_loader = data_loader_factory.create_data_loader("truth_data_loader", base_path=truth_path)
truth_data = truth_data_loader.load()

evaluation_suite = evaluation_suite_factory.create_evaluation_suite("semeval_2020_trial_evaluation")
eval_results = evaluation_suite.evaluate(predictions=answer_dict, truth=truth_data)

for task in answer_dict:
    for language in answer_dict[task]:

        task_path = f"{answer_path}{task}/"
        os.makedirs(task_path, exist_ok=True)

        with open(f"{task_path}{language}.txt", 'w', encoding='utf-8') as f_out:
            for word in answer_dict[task][language]:
                answer = int(answer_dict[task][language][word]) if task == "task1" else \
                    float(answer_dict[task][language][word])
                f_out.write('\t'.join((word, str(answer) + '\n')))

shutil.make_archive(out_zip_path, 'zip', in_zip_path)

print("done")
