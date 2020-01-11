import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from semeval2020.model.embeddingloader import EmbeddingLoader
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.spatial import distance
from semeval2020.model import clustering

########################################
#  Config Parameter ####################
########################################

languages = ['swedish']
corpora = ["corpus1", "corpus2"]

base_path = "../../semeval2020/embedding_data/"

########################################
#  Code ################################
########################################

corpora_to_load = list(itertools.product(languages, corpora))
corpora_embeddings = {f"{language}_{corpus}": EmbeddingLoader(base_path, language=language, corpus=corpus)
                      for language, corpus in corpora_to_load}
label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}
colors = ['black', 'red', 'orange', 'blue', 'gray', 'salmon', 'wheat', 'navy']
target_colors = {target: colors[idx] for idx, target in enumerate(corpora)}

answer_dict = {}
for lang_idx, language in enumerate(languages):
    emb_loaders = [emb_loader for emb_loader in corpora_embeddings.values() if emb_loader.language == language]
    target_words = emb_loaders[0].target_words
    answer_dict[language] = {}

    for fig_idx, word in enumerate(target_words):
        word_embeddings = []
        embeddings_label_encoded = []
        for emb_loader in emb_loaders:
            embedding = np.asarray(emb_loader[word])
            word_embeddings.append(embedding)
            embeddings_label_encoded.extend([label_encoding[emb_loader.corpus]] * len(embedding))

        x_data = np.vstack(word_embeddings)
        umap_instance = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='cosine')
        umap_embedded_data = umap_instance.fit_transform(x_data)

        km_clustering = clustering.find_best_clustering(umap_embedded_data)
        sense_frequencies = clustering.compute_cluster_sense_frequency(km_clustering, embeddings_label_encoded,
                                                                       list(range(len(corpora))))

        task_1_answer = any([True for sd in sense_frequencies if 0 in sense_frequencies[sd]])
        task_2_answer = distance.jensenshannon(sense_frequencies[0], sense_frequencies[1], 2.0)
        answer_dict[language][word] = {"task_1_answer": task_1_answer, "task_2_answer": task_2_answer}


print('done')


