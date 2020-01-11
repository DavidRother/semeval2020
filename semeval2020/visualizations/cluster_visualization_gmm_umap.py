import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from semeval2020.model.embeddingloader import EmbeddingLoader
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture

########################################
#  Config Parameter ####################
########################################

languages = ['german']
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

        x_data = np.vstack(word_embeddings)
        umap_instance = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='cosine')
        umap_embedded_data = umap_instance.fit_transform(x_data)

        sum_of_squared_distances = []
        K = range(1, min(len(umap_embedded_data) + 1, 10))
        gmm_list = []
        for k in K:
            gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3)
            gmm_labels = gmm.fit_predict(umap_embedded_data)
            gmm_list.append(gmm_labels)
            sum_of_squared_distances.append(gmm.bic(umap_embedded_data))

        knee = KneeLocator(K, sum_of_squared_distances, curve="convex", direction="decreasing")
        if not knee.knee:
            cluster_n = 1
        else:
            cluster_n = int(knee.knee)
        best_gmm = gmm_list[cluster_n - 1]
        num_values = len(set(best_gmm))
        plt.figure(fig_idx + len(languages) * lang_idx + len(languages) * len(target_words))
        sns.set(style='white', context='poster')
        _, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(umap_embedded_data[:, 0], umap_embedded_data[:, 1], c=best_gmm,
                    cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        if cluster_n > 1:
            color_bar = plt.colorbar(boundaries=np.arange(cluster_n + 1) - 0.5)
            color_bar.set_ticks(np.arange(cluster_n))
            color_bar.set_ticklabels([f"Sense {sense_number}" for sense_number in range(cluster_n)])
        plt.title(f"UMAP embedded {word}")

        plt.figure(fig_idx + len(languages) * lang_idx)
        _, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(umap_embedded_data[:, 0], umap_embedded_data[:, 1], c=embeddings_label_encoded,
                    cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        color_bar = plt.colorbar(boundaries=np.arange(len(corpora) + 1) - 0.5)
        color_bar.set_ticks(np.arange(len(corpora)))
        color_bar.set_ticklabels(corpora)
        plt.title(f"UMAP embedded {word}")

plt.show()


