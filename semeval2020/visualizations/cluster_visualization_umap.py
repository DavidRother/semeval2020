import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from semeval2020.model.embeddingloader import EmbeddingLoader
from sklearn.cluster import KMeans
from kneed import KneeLocator

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

        Sum_of_squared_distances = []
        K = range(1, min(len(umap_embedded_data) + 1, 15))
        km_list = []
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(umap_embedded_data)
            km_list.append(km)
            Sum_of_squared_distances.append(km.inertia_)

        knee = KneeLocator(K, Sum_of_squared_distances, curve="convex", direction="decreasing")
        best_km = km_list[int(knee.knee) - 1]
        plt.figure(fig_idx + len(languages) * lang_idx + len(languages) * len(target_words))
        sns.set(style='white', context='poster')
        _, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(umap_embedded_data[:, 0], umap_embedded_data[:, 1], c=best_km.labels_,
                    cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        color_bar = plt.colorbar(boundaries=np.arange(int(knee.knee) + 1) - 0.5)
        color_bar.set_ticks(np.arange(int(knee.knee)))
        color_bar.set_ticklabels([f"Sense {sense_number}" for sense_number in range(int(knee.knee))])
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


