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

        Sum_of_squared_distances = []
        K = range(1, min(len(x_data) + 1, 15))
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(x_data)
            Sum_of_squared_distances.append(km.inertia_)

        knee = KneeLocator(K, Sum_of_squared_distances, curve="convex", direction="decreasing")
        plt.figure(fig_idx + len(languages) * lang_idx)
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        try:
            plt.axvline(x=knee.knee)
        except TypeError:
            print(word)
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title(f'Elbow Method For Optimal k for word {word}')

plt.show()


