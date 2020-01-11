import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from semeval2020.model.embeddingloader import EmbeddingLoader
from sklearn.cluster import KMeans
from kneed import KneeLocator
from semeval2020.model import clustering

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

        km_clustering = clustering.find_best_clustering(umap_embedded_data)
        sense_frequencies = clustering.compute_cluster_sense_frequency(km_clustering, embeddings_label_encoded,
                                                                       list(range(len(corpora))))

        n_senses = len(list(sense_frequencies.values())[0])
        epoch_1_sense_frequency_distribution = sense_frequencies[0]
        epoch_2_sense_frequency_distribution = sense_frequencies[1]

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_senses)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, epoch_1_sense_frequency_distribution, bar_width, alpha=opacity, color='b',
                         label='Corpus 1')

        rects2 = plt.bar(index + bar_width, epoch_2_sense_frequency_distribution, bar_width, alpha=opacity, color='g',
                         label='Corpus 2')

        plt.xlabel('Sense')
        plt.ylabel('Relative Occurrence')
        plt.title(f'Sense Frequency Distribution for word {word}')
        plt.xticks(index + bar_width/2, index)
        plt.legend()

        plt.tight_layout()


plt.show()


