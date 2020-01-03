from collections import defaultdict
import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
from semeval2020.model.embeddingloader import EmbeddingLoader

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

for corp_idx, (corpus_specifier, emb_loader) in enumerate(corpora_embeddings.items()):
    for fig_idx, (word, embeddings) in enumerate(emb_loader.embeddings.items()):
        x_data = np.asarray(embeddings)
        umap_instance = umap.UMAP(n_neighbors=10, min_dist=0.05, metric='cosine')
        umap_embedded_data = umap_instance.fit_transform(x_data)
        plt.figure(fig_idx + len(emb_loader.embeddings) * corp_idx)
        plt.scatter(umap_embedded_data[:, 0], umap_embedded_data[:, 1])
        plt.title(f"UMAP embedded {word} from {corpus_specifier}")

plt.show()

# x_data = np.asarray(complete_embeddings)
# umap_instance = umap.UMAP(n_neighbors=10, min_dist=1, metric='cosine')
# embedded_data = umap_instance.fit_transform(x_data)
#
# label_encoding = {target: idx for idx, target in enumerate(epoch_targets)}
# target_data_encoded = [label_encoding[target] for target, embed in target_embeddings_complete.items() for emb in embed]
#
# colors = ['black', 'red', 'orange', 'blue', 'gray', 'salmon', 'wheat', 'navy']
# target_colors = {target: colors[idx] for idx, target in enumerate(epoch_targets)}
#
# sns.set(style='white', context='poster')
#
# fig, ax = plt.subplots(1, figsize=(14, 10))
# plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=target_data_encoded, cmap='Spectral', alpha=1.0)
# plt.setp(ax, xticks=[], yticks=[])
# cbar = plt.colorbar(boundaries=np.arange(len(epoch_targets) + 1) - 0.5)
# cbar.set_ticks(np.arange(len(epoch_targets)))
# cbar.set_ticklabels(epoch_targets)
# plt.title(f'Embedded via UMAP for both epochs')
# plt.show()


print('done')

