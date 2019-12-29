from collections import defaultdict
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from gensim.models.word2vec import PathLineSentences
from keras.preprocessing.sequence import pad_sequences
import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd

target_file = "../../trial_data_public/targets/german.txt"
# Get targets
with open(target_file, 'r', encoding='utf-8') as f_in:
    targets = [line.strip().split('\t')[0] for line in f_in]

epoch_targets = [target + ep for target, ep in itertools.product(targets, ['Modern_Text', 'Old_Text'])]
target_embeddings_complete = {target + ep: [] for target, ep in itertools.product(targets, ['Modern_Text', 'Old_Text'])}

data_source = "/home/david/PycharmProjects/semeval2020/semeval2020/main/embedding_data.csv"

df = pd.read_csv(data_source, header=[0], nrows=1)
target_epoch_list = df[:][0]

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

