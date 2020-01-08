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


state = (["../../trial_data_public/corpora/german/corpus2", 'Modern Text'],
         ["../../trial_data_public/corpora/german/corpus1", 'Old Text'])

windowSize = 16
corpDir = "../../trial_data_public/corpora/german/corpus1"
complete_embeddings = []
target_file = "../../trial_data_public/targets/german.txt"
# Get targets
with open(target_file, 'r', encoding='utf-8') as f_in:
    targets = [line.strip().split('\t')[0] for line in f_in]

epoch_targets = [target + ep for target, ep in itertools.product(targets, ['Modern_Text', 'Old_Text'])]
target_embeddings_complete = {target + ep: [] for target, ep in itertools.product(targets, ['Modern_Text', 'Old_Text'])}

outPath = "/"
target_file = "../../trial_data_public/targets/german.txt"

sentences = PathLineSentences(corpDir)

vocabulary = list(set([word for sentence in sentences for word in sentence if
                       len(sentence) > 1]))  # Skip one-word sentences to avoid zero-vectors
w2i = {w: i for i, w in enumerate(vocabulary)}

# Get counts from corpus
sentences = PathLineSentences(corpDir)




