import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import numpy as np

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

text = "stealing money from the bank vault"
text2 = "stealing money from the [MASK] vault"

# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"
marked_text2 = "[CLS] " + text2 + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)
tokenized_text2 = tokenizer.tokenize(marked_text2)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)

# Display the words with their indeces.
# for tup in zip(tokenized_text, indexed_tokens):
#     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

segments_ids = [1] * len(tokenized_text)
segments_ids2 = [1] * len(tokenized_text2)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

tokens_tensor2 = torch.tensor([indexed_tokens2])
segments_tensors2 = torch.tensor([segments_ids2])

# Load pre-trained model (weights)
model = BertModel.from_pretrained("bert-base-multilingual-cased")

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
    encoded_layers2, _ = model(tokens_tensor2, segments_tensors2)

    embedding = encoded_layers[11]
    embedding2 = encoded_layers2[11]
    dif_vecs = []
    for idx in [1, 2, 3, 4, 5, 6, 8, 9]:

        emb = embedding[:, idx, :]
        emb2 = embedding2[:, idx, :]
        dif_vecs.append(emb-emb2)

    mat = np.zeros((len(dif_vecs), len(dif_vecs)), dtype=float)
    for comb in combinations(enumerate(dif_vecs), 2):
        idx_elem1 = comb[0][0]
        idx_elem2 = comb[1][0]
        elem1 = comb[0][1].numpy()
        elem2 = comb[1][1].numpy()
        mat[idx_elem1, idx_elem2] = cosine_similarity(elem1, elem2)

    print(mat)
