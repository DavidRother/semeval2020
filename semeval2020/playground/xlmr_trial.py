import torch

from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import numpy as np

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.eval()

# Load pre-trained model tokenizer (vocabulary)

text = "stealing money from the bank vault"
text2 = "stealing money from the [MASK] vault"


# Split the sentence into tokens.
tokenized_text = xlmr.encode(text)
tokenized_text2 = xlmr.encode(text2)

last_layer_features = xlmr.extract_features(tokenized_text).detach().numpy()[0, 5, :].reshape(1, -1)
last_layer_features2 = xlmr.extract_features(tokenized_text2).detach().numpy()[0, 5, :].reshape(1, -1)

print(cosine_similarity(last_layer_features, last_layer_features2))
