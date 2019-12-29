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


state = (["../../trial_data_public/corpora/german/corpus2", 'Modern Text'],
         ["../../trial_data_public/corpora/german/corpus1", 'Old Text'])

windowSize = 16
corpDir = "../../trial_data_public/corpora/german/corpus1"
complete_embeddings = []
target_file = "../../trial_data_public/targets/german.txt"
# Get targets
with open(target_file, 'r', encoding='utf-8') as f_in:
    targets = [line.strip().split('\t')[0] for line in f_in]
epoch_targets = [target + ep for target, ep in itertools.product(targets, ['Modern Text', 'Old Text'])]
target_embeddings_complete = {target + ep: [] for target, ep in itertools.product(targets, ['Modern Text', 'Old Text'])}

outPath = "/home/david/PycharmProjects/semeval2020/semeval2020/main/embedding_data.csv"
for fig_idx, (corpDir, epoch) in enumerate(state):
    print(f'Working on {epoch}')
    target_file = "../../trial_data_public/targets/german.txt"

    sentences = PathLineSentences(corpDir)

    vocabulary = list(set([word for sentence in sentences for word in sentence if
                           len(sentence) > 1]))  # Skip one-word sentences to avoid zero-vectors
    w2i = {w: i for i, w in enumerate(vocabulary)}

    # Get counts from corpus
    sentences = PathLineSentences(corpDir)
    max_len = 0
    for sentence in sentences:
        max_len = max(len(sentence), max_len)

    target_sentences = {target: ([], [], []) for target in targets}
    target_embeddings = {target: [] for target in targets}
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    MAX_LEN = 128

    for sentence in sentences:
        if len(sentence) < 121:
            for target in targets:
                if target in sentence:
                    tokenized_target = tokenizer.tokenize(target)
                    tokenized_text = tokenizer.tokenize(' '.join(["[CLS]"] + sentence + ["[SEP]"]))
                    target_idx = [(i, i+len(tokenized_target)) for i, tok in enumerate(tokenized_text)
                                  if tokenized_text[i: i+len(tokenized_target)] == tokenized_target]
                    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_text)], maxlen=MAX_LEN,
                                              dtype="long",
                                              truncating="post", padding="post")
                    if target_idx:
                        target_sentences[target][0].append(input_ids[0])
                        seq_mask = [float(i > 0) for i in input_ids[0]]
                        target_sentences[target][1].append(seq_mask)
                        target_sentences[target][2].append(target_idx)

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    for target, stuff in target_sentences.items():
        # Convert inputs to PyTorch tensors
        for token, segment, target_idx in zip(*stuff):
            tokens_tensor = torch.tensor([token])
            segments_tensors = torch.tensor([segment])

            # Predict hidden states features for each layer
            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, token_type_ids=None, attention_mask=segments_tensors)
                embedding = torch.cat(encoded_layers[8:12], 2)
                embeddings = [torch.mean(embedding[:, idx[0]:idx[1], :], dim=1).numpy().flatten() for idx in target_idx]
                target_embeddings[target].extend([emb for emb in embeddings
                                                  if not (np.isnan(np.sum(emb)) or np.sum(emb) == 0)])

    for target, embeddings in target_embeddings.items():
        complete_embeddings.extend(embeddings)
        target_embeddings_complete[target + epoch].extend(embeddings)

data = {str(idx): [target_epoch, *embedding] for idx, (target_epoch, embedding_list) in
        enumerate(target_embeddings_complete.items()) for embedding in embedding_list}
df = pd.DataFrame.from_dict(data)


df.to_csv(outPath)

x_data = np.asarray(complete_embeddings)
umap_instance = umap.UMAP(n_neighbors=10, min_dist=1, metric='cosine')
embedded_data = umap_instance.fit_transform(x_data)

label_encoding = {target: idx for idx, target in enumerate(epoch_targets)}
target_data_encoded = [label_encoding[target] for target, embed in target_embeddings_complete.items() for emb in embed]

colors = ['black', 'red', 'orange', 'blue', 'gray', 'salmon', 'wheat', 'navy']
target_colors = {target: colors[idx] for idx, target in enumerate(epoch_targets)}

sns.set(style='white', context='poster')

fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=target_data_encoded, cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(len(epoch_targets) + 1) - 0.5)
cbar.set_ticks(np.arange(len(epoch_targets)))
cbar.set_ticklabels(epoch_targets)
plt.title(f'Embedded via UMAP for both epochs')
plt.show()


print('done')

