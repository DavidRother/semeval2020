from collections import defaultdict

from gensim.models.word2vec import PathLineSentences


windowSize = 16
corpDir = "../../trial_data_public/corpora/german/corpus1"
outPath = "/"

sentences = PathLineSentences(corpDir)
vocabulary = list(set([word for sentence in sentences for word in sentence if
                       len(sentence) > 1]))  # Skip one-word sentences to avoid zero-vectors
w2i = {w: i for i, w in enumerate(vocabulary)}

# Get counts from corpus
sentences = PathLineSentences(corpDir)
max_len = 0
for sentence in sentences:
    max_len = max(len(sentence), max_len)
print(max_len)
