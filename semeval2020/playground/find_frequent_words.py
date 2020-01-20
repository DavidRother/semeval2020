from collections import defaultdict, Counter
from gensim.models.word2vec import PathLineSentences
from semeval2020.factory_hub import config_factory

config_paths = config_factory.get_config("ProjectPaths")
language = "english"
corpus = "corpus2"
corpus1 = "corpus1"

sentences = list(PathLineSentences(f"{config_paths['corpus_path']}{language}/{corpus}"))
sentences1 = list(PathLineSentences(f"{config_paths['corpus_path']}{language}/{corpus1}"))

sents_flat = ' '.join([w for sl in sentences for w in sl])
sents_flat1 = ' '.join([w for sl in sentences1 for w in sl])

counts = dict(Counter(sents_flat.split()))
counts1 = defaultdict(int, dict(Counter(sents_flat1.split())))

num_occurrence = 20
a = [k for k in counts.keys() if counts[k] >= num_occurrence and counts1[k] >= num_occurrence]
print(a)
# print(Counter(sents_flat.split()))

print('done')


