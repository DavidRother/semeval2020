from semeval2020.language_models.xlmrwrapper import XLMRWrapper
from semeval2020.data_loader.sentence_loader import SentenceLoader
from semeval2020.model import preprocessing

import torch
import os.path
import pandas as pd
import tqdm
import numpy as np


########################################
#  Config Parameter ####################
########################################

language = 'latin'
corpus = "corpus1"

base_path = "../../data/main_task_data/"
model_string = 'xlmr.large'  # "dbmdz/bert-base-german-cased"

output_path = "../../data/xlmr_embedding_data_semeval2020/"
sentence_out_path = "../../data/xlmr_sentence_data_semeval2020/"

max_length_sentence = 100
padding_length = 100

auto = False

########################################
#  Code ################################
########################################

base_out_path = f"{output_path}{language}/{corpus}/"
sentence_output_final_path = f"{sentence_out_path}{language}/{corpus}/"
os.makedirs(base_out_path, exist_ok=True)

data_loader = SentenceLoader(base_path, language=language, corpus=corpus)
xlmr_model = XLMRWrapper(model_string=model_string)

target_words, sentences = data_loader.load()
sentences = preprocessing.sanitized_sentences(sentences, max_len=max_length_sentence)
sentences = (sentence[1:] for sentence in sentences)
sentences = preprocessing.filter_for_words(sentences, target_words)
sentences = preprocessing.remove_pos_tagging(sentences, target_words)
# print(len(list(sentences)))
target_words = [preprocessing.remove_pos_tagging_word(word) for word in target_words]
tokenized_target_sentences = xlmr_model.tokenize_sentences(sentences, target_words)
target_embeddings_dict = {target: [] for target in target_words}
target_sentences_dict = {target: [] for target in target_words}

xlmr_model.enter_eval_mode()

for tokenized_sentence, target_word_idx_dict in tqdm.tqdm(tokenized_target_sentences):
    target_embeddings = xlmr_model.compute_embeddings(tokenized_sentence, target_word_idx_dict)
    for target, embeddings in target_embeddings.items():
        target_embeddings_dict[target].extend(embeddings)
        sent = xlmr_model.xlmr.decode(tokenized_sentence)
        target_sentences_dict[target].extend([sent] * len(embeddings))

os.makedirs(base_out_path, exist_ok=True)

for target, target_embeddings in target_embeddings_dict.items():
    np.save(f"{base_out_path}{target}", target_embeddings)

os.makedirs(sentence_output_final_path, exist_ok=True)

for target, target_sentences in target_sentences_dict.items():
    df = pd.DataFrame(target_sentences)
    df.to_csv(f"{sentence_output_final_path}{target}.csv")




