from semeval2020.model.bertwrapper import BertWrapper
from semeval2020.data_loader.sentence_loader import SentenceLoader
from semeval2020.model import preprocessing

import torch
import os.path
import pandas as pd
import tqdm


########################################
#  Config Parameter ####################
########################################

language = 'latin'
corpus = "corpus1"

base_path = "../../data/main_task_data/"
model_string = "bert-base-multilingual-cased"  # "dbmdz/bert-base-german-cased"

output_path = "../../data/embedding_data_semeval2020/"
sentence_out_path = "../../data/sentence_data_semeval2020/"

max_length_sentence = 70
padding_length = 100

auto = False

########################################
#  Code ################################
########################################

base_out_path = f"{output_path}{language}/{corpus}/"
sentence_output_final_path = f"{sentence_out_path}{language}/{corpus}/"
os.makedirs(base_out_path, exist_ok=True)

data_loader = SentenceLoader(base_path, language=language, corpus=corpus)
bert_model = BertWrapper(model_string=model_string, auto=auto)

target_words, sentences = data_loader.load()
sentences = preprocessing.sanitized_sentences(sentences, max_len=max_length_sentence)
sentences = (sentence[1:] for sentence in sentences)
sentences = preprocessing.filter_for_words(sentences, target_words)
sentences = preprocessing.remove_pos_tagging(sentences, target_words)
tokenized_target_sentences = bert_model.tokenize_sentences(sentences, target_words)
target_embeddings_dict = {target: [] for target in target_words}
target_sentences_dict = {target: [] for target in target_words}

bert_model.enter_eval_mode()

for tokenized_sentence, target_word_idx_dict in tqdm.tqdm(tokenized_target_sentences):
    input_ids = bert_model.get_tokenized_input_ids(tokenized_sentence, padding_length=padding_length)
    attention_mask = bert_model.get_attention_mask(input_ids)

    input_id_tensor = torch.tensor([input_ids])
    attention_mask_tensor = torch.tensor([attention_mask])
    target_embeddings = bert_model.compute_embeddings(input_id_tensor, attention_mask_tensor, target_word_idx_dict)
    for target, embeddings in target_embeddings.items():
        target_embeddings_dict[target].extend(embeddings)
        sent = ' '.join(tokenized_sentence)
        target_sentences_dict[target].extend([sent] * len(embeddings))

os.makedirs(base_out_path, exist_ok=True)

for target, target_embeddings in target_embeddings_dict.items():
    df = pd.DataFrame.from_records(target_embeddings)
    df.to_csv(f"{base_out_path}{target}.csv")

os.makedirs(sentence_output_final_path, exist_ok=True)

for target, target_sentences in target_sentences_dict.items():
    df = pd.DataFrame(target_sentences)
    df.to_csv(f"{sentence_output_final_path}{target}.csv")




