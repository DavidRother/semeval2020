from semeval2020.model.bertwrapper import BertWrapper
from semeval2020.model.dataloader import Dataloader
from semeval2020.model import preprocessing

import torch
import os.path
import pandas as pd
import tqdm


########################################
#  Config Parameter ####################
########################################

language = 'german'
corpus = "corpus2"

base_path = "../../trial_data_public/"
model_string = "bert-base-multilingual-cased"

output_path = "../../semeval2020/embedding_data/"

max_length_sentence = 100
padding_length = 128

########################################
#  Code ################################
########################################

base_out_path = f"{output_path}{language}/{corpus}/"
os.makedirs(base_out_path, exist_ok=True)

data_loader = Dataloader(base_path, language=language, corpus=corpus)
bert_model = BertWrapper(model_string=model_string)

data_loader.sentences = preprocessing.sanitized_sentences(data_loader.sentences, max_len=max_length_sentence)
data_loader.sentences = preprocessing.filter_for_words(data_loader.sentences, data_loader.target_words)
tokenized_target_sentences = bert_model.tokenize_sentences(data_loader.sentences, data_loader.target_words)
target_embeddings_dict = {target: [] for target in data_loader.target_words}

bert_model.enter_eval_mode()

for tokenized_sentence, target_word_idx_dict in tqdm.tqdm(tokenized_target_sentences):
    input_ids = bert_model.get_tokenized_input_ids(tokenized_sentence, padding_length=padding_length)
    attention_mask = bert_model.get_attention_mask(input_ids)

    input_id_tensor = torch.tensor([input_ids])
    attention_mask_tensor = torch.tensor([attention_mask])
    target_embeddings = bert_model.compute_embeddings(input_id_tensor, attention_mask_tensor, target_word_idx_dict)
    for target, embeddings in target_embeddings.items():
        target_embeddings_dict[target].extend(embeddings)

for target, target_embeddings in target_embeddings_dict.items():
    df = pd.DataFrame.from_records(target_embeddings)
    df.to_csv(f"{base_out_path}{target}.csv")




