# BERT imports
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertModel
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

sentence = "Hi this is me a guy named David"
ex_sent = "[CLS] " + sentence + " [SEP]"

print(ex_sent)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_sent = tokenizer.tokenize(ex_sent)

print(tokenized_sent)

# pad the input tokens
MAX_LEN = 128

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_sent)], maxlen=MAX_LEN, dtype="long",
                          truncating="post", padding="post")

# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

model = BertModel.from_pretrained("bert-base-uncased")

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)

out = model(prediction_inputs, token_type_ids=None, attention_mask=prediction_masks)

print('finished')
