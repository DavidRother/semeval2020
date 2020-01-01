from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences


def sanitized_sentences(sentences, max_len=100):
    for sentence in sentences:
        for split_sentence in __split_sentence(sentence, max_len):
            yield split_sentence


def tokenize_sentences(sentences, tokenizer_model='bert-base-multilingual-cased',
                       word_to_index=None, padding_length=128):
    word_to_index = word_to_index or []
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
    tokenized_target_words = {word: tokenizer.tokenize(word) for word in word_to_index}
    for sentence in sentences:
        tokenized_text = tokenizer.tokenize(' '.join(["[CLS]"] + sentence + ["[SEP]"]))
        word_to_idx_dict = {word: [(i, i+len(tokenized_target_words[word])) for i, tok in enumerate(tokenized_text)
                                   if tokenized_text[i: i+len(tokenized_target_words[word])] ==
                                   tokenized_target_words[word]] for word in word_to_index}
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_text)], maxlen=padding_length,
                                  dtype="long", truncating="post", padding="post")[0]
        attention_mask = []


def __split_sentence(sentence, max_len):
    if len(sentence) <= max_len:
        yield sentence
    else:
        yield sentence[:max_len]
        for sentence_split in __split_sentence(sentence[max_len:], max_len):
            yield sentence_split

