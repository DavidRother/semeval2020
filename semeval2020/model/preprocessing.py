

def sanitized_sentences(sentences, max_len=100):
    for sentence in sentences:
        for split_sentence in __split_sentence(sentence, max_len):
            yield split_sentence


def __split_sentence(sentence, max_len):
    if len(sentence) <= max_len:
        yield sentence
    else:
        yield sentence[:max_len]
        for sentence_split in __split_sentence(sentence[max_len:], max_len):
            yield sentence_split


def filter_for_words(sentences, target_words):
    for sentence in sentences:
        if any((target for target in target_words if target in sentence)):
            yield sentence

