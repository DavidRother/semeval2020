class EmbeddingLoader:

    def __init__(self, base_path, language='german', corpus='corpus2', explicit_word_list=None):
        self.explicit_word_list = explicit_word_list or []
        self.language = language
        self.corpus = corpus
        self.base_path = base_path
        self.target_words = []

