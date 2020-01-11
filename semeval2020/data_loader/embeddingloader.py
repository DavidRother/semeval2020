from os import listdir
import os.path
import pandas as pd
from semeval2020.factory_hub import abstract_data_loader, data_loader_factory


class EmbeddingLoader(abstract_data_loader.AbstractDataLoader):

    def __init__(self, base_path, language='german', corpus='corpus2', explicit_word_list=None):
        self.explicit_word_list = explicit_word_list or []
        self.language = language
        self.corpus = corpus
        self.base_path = base_path
        self.target_words = self.find_target_words()
        self.embeddings = self.load_embeddings(self.target_words)

    @staticmethod
    def _find_csv_filenames(path_to_dir, suffix=".csv"):
        filenames = listdir(path_to_dir)
        return [filename for filename in filenames if filename.endswith(suffix)]

    def find_target_words(self):
        target_dir = f"{self.base_path}{self.language}/{self.corpus}/"
        csv_filenames = self._find_csv_filenames(target_dir)
        return [os.path.splitext(filename)[0] for filename in csv_filenames]

    def load_embeddings(self, target_words):
        target_dir = f"{self.base_path}{self.language}/{self.corpus}/"
        embedding_dict = {target_word: None for target_word in target_words}
        for filename in self._find_csv_filenames(target_dir):
            word = os.path.splitext(filename)[0]
            embedding_dict[word] = pd.read_csv(f"{target_dir}/{filename}")
        return embedding_dict

    def load(self):
        target_words = self.find_target_words()
        return target_words, self.load_embeddings(target_words)

    def __getitem__(self, key):
        return self.embeddings[key]


data_loader_factory.register("embedding_loader", EmbeddingLoader)
