from nltk.corpus import senseval
from nltk.stem import WordNetLemmatizer
from semeval2020.factory_hub import abstract_data_loader, data_loader_factory


class SensEvalLoader(abstract_data_loader.AbstractDataLoader):

    def __init__(self, base_path, language='english'):
        self.language = language
        self.base_path = base_path

    def __parse_target_words(self):
        target_file = f"{self.base_path}targets/{self.language}.txt"
        with open(target_file, 'r', encoding='utf-8') as f_in:
            return [line.strip().split('\t')[0] for line in f_in]

    @staticmethod
    def __parse_corpus(target_words, num_sentences_per_word):
        lem = WordNetLemmatizer()
        word_to_sentence = {}
        for word in target_words:
            word_to_sentence[word] = {'sentences': [], 'senses': []}
            for inst in senseval.instances(word + '.pos'):
                context = [lem.lemmatize(c[0]) for c in inst.context]
                if word in context:
                    word_to_sentence[word]['sentences'].append(context)
                    word_to_sentence[word]['senses'].append(inst.senses)
                if len(word_to_sentence[word]['sentences']) == num_sentences_per_word:
                    if len(word_to_sentence[word]['sentences']) != len(word_to_sentence[word]['senses']):
                        print('different numbers of senses and sentences')
                    break
        return word_to_sentence

    def load(self, num_words=100):
        target_words = self.__parse_target_words()
        return target_words, self.__parse_corpus(target_words, num_words)


data_loader_factory.register("senseval_loader", SensEvalLoader)
