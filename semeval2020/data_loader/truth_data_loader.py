from semeval2020.factory_hub import abstract_data_loader, data_loader_factory


class TruthDataLoader(abstract_data_loader.AbstractDataLoader):

    def __init__(self, base_path, languages=("english", "german", "latin", "swedish"), tasks=("task1", "task2")):
        self.languages = languages
        self.tasks = tasks
        self.base_path = base_path

    def __parse_word_scores(self):
        truth_dict = {}
        for task in self.tasks:
            truth_dict[task] = {}
            for language in self.languages:
                truth_dict[task][language] = {}
                target_file = f"{self.base_path}{task}/{language}.txt"
                with open(target_file, 'r', encoding='utf-8') as f_in:
                    word_score_pairs = [line.strip().split('\t') for line in f_in]
                    for word, score in word_score_pairs:
                        truth_dict[task][language][word] = int(score) if task == "task1" else float(score)
        return truth_dict

    def load(self):
        return self.__parse_word_scores()


data_loader_factory.register("truth_data_loader", TruthDataLoader)
