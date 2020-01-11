from semeval2020.factory_hub import abstract_evaluation_suite, evaluation_suite_factory
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata, spearmanr
import numpy as np

from typing import Dict, Union


class EvaluationSuite(abstract_evaluation_suite.AbstractEvaluationSuite):

    def __init__(self, languages=("german", "english", "latin", "swedish"), tasks=("task1", "task2")):
        self.languages = languages
        self.tasks = tasks

    def evaluate(self, predictions: Dict[str, Dict[str, Dict[str, Union[float, int]]]],
                 truth: Dict[str, Dict[str, Dict[str, Union[float, int]]]]):
        language_accuracies = {}
        language_ranking_scores = {}
        if "task1" in self.tasks:
            language_accuracies = {}
            for language in self.languages:
                language_accuracies[f"ACC {language}"] = self._compute_language_accuracy(
                    predictions["task1"][language], truth["task1"][language])
            language_accuracies['ACC ALL'] = np.mean(np.asarray(list(language_accuracies.values())))

        if "task2" in self.tasks:
            for language in self.languages:
                language_ranking_scores[f"RANKING SCORE {language}"] = self._compute_language_ranking_score(
                    predictions["task2"][language], truth["task2"][language])
            language_ranking_scores['RANKING SCORE ALL'] = np.mean(np.asarray(list(language_ranking_scores.values())))
        return language_accuracies, language_ranking_scores

    @staticmethod
    def _compute_language_accuracy(prediction: Dict[str, int], truth: Dict[str, int]):
        y_pred = []
        y_true = []
        for word in prediction:
            y_pred.append(prediction[word])
            y_true.append(truth[word])
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def _compute_language_ranking_score(prediction: Dict[str, float], truth: Dict[str, float]):
        y_pred = []
        y_true = []
        for word in prediction:
            y_pred.append(prediction[word])
            y_true.append(truth[word])
        return spearmanr(rankdata(y_true), rankdata(y_pred))[0]


evaluation_suite_factory.register("semeval_2020_trial_evaluation", EvaluationSuite)

