from semeval2020.factory_hub import abstract_model, model_factory
import hdbscan
from semeval2020.model import model_utilities
from itertools import compress


class MyHDBSCAN(abstract_model.AbstractModel):

    def __init__(self, min_ratio=0.05, max_min_cluster_size_and_samples=100):
        self.hdbscan = None
        self.max_min_cluster_size_and_samples = max_min_cluster_size_and_samples
        self.min_ratio = min_ratio

    def fit(self, data):
        min_cluster_size = min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data)))
        min_samples = min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data)))
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        self.hdbscan.fit(data)

    def fit_predict(self, data, embedding_epochs_labeled=None, k=2, n=5):
        return self.predict(data, embedding_epochs_labeled, k=k, n=n)

    def predict(self, data, embedding_epochs_labeled=None, k=2, n=5):
        min_cluster_size = min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data)))
        min_samples = min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data)))
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

        labels = self.hdbscan.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        # if -1 in labels:
        #     indexer = [label != -1 for label in labels]
        #     labels = list(compress(labels, indexer))
        #     embedding_epochs_labeled = list(compress(embedding_epochs_labeled, indexer))
        return model_utilities.compute_task_answers(labels, embedding_epochs_labeled, epoch_labels, k, n)

    def fit_predict_labeling(self, data, **kwargs):
        min_cluster_size = min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data)))
        min_samples = min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data)))
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

        labels = self.hdbscan.fit_predict(data)
        return labels

    def predict_labeling(self, data, **kwargs):
        raise NotImplementedError()


model_factory.register("HDBSCAN", MyHDBSCAN)
