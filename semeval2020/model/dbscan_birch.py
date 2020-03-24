from semeval2020.factory_hub import abstract_model, model_factory
from scipy.spatial import distance
from semeval2020.model import model_utilities
from sklearn.cluster import DBSCAN, Birch
from itertools import compress
import numpy as np


class MyDBSCANBIRCH(abstract_model.AbstractModel):

    def __init__(self, eps=1, min_samples=5, threshold=1.1, branching_factor=50, max_clusters=10):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.birch = None
        self.max_clusters = max_clusters

    def fit(self, data):
        labels = self.dbscan.fit_predict(data)
        num_labels = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
        self.birch = Birch(n_clusters=num_labels, threshold=self.threshold, branching_factor=self.branching_factor)

    def fit_predict(self, data, embedding_epochs_labeled=None, k=2, n=5):
        return self.predict(data, embedding_epochs_labeled, k=k, n=n)

    def predict(self, data, embedding_epochs_labeled=None, k=2, n=5):
        labels = self.dbscan.fit_predict(data)
        num_labels = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
        num_labels = min(self.max_clusters, num_labels)
        self.birch = Birch(n_clusters=num_labels, threshold=self.threshold, branching_factor=self.branching_factor)
        labels = self.birch.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        return model_utilities.compute_task_answers(labels, embedding_epochs_labeled, epoch_labels, k, n)

    def fit_predict_labeling(self, data, **kwargs):
        labels = self.dbscan.fit_predict(data)
        num_labels = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
        self.birch = Birch(n_clusters=num_labels, threshold=self.threshold, branching_factor=self.branching_factor)
        return self.birch.fit_predict(data)

    def predict_with_extra_return(self, data, embedding_epochs_labeled=None, k=2, n=5):
        labels = self.dbscan.fit_predict(data)
        num_labels = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
        num_labels = min(self.max_clusters, num_labels)
        self.birch = Birch(n_clusters=num_labels, threshold=self.threshold, branching_factor=self.branching_factor)
        labels = self.birch.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        task_answers = model_utilities.compute_task_answers(labels, embedding_epochs_labeled, epoch_labels, k, n)
        return task_answers[0], task_answers[1], labels

    def predict_labeling(self, data, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def compute_cluster_sense_frequency(cluster_labels, embeddings_epoch_label, epoch_labels):
        n_cluster = len(set(cluster_labels))
        cluster_epoch_combined = list(zip(cluster_labels, embeddings_epoch_label))
        sense_frequencies = {epoch_label: [] for epoch_label in epoch_labels}
        for epoch in epoch_labels:
            count_epoch_total = sum(int(epoch == epoch_label) for cluster_label, epoch_label in cluster_epoch_combined)
            for sense_label in range(n_cluster):
                count_sense_epoch = sum(int(cluster_label == sense_label and epoch == epoch_label)
                                        for cluster_label, epoch_label in cluster_epoch_combined)
                sense_frequency_epoch = count_sense_epoch / count_epoch_total
                sense_frequencies[epoch].append(sense_frequency_epoch)
        return sense_frequencies


model_factory.register("DBSCAN_BIRCH", MyDBSCANBIRCH)
model_factory.register("DBSCAN_BIRCHLanguage", MyDBSCANBIRCH)
