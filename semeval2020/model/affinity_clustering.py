from sklearn.cluster import AffinityPropagation
from semeval2020.factory_hub import abstract_model, model_factory
from scipy.spatial import distance
import numpy as np


class MyAffinityPropagation(abstract_model.AbstractModel):

    def __init__(self, preference=-50, damping=0.5):
        self.ap = AffinityPropagation(preference=preference, damping=damping)

    def fit(self, data):
        self.ap.fit(data)

    def fit_predict(self, data, embedding_epochs_labeled=None):
        return self.predict(data, embedding_epochs_labeled)

    def predict(self, data, embedding_epochs_labeled=None):
        labels = self.ap.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        sense_frequencies = self.compute_cluster_sense_frequency(labels, embedding_epochs_labeled, epoch_labels)
        task_1_answer = int(any([True for sd in sense_frequencies if 0 in sense_frequencies[sd]]))
        task_2_answer = distance.jensenshannon(sense_frequencies[0], sense_frequencies[1], 2.0)
        if np.isnan(task_2_answer):
            task_1_answer = 0
            task_2_answer = 0.5
        return task_1_answer, task_2_answer

    def fit_predict_labeling(self, data, **kwargs):
        self.fit(data)
        return self.ap.predict(data)

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


model_factory.register("AffinityPropagation", MyAffinityPropagation)

