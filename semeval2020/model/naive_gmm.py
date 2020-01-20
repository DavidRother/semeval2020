from semeval2020.factory_hub import abstract_model, model_factory
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from scipy.spatial import distance


class NaiveGMM(abstract_model.AbstractModel):

    def __init__(self, n_components, covariance_type, reg_covar):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.gmm = None

    def fit(self, data):
        self.gmm = self._find_clustering(data)

    def fit_predict(self, data, embedding_epochs_labeled=None):
        self.fit(data)
        return self.predict(data, embedding_epochs_labeled)

    def predict(self, data, embedding_epochs_labeled=None):
        labels = self.gmm.predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        sense_frequencies = self.compute_cluster_sense_frequency(labels, embedding_epochs_labeled, epoch_labels)
        task_1_answer = int(any([True for sd in sense_frequencies if 0 in sense_frequencies[sd]]))
        task_2_answer = distance.jensenshannon(sense_frequencies[0], sense_frequencies[1], 2.0)
        return task_1_answer, task_2_answer

    def fit_predict_labeling(self, data, **kwargs):
        self.fit(data)
        return self.gmm.predict(data)

    def _find_clustering(self, data):
        sum_of_squared_distances = []
        K = range(1, min(len(data) + 1, 5))
        gmm_list = []
        for k in K:
            gmm = GaussianMixture(n_components=k, covariance_type=self.covariance_type, reg_covar=self.reg_covar)
            gmm.fit(data)
            gmm_list.append(gmm)
            sum_of_squared_distances.append(gmm.bic(data))

        knee = KneeLocator(K, sum_of_squared_distances, curve="convex", direction="decreasing")
        if not knee.knee:
            return gmm_list[0]
        return gmm_list[knee.knee - 1]

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


model_factory.register("NaiveGMM", NaiveGMM)

