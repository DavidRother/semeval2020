from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator


def find_best_clustering(data):
    sum_of_squared_distances = []
    K = range(1, min(len(data) + 1, 15))
    km_list = []
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        km_list.append(km)
        sum_of_squared_distances.append(km.inertia_)

    knee = KneeLocator(K, sum_of_squared_distances, curve="convex", direction="decreasing")
    if not knee.knee:
        return km_list[0]
    return km_list[knee.knee - 1].labels_


def find_gmm_clustering(data):
    sum_of_squared_distances = []
    K = range(1, min(len(data) + 1, 5))
    gmm_list = []
    for k in K:
        gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3)
        gmm_labels = gmm.fit_predict(data)
        gmm_list.append(gmm_labels)
        sum_of_squared_distances.append(gmm.bic(data))

    knee = KneeLocator(K, sum_of_squared_distances, curve="convex", direction="decreasing")
    if not knee.knee:
        return gmm_list[0]
    return gmm_list[knee.knee - 1]


def compute_cluster_sense_frequency(cluster_labels, embeddings_epoch_label, epoch_labels):
    n_cluster = len(set(cluster_labels))
    cluster_epoch_combined = list(zip(cluster_labels, embeddings_epoch_label))
    sense_frequencies = {epoch_label: [] for epoch_label in epoch_labels}
    for epoch in epoch_labels:
        count_epoch_total = sum(int(epoch == epoch_label) for cluster_label, epoch_label in cluster_epoch_combined)
        for sense_label in range(n_cluster):
            count_sense_epoch = sum(int(cluster_label == sense_label and epoch == epoch_label)
                                    for cluster_label, epoch_label in cluster_epoch_combined)
            sense_frequency_epoch = count_sense_epoch/count_epoch_total
            sense_frequencies[epoch].append(sense_frequency_epoch)
    return sense_frequencies
