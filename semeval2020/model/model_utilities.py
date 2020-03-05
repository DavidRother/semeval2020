from scipy.spatial import distance


def compute_cluster_sense_frequency(cluster_labels, embeddings_epoch_label, epoch_labels, k, n):
    n_cluster = len(set(cluster_labels))
    cluster_epoch_combined = list(zip(cluster_labels, embeddings_epoch_label))
    sense_frequencies_task1 = {epoch_label: [] for epoch_label in epoch_labels}
    sense_frequencies_task2 = {epoch_label: [] for epoch_label in epoch_labels}
    for epoch in epoch_labels:
        count_epoch_total = sum(int(epoch == epoch_label) for cluster_label, epoch_label in cluster_epoch_combined)
        for sense_label in range(n_cluster):
            count_sense_epoch = sum(int(cluster_label == sense_label and epoch == epoch_label)
                                    for cluster_label, epoch_label in cluster_epoch_combined)
            sense_frequency_epoch = count_sense_epoch / count_epoch_total
            sense_frequencies_task2[epoch].append(sense_frequency_epoch)
            sense_frequencies_task1[epoch].append(count_sense_epoch)
    return sense_frequencies_task1, sense_frequencies_task2


def compute_task1_answer(sense_frequencies, k, n):
    for epoch_1_count, epoch_2_count in zip(*sense_frequencies.values()):
        if (epoch_1_count <= k and epoch_2_count > n) or (epoch_2_count <= k and epoch_1_count > n):
            return 1
    return 0


def compute_task2_answer(sense_frequencies):
    return distance.jensenshannon(sense_frequencies[0], sense_frequencies[1], 2.0)


def compute_task_answers(cluster_labels, embeddings_epoch_label, epoch_labels, k, n):
    sf1, sf2 = compute_cluster_sense_frequency(cluster_labels, embeddings_epoch_label, epoch_labels, k, n)
    return compute_task1_answer(sf1, k, n), compute_task2_answer(sf2)

