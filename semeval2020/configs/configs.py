from semeval2020.factory_hub import config_factory


paths = {"embedding_data_path": "../../data/embedding_data/",
         "embedding_data_path_main": "../../data/embedding_data_semeval2020/",
         "embedding_data_path_old": "../../data/embedding_data_old/",
         "embedding_data_path_old2": "../../data/embedding_data_old2/",
         "auto_embedding_data_path_old2": "../../data/auto_embedded_data_old2/",
         "auto_embedding_data_path_main": "../../data/auto_embedded_data_semeval2020/",
         "truth_trial_data_path": "../../trial_data_public/truth/",
         "truth_test_data_path": "../../data/test_data_truth/",
         "evalpy_path": "../../data/evalpy/",
         "answer_path": "../../my_results/answer/answer/",
         "answer_path_old": "../../my_results/answer_old/answer/",
         "answer_path_main": "../../my_results_main/answer/answer/",
         "out_zip_path": "../../my_results/answer",
         "in_zip_path": "../../my_results/answer/",
         "out_zip_path_main": "../../my_results_main/answer",
         "in_zip_path_main": "../../my_results_main/answer/",
         "out_zip_path_old": "../../my_results/answer.txt",
         "in_zip_path_old": "../../my_results/answer_old/answer/",
         "trial_data_path": "../../trial_data_public/",
         "sentence_data": "../../data/sentence_data/",
         "sense_sentence_mappings": "../../data/sense_to_sentence_mapping/",
         "corpus_path": "../../trial_data_public/corpora/"}

naive_gmm = {"n_components": 10, "covariance_type": "diag", "reg_covar": 1e-3}

gmm = {"n_components": 5, "covariance_type": "diag", "reg_covar": 1e-3}

umap = {"n_neighbors": 5, "min_dist": 0.0, "metric": 'cosine', "n_components": 2}

umap_ae = {"n_neighbors": 7, "min_dist": 0.0, "metric": 'cosine', "n_components": 2}

umap_ae_language = {"latin": {"n_neighbors": 3, "min_dist": 0.0, "metric": 'cosine', "n_components": 10},
                    "german": {"n_neighbors": 6, "min_dist": 0.0, "metric": 'cosine', "n_components": 10},
                    "english": {"n_neighbors": 6, "min_dist": 0.0, "metric": 'cosine', "n_components": 10},
                    "swedish": {"n_neighbors": 5, "min_dist": 0.0, "metric": 'cosine', "n_components": 10}}
# "n_neighbors": 5, "min_dist": 0.0, "metric": 'cosine', "n_components": 10

dbscan = {"eps": 2.5, "min_samples": 5}

birch = {"n_clusters": None, "threshold": 1.1, "branching_factor": 300}

dbscan_birch = {"eps": 2.5, "min_samples": 5, "threshold": 1.5, "branching_factor": 30, "max_clusters": 20}
# {"eps": 0.8, "min_samples": 3, "threshold": 1.5, "branching_factor": 30}

auto_encoder = {"learning_rate": 1e-3, "weight_decay": 1e-5, "num_epochs": 200, "batch_size": 128, "input_size": 768}

t_sne = {"n_components": 2, "perplexity": 50, "metric": "cosine"}

t_sne_ae = {"n_components": 2, "perplexity": 10, "metric": "cosine"}

t_sne_ae_language = {"latin": {"n_components": 10, "perplexity": 50, "metric": "cosine"},
                     "german": {"n_components": 10, "perplexity": 50, "metric": "cosine"},
                     "english": {"n_components": 10, "perplexity": 50, "metric": "cosine"},
                     "swedish": {"n_components": 10, "perplexity": 50, "metric": "cosine"}}

pca_ae_language = {"latin": {"n_components": 10},
                   "german": {"n_components": 10},
                   "english": {"n_components": 10},
                   "swedish": {"n_components": 10}}

affinity_propagation = {"preference": -50, "damping": 0.5}

agglomerative_clustering = {"n_clusters": None, "distance_threshold": 2.0}

hdbscan = {"latin": {"min_ratio": 0.010, "max_min_cluster_size_and_samples": 80, "noise_filter": True},
           "german": {"min_ratio": 0.06, "max_min_cluster_size_and_samples": 80, "noise_filter": True},
           "english": {"min_ratio": 0.030, "max_min_cluster_size_and_samples": 80, "noise_filter": True},
           "swedish": {"min_ratio": 0.025, "max_min_cluster_size_and_samples": 80, "noise_filter": True}}

hdbscan_language = {"latin": {"min_ratio": 0.010, "max_min_cluster_size_and_samples": 80, "noise_filter": True},
                    "german": {"min_ratio": 0.06, "max_min_cluster_size_and_samples": 80, "noise_filter": True},
                    "english": {"min_ratio": 0.030, "max_min_cluster_size_and_samples": 80, "noise_filter": True},
                    "swedish": {"min_ratio": 0.025, "max_min_cluster_size_and_samples": 80, "noise_filter": True}}

dbscan_language = {"latin": {"eps": 2.5, "min_samples": 5},
                   "german": {"eps": 2.5, "min_samples": 5},
                   "english": {"eps": 2.5, "min_samples": 5},
                   "swedish": {"eps": 2.5, "min_samples": 5}}

dbscan_birch_language = {"latin": {"eps": 2.5, "min_samples": 5, "threshold": 1.5, "branching_factor": 30, "max_clusters": 20},
                         "german": {"eps": 2.5, "min_samples": 5, "threshold": 1.5, "branching_factor": 30, "max_clusters": 20},
                         "english": {"eps": 2.5, "min_samples": 5, "threshold": 1.5, "branching_factor": 30, "max_clusters": 20},
                         "swedish": {"eps": 2.5, "min_samples": 5, "threshold": 1.5, "branching_factor": 30, "max_clusters": 20}}

gmm_language = {"latin": {"n_components": 5, "covariance_type": "diag", "reg_covar": 1e-3},
                "german": {"n_components": 5, "covariance_type": "diag", "reg_covar": 1e-3},
                "english": {"n_components": 5, "covariance_type": "diag", "reg_covar": 1e-3},
                "swedish": {"n_components": 5, "covariance_type": "diag", "reg_covar": 1e-3}}

task_params = {"latin": {"k": 0, "n": 1},
               "german": {"k": 2, "n": 5},
               "english": {"k": 2, "n": 5},
               "swedish": {"k": 2, "n": 5}}


config_factory.register("ProjectPaths", paths)
config_factory.register("NaiveGMM", naive_gmm)
config_factory.register("GMM", gmm)
config_factory.register("UMAP", umap)
config_factory.register("UMAP_AE", umap_ae)
config_factory.register("UMAP_AE_Language", umap_ae_language)
config_factory.register("DBSCAN", dbscan)
config_factory.register("BIRCH", birch)
config_factory.register("AutoEncoder", auto_encoder)
config_factory.register("TSNE", t_sne)
config_factory.register("TSNE_AE", t_sne_ae)
config_factory.register("DBSCAN_BIRCH", dbscan_birch)
config_factory.register("AffinityPropagation", affinity_propagation)
config_factory.register("AgglomerativeClustering", agglomerative_clustering)
config_factory.register("TaskParameter", task_params)
config_factory.register("HDBSCAN", hdbscan)
config_factory.register("HDBSCANLanguage", hdbscan_language)
config_factory.register("DBSCANLanguage", dbscan_language)
config_factory.register("PCA_AE_Language", pca_ae_language)
