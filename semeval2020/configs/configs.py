from semeval2020.factory_hub import config_factory


paths = {"embedding_data_path": "../../data/embedding_data/",
         "embedding_data_path_old": "../../data/embedding_data_old/",
         "truth_trial_data_path": "../../trial_data_public/truth/",
         "answer_path": "../../my_results/answer/answer/",
         "answer_path_old": "../../my_results/answer_old/answer/",
         "out_zip_path": "../../my_results/answer",
         "in_zip_path": "../../my_results/answer/",
         "out_zip_path_old": "../../my_results/answer.txt",
         "in_zip_path_old": "../../my_results/answer_old/answer/",
         "trial_data_path": "../../trial_data_public/",
         "sentence_data": "../../data/sentence_data/",
         "sense_sentence_mappings": "../../data/sense_to_sentence_mapping/",
         "corpus_path": "../../trial_data_public/corpora/"}

naive_gmm = {"n_components": 10, "covariance_type": "diag", "reg_covar": 1e-3}

umap = {"n_neighbors": 10, "min_dist": 0.0, "metric": 'cosine', "n_components": 10}

umap_ae = {"n_neighbors": 20, "min_dist": 0.0, "metric": 'euclidean', "n_components": 10}

dbscan = {"eps": 0.8, "min_samples": 5}

birch = {"n_clusters": None, "threshold": 1.1, "branching_factor": 300}

dbscan_birch = {"eps": 0.8, "min_samples": 5, "threshold": 1.1, "branching_factor": 50}

auto_encoder = {"learning_rate": 1e-3, "weight_decay": 1e-5, "num_epochs": 100, "batch_size": 128, "input_size": 768}

t_sne = {"n_components": 2, "perplexity": 50, "metric": "cosine"}

t_sne_ae = {"n_components": 2, "perplexity": 50, "metric": "euclidean"}


config_factory.register("ProjectPaths", paths)
config_factory.register("NaiveGMM", naive_gmm)
config_factory.register("UMAP", umap)
config_factory.register("UMAP_AE", umap_ae)
config_factory.register("DBSCAN", dbscan)
config_factory.register("BIRCH", birch)
config_factory.register("AutoEncoder", auto_encoder)
config_factory.register("TSNE", t_sne)
config_factory.register("TSNE_AE", t_sne_ae)
config_factory.register("DBSCAN_BIRCH", dbscan_birch)
