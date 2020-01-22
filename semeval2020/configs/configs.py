from semeval2020.factory_hub import config_factory


paths = {"embedding_data_path": "../../data/embedding_data/",
         "truth_trial_data_path": "../../trial_data_public/truth/",
         "answer_path": "../../my_results/answer/answer/",
         "out_zip_path": "../../my_results/answer",
         "in_zip_path": "../../my_results/answer/",
         "trial_data_path": "../../trial_data_public/",
         "sentence_data": "../../data/sentence_data/",
         "sense_sentence_mappings": "../../data/sense_to_sentence_mapping/",
         "corpus_path": "../../trial_data_public/corpora/"}

naive_gmm = {"n_components": 4, "covariance_type": "diag", "reg_covar": 1e-3}

umap = {"n_neighbors": 5, "min_dist": 0.0, "metric": 'cosine', "n_components": 2}

dbscan = {"eps": 0.8, "min_samples": 5}

birch = {"n_clusters": None, "threshold": 1.1, "branching_factor": 50}


config_factory.register("ProjectPaths", paths)
config_factory.register("NaiveGMM", naive_gmm)
config_factory.register("UMAP", umap)
config_factory.register("DBSCAN", dbscan)
config_factory.register("BIRCH", birch)