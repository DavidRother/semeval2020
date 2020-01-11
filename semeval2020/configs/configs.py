from semeval2020.factory_hub import config_factory


paths = {"embedding_data_path": "../../semeval2020/embedding_data/",
         "truth_trial_data_path": "../../trial_data_public/truth/",
         "answer_path": "../../my_results/answer/answer/",
         "out_zip_path": "../../my_results/answer",
         "in_zip_path": "../../my_results/answer/",
         "trial_data_path": "../../trial_data_public/"}

naive_gmm = {"n_components": 4, "covariance_type": "diag", "reg_covar": 1e-3}

umap = {"n_neighbors": 10, "min_dist": 0.1, "metric": 'cosine', "n_components": 2}

dbscan = {"eps": 0.8, "min_samples": 5}


config_factory.register("ProjectPaths", paths)
config_factory.register("NaiveGMM", naive_gmm)
config_factory.register("UMAP", umap)
config_factory.register("DBSCAN", dbscan)
