from semeval2020.factory_hub import data_loader_factory
base_path = "../../trial_data_public/truth/"
truth_data_loader = data_loader_factory.create_data_loader("truth_data_loader", base_path=base_path)
data = truth_data_loader.load()


