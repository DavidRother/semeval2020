from tokenizers import ByteLevelBPETokenizer
from semeval2020.factory_hub import config_factory

project_paths = config_factory.get_config("ProjectPaths")

paths = ["../../old_task_data/corpora/german/corpus1/corpus1.txt",
         "../../old_task_data/corpora/german/corpus2/corpus2.txt"]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>",
                                                                                 "<mask>"])

# Save files to disk
tokenizer.save("../../data/my_tokenizer", "old_task_german")


