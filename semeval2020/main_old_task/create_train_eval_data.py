from semeval2020.factory_hub import config_factory
import random


project_paths = config_factory.get_config("ProjectPaths")

paths = ["../../old_task_data/corpora/german/corpus1/corpus1.txt",
         "../../old_task_data/corpora/german/corpus2/corpus2.txt"]

with open("../../old_task_data/corpora/german/corpus1/corpus1.txt") as f:
    lines1 = [line.rstrip() for line in f]

with open("../../old_task_data/corpora/german/corpus2/corpus2.txt") as f:
    lines2 = [line.rstrip() for line in f]

train = []
test = []

for line in lines1 + lines2:
    rand = random.uniform(0, 1)
    if rand > 0.85:
        test.append(line)
    else:
        train.append(line)

with open('../../data/german_old/train.txt', 'w') as f:
    f.write('\n'.join(train))

with open('../../data/german_old/test.txt', 'w') as f:
    f.write('\n'.join(test))
