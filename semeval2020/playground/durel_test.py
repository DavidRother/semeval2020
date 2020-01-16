import pandas as pd


durel_path = "../../DURel/testset/"
durel_testset = "testset.csv"

df = pd.read_csv(durel_path + durel_testset, sep="\t")

print(df.head())

print('done')
