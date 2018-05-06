import pandas as pd
from optimise_dataframe import *

# file = './data/ml-2018-03-processed.pkl.gz'
file = './data/data_with_labels.pkl.gz'

df = pd.read_pickle(file, compression='gzip')

df = optimise_df(df)

print(df.info())

print('Saving processed data')
df.to_pickle(file, compression='gzip')

