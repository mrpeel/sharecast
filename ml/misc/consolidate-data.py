import numpy as np
import pandas as pd
from memory_profiler import profile
import gc
import random


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# Load data
@profile
def load_data(base_path, increments):
    loading_data = pd.DataFrame()
    for increment in increments:
        path = base_path % increment
        frame = pd.read_csv(path, compression='gzip', parse_dates=['quoteDate'], infer_datetime_format=True, low_memory=False)
        loading_data = loading_data.append(frame, ignore_index=True)
        del frame
        print('Loaded:', path)

    return loading_data

date_prefix = '20171224'
month_name = 'dec'

share_data = load_data(base_path='data/companyQuotes-' + date_prefix + '-%03d.csv.gz', increments=range(1, 61))
gc.collect()


print('Pickling data')

share_data.to_pickle('data/ml-' + month_name + '-data.pkl.gz', compression='gzip')

symbols = share_data['symbol'].unique().tolist()

random_500 = random.sample(symbols, 500)
sample2 = share_data[share_data['symbol'].isin(random_500)]

sample2.to_pickle('data/ml-' + month_name + '-sample.pkl.gz', compression='gzip')
