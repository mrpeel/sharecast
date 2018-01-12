from tpot import TPOTClassifier
from eval_results import *
import numpy as np
import pandas as pd

column_descriptions = {
    'symbol': 'categorical',
    'quoteDate': 'date',
    'exDividendDate': 'date',
    'lastTradePriceOnly': 'ignore',
    '4WeekBollingerPrediction': 'categorical',
    '4WeekBollingerType': 'categorical',
    '12WeekBollingerPrediction': 'categorical',
    '12WeekBollingerType': 'categorical',
    'Target': 'output',
    'Future8WeekRiskAdjustedReturn': 'ignore',
    'Future12WeekDividend': 'ignore',
    'Future12WeekPrice': 'ignore',
    'Future12WeekReturn': 'ignore',
    'Future12WeekRiskAdjustedReturn': 'ignore',
    'Future26WeekDividend': 'ignore',
    'Future26WeekPrice': 'ignore',
    'Future26WeekReturn': 'ignore',
    'Future26WeekRiskAdjustedReturn': 'ignore',
    'Future52WeekDividend': 'ignore',
    'Future52WeekPrice': 'ignore',
    'Future52WeekReturn': 'ignore',
    'Future52WeekRiskAdjustedReturn': 'ignore',
}

def load_data():
    print('Loading data')
    # Load base data
    share_data = pd.read_pickle('data/ml-aug-sample.pkl.gz', compression='gzip')

    for key in column_descriptions:
        if column_descriptions[key]=='ignore' and key in share_data.columns:
            share_data = share_data.drop([key])

    return share_data


def split_data(share_data):
    print('Splitting data')
    # Split into feature learning data (fl) and training data (tr)
    symbols = share_data['symbol'].unique()
    # For testing only take the first 10 elements
    # symbols = symbols[:10]
    symbol_map = {}
    symbol_num = 0

    print('No of symbols:', len(symbols))

    tr = pd.DataFrame()
    ts = pd.DataFrame()

    # prep data for fitting into both model types
    for symbol in symbols:
        model_data = share_data.loc[share_data['symbol'] == symbol]
        model_data.reset_index()

        print('Symbol:', symbol, ', num:', symbol_num, ', length:', len(model_data))

        # Create an 85 / 15 split for train / test
        msk = np.random.rand(len(model_data)) < 0.85

        # Prep data frames
        symbol_ts = model_data[~msk]
        symbol_tr = model_data[msk]
        symbol_ts.reset_index()
        symbol_tr.reset_index()

        print('Length symbol tr:', len(symbol_tr))
        print('Length symbol ts:', len(symbol_ts))

        tr = tr.append(symbol_tr)
        ts = ts.append(symbol_ts)

        # Set up map of symbol name to number
        symbol_map[symbol] = symbol_num

        print('Length tr:', len(tr))
        print('Length ts:', len(ts))

        symbol_num += 1

        if symbol_num > 15:
            break


    print(symbol_map)

    # Clean-up the initial data variable
    return symbol_map, tr, ts


def verify_dates(df):
    print('Verifying dates')
    #Fix date columns which may not be recoginsed as a dat
    df['quoteDate'] = pd.to_datetime(df['quoteDate'])
    df['exDividendDate'] = pd.to_datetime(df['exDividendDate'], errors='coerce')

    # Set relative dividend date as a number
    df['exDividendRelative'] =  df['exDividendDate'] - df['quoteDate']

    # Convert any string for nulls to integer
    df['exDividendRelative'] = df['exDividendRelative'].apply(
        lambda x: -999 if pd.isnull(x) else x.days)

    return df

def classify_target(df, column_index):
    print('Classifying target values')
    # Remove all nan values
    df = df.dropna(subset=[column_index], how='all')
    # Convert numeric to label vals
    bins = [-99999999., -50., -25., -10., -5., -0.25, 0.25, 1., 2., 5., 10., 20., 50., 100., 1000., 99999999999999999.]
    # bin_labels = ['> 50% loss', '25 - 50% loss', '10 - 25% loss', '5 - 10% loss', '< 5% loss', 'Steady', '< 1% gain',
    #               '1 - 2% gain', '2 - 5% gain', '5 - 10% gain', '10 - 20% gain', '20 - 50% gain', '50 - 100% gain',
    #               '100 - 1000% gain', '> 1000% gain']

    df['Target'] = pd.cut(df[column_index], bins, labels=False)

    df = df.drop([column_index], axis=1)

    return df



def main():
    pd_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    pd_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')

    pd_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    pd_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    pd_train_y = classify_target(pd_train_actuals, 0)
    pd_test_y = classify_target(pd_test_actuals, 0)


    train_x = pd_train_x.values
    train_y = pd_train_y['Target'].values

    test_x = pd_test_x.values
    test_y = pd_test_y['Target'].values

    del pd_train_x, pd_test_x, pd_train_actuals, pd_test_actuals, pd_train_y, pd_test_y

    tpot = TPOTClassifier(verbosity=3, cv=2, subsample=0.7, early_stop=3, max_eval_time_mins=10,
                          periodic_checkpoint_folder='./models/')

    tpot.fit(train_x, train_y)
    print(tpot.score(test_x, test_y))
    tpot.export('./models/tpot_sharecast_pipeline.py')


if __name__ == "__main__":
    main()