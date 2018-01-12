from auto_ml import Predictor
from eval_results import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data():
    # Load base data
    share_data = pd.read_pickle('data/ml-aug-sample.pkl.gz', compression='gzip')

    return share_data


def split_data(share_data):
    # Split into feature learning data (fl) and training data (tr)
    symbols = share_data['symbol'].unique()
    # For testing only take the first 10 elements
    # symbols = symbols[:10]
    symbol_map = {}
    symbol_num = 0

    print('No of symbols:', len(symbols))

    fl = pd.DataFrame()
    tr = pd.DataFrame()
    ts = pd.DataFrame()

    # prep data for fitting into both model types
    for symbol in symbols:
        model_data = share_data.loc[share_data['symbol'] == symbol]
        model_data.reset_index()

        print('Symbol:', symbol, ', num:', symbol_num, ', length:', len(model_data))

        # Create an 90 / 10 split for train / test
        msk = np.random.rand(len(model_data)) < 0.9

        # Prep data frames
        symbol_ts = model_data[~msk]
        symbol_main = model_data[msk]
        symbol_ts.reset_index()
        symbol_main.reset_index()

        # Now re-split the training data between deep learning and gradient boosting
        msk = np.random.rand(len(symbol_main)) < 0.7
        symbol_tr = symbol_main[msk]
        symbol_fl = symbol_main[~msk]
        symbol_tr.reset_index()
        symbol_fl.reset_index()

        print('Length symbol fl:', len(symbol_fl))
        print('Length symbol tr:', len(symbol_tr))
        print('Length symbol ts:', len(symbol_ts))

        tr = tr.append(symbol_tr)
        ts = ts.append(symbol_ts)

        # Set up map of symbol name to number
        symbol_map[symbol] = symbol_num

        print('Length fl:', len(fl))
        print('Length tr:', len(tr))
        print('Length ts:', len(ts))

        symbol_num += 1

        if symbol_num >= 15:
            break

    print(symbol_map)

    # Clean-up the initial data variable
    return symbol_map, fl, tr, ts


def verify_dates(df):
    #Fix date columns which may not be recoginsed as a dat
    df['quoteDate'] = pd.to_datetime(df['quoteDate'])
    df['exDividendDate'] = pd.to_datetime(df['exDividendDate'], errors='coerce')

    # Set relative dividend date as a number
    df['exDividendRelative'] =  df['exDividendDate'] - df['quoteDate']

    # Convert any string for nulls to integer
    df['exDividendRelative'] = df['exDividendRelative'].apply(
        lambda x: -999 if pd.isnull(x) else x.days)

    return df

def classify_target(df, column_name):
    # Remove all nan values
    df = df.dropna(subset=[column_name], how='all')
    # Convert numeric to label vals
    bins = [-99999999., -50., -25., -10., -5., -0.25, 0.25, 1., 2., 5., 10., 20., 50., 100., 1000., 99999999999999999.]
    # bin_labels = ['> 50% loss', '25 - 50% loss', '10 - 25% loss', '5 - 10% loss', '< 5% loss', 'Steady', '< 1% gain',
    #               '1 - 2% gain', '2 - 5% gain', '5 - 10% gain', '10 - 20% gain', '20 - 50% gain', '50 - 100% gain',
    #               '100 - 1000% gain', '> 1000% gain']

    df['Target'] = pd.cut(df[column_name], bins, labels=False)

    return df


def main():
    share_data = load_data()
    share_data = verify_dates(share_data)
    share_data = classify_target(share_data, 'Future8WeekReturn')
    symbol_map, fl, tr, ts = split_data(share_data)

    del share_data

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
        'Future8WeekReturn': 'ignore',
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


    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    # ml_predictor.train_categorical_ensemble(tr, optimize_final_model=False, categorical_column='symbol', take_log_of_y=True,
    #                                          perform_feature_selection=False, verbose=True, ml_for_analytics=True,
    #                                          model_names=['XGBClassifier', 'LGBMClassifier'],
    #                                          perform_feature_scaling=True, verify_features=False, cv=2, feature_learning=False,
    #                                          fl_data=fl,
    #                                          prediction_intervals=True)

    ml_predictor.train(tr, optimize_final_model=False, perform_feature_selection=False, verbose=True,
                       ml_for_analytics=True, model_names=['XGBClassifier', 'LGBMClassifier', 'DeepLearningClassifier'],
                       perform_feature_scaling=True, verify_features=False, cv=2,
                       # feature_learning=True, fl_data=fl,
                       scoring="accuracy_score")


    ml_predictor.score(ts, ts['Target'], verbose=3)
    ml_predictor.save(file_name='./models/auto_ml_sharecast_classifier_pipeline.dill', verbose=True)


if __name__ == "__main__":
    main()