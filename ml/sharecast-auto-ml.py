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

        # Make sure a minimum number of rows are present in sample for symbol
        if len(symbol_tr) > 150:
            fl = fl.append(symbol_fl)
            # Temporary - only add first 15 to main training set
            if symbol_num < 15:
                tr = tr.append(symbol_tr)
                ts = ts.append(symbol_ts)

                # Set up map of symbol name to number
                symbol_map[symbol] = symbol_num
        else:
            print(symbol, 'has insufficient training data, tr len:', len(symbol_tr))

        print('Length fl:', len(fl))
        print('Length tr:', len(tr))
        print('Length ts:', len(ts))

        symbol_num += 1

    print(symbol_map)

    # Clean-up the initial data variable
    return symbol_map, fl, tr, ts


def clip_and_scale_values(df, column_name):
    # Remove all nan values
    df = df.dropna(subset=[column_name], how='all')
    # Clip to -99 to 1000 range
    df[column_name] = df[column_name].clip(-99, 1000)

    # Scale to range 4 - 5.  This fixes issues with log of y for values < 1 or log of y becoming
    #  a value below 1.  By setting the minimum value to 4, a log of y value outside the current
    #  range is unlikely to drop below 1
    scaler = MinMaxScaler(feature_range=(4,5))
    df['Target'] = df[column_name]
    # Scaler requires values as a 2 dimeniosnal array, so reshape values on input
    df['Target'] = scaler.fit_transform(df['Target'].values.reshape(df['Target'].values.shape[0],1))

    #Fix date columns which may not be recoginsed as a dat
    df['quoteDate'] = pd.to_datetime(df['quoteDate'])
    df['exDividendDate'] = pd.to_datetime(df['exDividendDate'], errors='coerce')

    # Set relative dividend date as a number
    df['exDividendRelative'] =  df['exDividendDate'] - df['quoteDate']

    # Convert any string for nulls to integer
    df['exDividendRelative'] = df['exDividendRelative'].apply(
        lambda x: -999 if pd.isnull(x) else x.days)

    return scaler, df


def main():
    share_data = load_data()
    target_scaler, share_data = clip_and_scale_values(share_data, 'Future8WeekReturn')
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


    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor. train_categorical_ensemble(tr, optimize_final_model=False, categorical_column='symbol', take_log_of_y=True,
                                             perform_feature_selection=False, verbose=True, ml_for_analytics=True,
                                             model_names=['XGBRegressor', 'LGBMRegressor'],
                                             perform_feature_scaling=True, verify_features=False, cv=2, feature_learning=False,
                                             # fl_data=fl,
                                             prediction_intervals=True)

    ml_predictor.score(ts, ts['Target'], verbose=3)
    ml_predictor.save(file_name='./models/auto_ml_sharecast_pipeline.dill', verbose=True)

    test_y = ts['Target'].values
    prediction_vals = np.array(ml_predictor.predict(ts))
    prediction_vals = prediction_vals.reshape(prediction_vals.shape[0],1)
    transformed_prediction_vals = target_scaler.inverse_transform(prediction_vals)

    eval_results({'auto_ml_predictor': {
        'actual_y': test_y,
        'y_predict': transformed_prediction_vals
    }
    })


    range_results({
        'auto_ml_predictor': transformed_prediction_vals
        }, test_y)

if __name__ == "__main__":
    main()