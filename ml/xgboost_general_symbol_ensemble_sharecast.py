
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pickle
import gzip
import sys
#from sklearn.externals import joblib
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neighbors import KNeighborsRegressor
from categorical_encoder import *
from eval_results import *
#from autoencoder import *

from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

#import matplotlib.pyplot as plt

# from memory_profiler import profile




COLUMNS = ['symbol', '4WeekBollingerPrediction', '4WeekBollingerType', '12WeekBollingerPrediction',
            '12WeekBollingerType', 'adjustedPrice', 'quoteMonth', 'volume', 'previousClose', 'change',
            'changeInPercent', '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
            'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'Price200DayAverage',
            'Price52WeekPercChange', '1WeekVolatility', '2WeekVolatility', '4WeekVolatility', '8WeekVolatility',
            '12WeekVolatility', '26WeekVolatility', '52WeekVolatility', 'allordpreviousclose', 'allordchange',
            'allorddayshigh', 'allorddayslow', 'allordpercebtChangeFrom52WeekHigh',
            'allordpercentChangeFrom52WeekLow', 'asxpreviousclose', 'asxchange', 'asxdayshigh',
            'asxdayslow', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow', 'exDividendRelative',
            'exDividendPayout', '640106_A3597525W', 'AINTCOV', 'AverageVolume', 'BookValuePerShareYear',
            'CashPerShareYear', 'DPSRecentYear', 'EBITDMargin', 'EPS', 'EPSGrowthRate10Years',
            'EPSGrowthRate5Years', 'FIRMMCRT', 'FXRUSD', 'Float', 'GRCPAIAD', 'GRCPAISAD', 'GRCPBCAD',
            'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD', 'H01_GGDPCVGDP', 'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP',
            'IAD', 'LTDebtToEquityQuarter', 'LTDebtToEquityYear', 'MarketCap',
            'NetIncomeGrowthRate5Years', 'NetProfitMarginPercent', 'OperatingMargin', 'PE',
            'PriceToBook', 'ReturnOnAssets5Years', 'ReturnOnAssetsTTM', 'ReturnOnAssetsYear',
            'ReturnOnEquity5Years', 'ReturnOnEquityTTM', 'ReturnOnEquityYear', 'RevenueGrowthRate10Years',
            'RevenueGrowthRate5Years', 'TotalDebtToAssetsQuarter', 'TotalDebtToAssetsYear',
            'TotalDebtToEquityQuarter', 'TotalDebtToEquityYear', 'bookValue', 'earningsPerShare',
            'ebitda', 'epsEstimateCurrentYear', 'marketCapitalization', 'peRatio', 'pegRatio', 'pricePerBook',
            'pricePerEpsEstimateCurrentYear', 'pricePerEpsEstimateNextYear', 'pricePerSales']


# returns = {
#     '1': 'Future1WeekReturn',
#     '2': 'Future2WeekReturn',
#     '4': 'Future4WeekReturn',
#     '8': 'Future8WeekReturn',
#     '12': 'Future12WeekReturn',
#     '26': 'Future26WeekReturn',
#     '52': 'Future52WeekReturn',
#     '1ra': 'Future1WeekRiskAdjustedReturn',
#     '2ra': 'Future2WeekRiskAdjustedReturn',
#     '4ra': 'Future4WeekRiskAdjustedReturn',
#     '8ra': 'Future8WeekRiskAdjustedReturn',
#     '12ra': 'Future12WeekRiskAdjustedReturn',
#     '26ra': 'Future26WeekRiskAdjustedReturn',
#     '52ra': 'Future52WeekRiskAdjustedReturn'
# }


LABEL_COLUMN = "Future8WeekReturn"
CATEGORICAL_COLUMNS = ['symbol', '4WeekBollingerPrediction', '4WeekBollingerType',
                       '12WeekBollingerPrediction', '12WeekBollingerType', 'quoteDate_YEAR',
                       'quoteDate_MONTH', 'quoteDate_DAY', 'quoteDate_DAYOFWEEK']
CONTINUOUS_COLUMNS = ['adjustedPrice', 'quoteDate_TIMESTAMP', 'volume', 'previousClose', 'change',
                'changeInPercent','52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
                'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'Price200DayAverage',
                'Price52WeekPercChange', '1WeekVolatility', '2WeekVolatility', '4WeekVolatility', '8WeekVolatility',
                '12WeekVolatility', '26WeekVolatility', '52WeekVolatility', 'allordpreviousclose', 'allordchange',
                'allorddayshigh', 'allorddayslow', 'allordpercebtChangeFrom52WeekHigh',
                'allordpercentChangeFrom52WeekLow', 'asxpreviousclose', 'asxchange', 'asxdayshigh',
                'asxdayslow', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow', 'exDividendRelative',
                'exDividendPayout', '640106_A3597525W', 'AINTCOV', 'AverageVolume', 'BookValuePerShareYear',
                'CashPerShareYear', 'DPSRecentYear', 'EBITDMargin', 'EPS', 'EPSGrowthRate10Years',
                'EPSGrowthRate5Years', 'FIRMMCRT', 'FXRUSD', 'Float', 'GRCPAIAD', 'GRCPAISAD', 'GRCPBCAD',
                'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD', 'H01_GGDPCVGDP', 'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP',
                'IAD', 'LTDebtToEquityQuarter', 'LTDebtToEquityYear', 'MarketCap',
                'NetIncomeGrowthRate5Years', 'NetProfitMarginPercent', 'OperatingMargin', 'PE',
                'PriceToBook', 'ReturnOnAssets5Years', 'ReturnOnAssetsTTM', 'ReturnOnAssetsYear',
                'ReturnOnEquity5Years', 'ReturnOnEquityTTM', 'ReturnOnEquityYear', 'RevenueGrowthRate10Years',
                'RevenueGrowthRate5Years', 'TotalDebtToAssetsQuarter', 'TotalDebtToAssetsYear',
                'TotalDebtToEquityQuarter', 'TotalDebtToEquityYear', 'bookValue', 'earningsPerShare',
                'ebitda', 'epsEstimateCurrentYear', 'marketCapitalization', 'peRatio', 'pegRatio', 'pricePerBook',
                'pricePerEpsEstimateCurrentYear', 'pricePerEpsEstimateNextYear', 'pricePerSales',
                '4WeekBollingerBandLower', '4WeekBollingerBandUpper', '12WeekBollingerBandLower',
                '12WeekBollingerBandUpper', 'Beta', 'daysHigh', 'daysLow']


def save(object, filename):
        """Saves a compressed object to disk
        """
        with gzip.open(filename, 'wb') as f:
            f.write(pickle.dumps(object, pickle.HIGHEST_PROTOCOL))

        # max_bytes = 2 ** 31 - 1
        #
        # ## write
        # bytes_out = pickle.dumps(object)
        # with open(filename, 'wb') as f_out:
        #     for idx in range(0, sys.getsizeof(object), max_bytes):
        #         f_out.write(bytes_out[idx:idx + max_bytes])

# @profile
def load(filename):
        """Loads a compressed object from disk
        """
        # file = gzip.GzipFile(filename, 'rb')
        # buffer = ""
        # while True:
        #         data = file.read()
        #         if data == "":
        #                 break
        #         buffer += data
        # object = pickle.loads(buffer)
        # file.close()
        object = pickle.dump(object, open(filename, "rb"))
        return object

# @profile
def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log(np.absolute(return_vals) + 1)
    return_vals[neg_mask] *= -1.
    return return_vals

# @profile
def safe_exp(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.exp(np.clip(np.absolute(return_vals), -7, 7)) - 1
    return_vals[neg_mask] *= -1.
    return return_vals

def y_scaler(input_array):
    transformed_array = safe_log(input_array)
    scaler = MaxAbsScaler()
    #transformed_array = scaler.fit_transform(transformed_array)
    return transformed_array, scaler

def y_inverse_scaler(prediction_array):
    transformed_array = prediction_array #scaler.inverse_transform(prediction_array)
    transformed_array = safe_exp(transformed_array)
    return transformed_array

# @profile
def mle(actual_y, prediction_y):
    """
    Compute the Root Mean  Log Error

    Args:
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
    """

    return np.mean(np.absolute(safe_log(prediction_y) - safe_log(actual_y)))

# @profile
def mle_eval(actual_y, eval_y):
    """
    Used during xgboost training

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', np.mean(np.absolute(safe_log(actual_y) - safe_log(prediction_y)))

# @profile
def mae_eval(y, y0):
    y0 = y0.get_label()
    assert len(y) == len(y0)
    # return 'error', np.sqrt(np.mean(np.square(np.log(y + 1) - np.log(y0 + 1))))
    return 'error', np.mean(np.absolute(y - y0)), False

# @profile
def safe_mape(actual_y, prediction_y):
    """
    Calculate mean absolute percentage error

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    diff = np.absolute((actual_y - prediction_y) / np.clip(np.absolute(actual_y), 0.25, None))
    return 100. * np.mean(diff)


# @profile
def mape_eval(actual_y, eval_y):
    """
    Used during xgboost training

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', safe_mape(actual_y, prediction_y)

# @profile
def mape_log_y(actual_y, prediction_y):
    inverse_actual = actual_y.copy()
    inverse_actual = y_inverse_scaler(inverse_actual)

    inverse_prediction = prediction_y.copy()
    inverse_prediction = y_inverse_scaler(inverse_prediction)

    return safe_mape(inverse_actual, inverse_prediction)

# @profile
def mape_log_y_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', mape_log_y(actual_y, prediction_y)

# @profile
def huber_loss(y_true, y_pred, delta=1.8):
    """
    Variation on the huber loss function, weighting the cost of lower errors over higher errors
    """
    abs_diff = np.absolute(y_true-y_pred) + 1
    flag = abs_diff < delta
    return (flag) * delta * (abs_diff ** 2.) + (~flag) * 0.5 * (abs_diff - 0.5 * delta)

# @profile
def huber_loss_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', huber_loss(actual_y, prediction_y)

# @profile
def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            0.25,
                                            None))
    return 100. * K.mean(diff, axis=-1)

# @profile
def drop_unused_columns(df, data_cols):
    # Check for columns to drop
    print('Keeping columns:', list(data_cols))
    cols_to_drop = []
    for col in df.columns:
        if col not in data_cols:
            cols_to_drop.append(col)

    print('Dropping columns:', list(cols_to_drop))
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

# @profile
def convert_date(df, column_name):
    df[column_name + "_TIMESTAMP"] = (pd.DatetimeIndex(df[column_name]) - pd.datetime(2007, 1, 1)).total_seconds()

    df[column_name + "_YEAR"] = pd.DatetimeIndex(df[column_name]).year.astype('str')
    df[column_name + "_MONTH"] = pd.DatetimeIndex(df[column_name]).month.astype('str')
    df[column_name + "_DAY"] = pd.DatetimeIndex(df[column_name]).day.astype('str')
    df[column_name + "_DAYOFWEEK"] = pd.DatetimeIndex(df[column_name]).dayofweek.astype('str')

# @profile
def setup_data_columns(df):
    # Remove columns not referenced in either algorithm
    columns_to_keep = [LABEL_COLUMN, 'quoteDate', 'exDividendDate']
    columns_to_keep.extend(CONTINUOUS_COLUMNS)
    columns_to_keep.extend(CATEGORICAL_COLUMNS)
    return_df = drop_unused_columns(df, columns_to_keep)
    return return_df

# @profile
def load_data():
    """Load pickled data and run combined prep """
    # Return dataframe and mask to split data
    df = pd.read_pickle('data/ml-aug-data.pkl.gz', compression='gzip')
    #df = pd.read_pickle('data/ml-july-data.pkl.gz', compression='gzip')
    gc.collect()

    df = setup_data_columns(df)

    # Convert quote dates data to year and month
    df['quoteDate'] = pd.to_datetime(df['quoteDate'])
    df['exDividendDate'] = pd.to_datetime(df['exDividendDate'], errors='coerce')

    # Reset dividend date as a number
    df['exDividendRelative'] = \
        df['exDividendDate'] - \
        df['quoteDate']

    # convert string difference value to integer
    df['exDividendRelative'] = df['exDividendRelative'].apply(
        lambda x: -999 if pd.isnull(x) else x.days)

    convert_date(df, 'quoteDate')

    # df['quoteYear'], df['quoteMonth'], = \
    #     df['quoteDate'].dt.year, \
    #     df['quoteDate'].dt.month.astype('int8')

    # Remove dates columns
    df.drop(['quoteDate', 'exDividendDate'], axis=1, inplace=True)

    df = df.dropna(subset=[LABEL_COLUMN], how='all')

    # Clip to -99 to 1000 range
    df[LABEL_COLUMN] = df[LABEL_COLUMN].clip(-99, 1000)

    # Add scaled value for y - using log of y
    ############ - Double log ############################
    df[LABEL_COLUMN + '_scaled'] = safe_log(df[LABEL_COLUMN].values)

    # Fill N/A vals with dummy number
    df.fillna(-999, inplace=True)

    return df

# @profile
def prep_data():
    df = load_data()

    scaler = MinMaxScaler(feature_range=(0,1)) #StandardScaler()
    df[CONTINUOUS_COLUMNS] = scaler.fit_transform(df[CONTINUOUS_COLUMNS].as_matrix())
    return df


# @profile
def divide_data(share_data):
    # Use pandas dummy columns for categorical columns
    # share_data = pd.get_dummies(data=share_data, columns=['4WeekBollingerPrediction',
    #                                                       '4WeekBollingerType',
    #                                                       '12WeekBollingerPrediction',
    #                                                       '12WeekBollingerType'])

    # Use categorical entity embedding encoder
    ce = Categorical_encoder(strategy="dummification", verbose=False)
    df_train_transform = ce.fit_transform(share_data.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1),
                     share_data[LABEL_COLUMN + '_scaled'])

    df_train_transform['symbol'] = share_data['symbol']
    df_train_transform[LABEL_COLUMN] = share_data[LABEL_COLUMN]
    df_train_transform[LABEL_COLUMN + '_scaled'] = share_data[LABEL_COLUMN + '_scaled']

    symbol_models = {}
    symbols = df_train_transform['symbol'].unique().tolist()
    symbol_map = {}
    symbol_num = 0

    print('No of symbols:', len(symbols))

    df_all_train_x = pd.DataFrame()
    df_all_train_y = pd.DataFrame()
    df_all_train_actuals = pd.DataFrame()
    df_all_test_x = pd.DataFrame()
    df_all_test_y = pd.DataFrame()
    df_all_test_actuals = pd.DataFrame()

    symbols_train_x = {}
    symbols_train_y = {}
    symbols_train_actuals = {}
    symbols_test_x = {}
    symbols_test_y = {}
    symbols_test_actuals = {}

    # prep data for fitting into both model types
    for symbol in symbols:
        gc.collect()

        print('Symbol:', symbol, 'num:', symbol_num)

        # Update string to integer ## replaced with encoder
        # df_train_transform.loc[df_train_transform.symbol == symbol, 'symbol'] = symbol_num

        # Take copy of model data and re-set the pandas indexes
        # model_data = df_train_transform.loc[df_train_transform['symbol'] == symbol_num]
        model_data = df_train_transform.loc[df_train_transform['symbol'] == symbol]

        # Remove symbol as it has now been encoded separately
        model_data.drop(['symbol'], axis=1, inplace=True)

        msk = np.random.rand(len(model_data)) < 0.75

        # Prep dataframes and reset index for appending
        df_train = model_data[msk]
        df_test = model_data[~msk]
        df_train.reset_index()
        df_test.reset_index()

        # MAke sure a minimum number of rows are present in sample for symbol
        if (len(df_train) > 150 & len(df_test) > 50):
            symbols_train_y[symbol] = df_train[LABEL_COLUMN + '_scaled'].values
            symbols_train_actuals[symbol] = df_train[LABEL_COLUMN].values
            symbols_train_x[symbol] = df_train.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1).values

            symbols_test_actuals[symbol] = df_test[LABEL_COLUMN].values
            symbols_test_y[symbol] = df_test[LABEL_COLUMN + '_scaled'].values
            symbols_test_x[symbol] = df_test.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1).values

            df_all_train_y = pd.concat([df_all_train_y, df_train[LABEL_COLUMN + '_scaled']])
            df_all_train_actuals = pd.concat([df_all_train_actuals, df_train[LABEL_COLUMN]])
            df_all_train_x = df_all_train_x.append(df_train.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

            df_all_test_actuals = pd.concat([df_all_test_actuals, df_test[LABEL_COLUMN]])
            df_all_test_y = pd.concat([df_all_test_y, df_test[LABEL_COLUMN + '_scaled']])
            df_all_test_x = df_all_test_x.append(df_test.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

            # Set up map of symbol name to number
            symbol_map[symbol] = symbol_num

        symbol_num += 1

    print(symbol_map)

    # Clean-up the initial data variable
    return symbol_map, symbols_train_y, symbols_train_actuals, symbols_train_x, symbols_test_actuals, \
           symbols_test_y, symbols_test_x, df_all_train_y, df_all_train_actuals, df_all_train_x,\
           df_all_test_actuals, df_all_test_y, df_all_test_x

# @profile
def train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x):
    #Train general model
    # Create model
    model = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=110, base_score=0.35, colsample_bylevel=0.8,
                             colsample_bytree=0.8, gamma=0, learning_rate=0.075, max_delta_step=0,
                             min_child_weight=0)

    all_train_y = df_all_train_y.as_matrix()
    all_train_x = df_all_train_x.as_matrix()
    all_test_actuals = df_all_test_actuals.as_matrix()
    all_test_y = df_all_test_y.as_matrix()
    all_test_x = df_all_test_x.as_matrix()


    # mape_vals_train = keras_models['mape_model'].predict(all_train_x)
    # mape_vals_test = keras_models['mape_model'].predict(all_test_x)
    # mae_vals_train = keras_models['mae_model'].predict(all_train_x)
    # mae_vals_test = keras_models['mae_model'].predict(all_test_x)
    #
    # # Use keras models to generate extra outputs
    # lgbm_train_x = np.column_stack([all_train_x, mape_vals_train])
    # lgbm_train_x = np.column_stack([lgbm_train_x, mae_vals_train])
    # lgbm_test_x = np.column_stack([all_test_x, mape_vals_test])
    # lgbm_test_x = np.column_stack([lgbm_test_x, mae_vals_test])

    # lgbm_train_x = all_train_x
    # lgbm_test_x = all_test_x


    # Add lgbm predictions
    # lgbm_predictions_train = lgbm_models['log_y'].predict(lgbm_train_x)
    # lgbm_predictions_test = lgbm_models['log_y'].predict(lgbm_test_x)
    #
    # all_train_x = np.column_stack([all_train_x, lgbm_predictions_train])
    # all_test_x = np.column_stack([all_test_x, lgbm_predictions_test])


    # Use keras models to generate extra outputs
    # all_train_x = np.column_stack([all_train_x, mape_vals_train])
    # all_train_x = np.column_stack([all_train_x, mae_vals_train])
    # all_test_x = np.column_stack([all_test_x, mape_vals_test])
    # all_test_x = np.column_stack([all_test_x, mae_vals_test])

    eval_set = [(all_test_x, all_test_y)]
    model.fit(all_train_x, all_train_y, early_stopping_rounds=250, eval_metric='mae', eval_set=eval_set,
    #model.fit(all_train_x, all_train_y, early_stopping_rounds=250, eval_metric=mape_log_y_eval, eval_set=eval_set,
    #model.fit(stacked_train_x, all_train_y, early_stopping_rounds=250, eval_metric=huber_loss_eval, eval_set=eval_set,
                verbose=True)


    gc.collect()

    predictions = model.predict(all_test_x)
    #### Double exp #######
    inverse_scaled_predictions = safe_exp(predictions)

    eval_results({'general_mae': {
                        'log_y': all_test_y,
                        'actual_y': all_test_actuals,
                        'log_y_predict': predictions,
                        'y_predict': inverse_scaled_predictions
                }
    })


    range_results({
        'general_mae': inverse_scaled_predictions
        }, all_test_actuals)


    return model

# @profile
def train_symbol_models(symbol_map, symbols_train_y, symbols_train_x, symbols_test_actuals, symbols_test_y,
                        symbols_test_x, gen_model, lgbm_models, keras_models):

    # Create and execute models
    symbol_models = {}
    all_results = pd.DataFrame()
    results_output = pd.DataFrame()

    ## Run the predictions across using each symbol model and the genral model
    for symbol in symbol_map:
        train_y = symbols_train_y[symbol]
        train_x = symbols_train_x[symbol]
        test_actuals = symbols_test_actuals[symbol]
        test_y = symbols_test_y[symbol]
        test_x = symbols_test_x[symbol]

        lgbm_predictions_train = lgbm_models['log_y'].predict(train_x)
        lgbm_predictions_test = lgbm_models['log_y'].predict(test_x)

        # gen_predictions_train = gen_model.predict(np.column_stack([train_x, lgbm_predictions_train]))
        # gen_predictions_test = gen_model.predict(np.column_stack([test_x, lgbm_predictions_test]))
        #
        # stacked_train_x = np.column_stack([np.column_stack([train_x, lgbm_predictions_train]),gen_predictions_train])
        # stacked_test_x = np.column_stack([np.column_stack([test_x, lgbm_predictions_test]), gen_predictions_test])

        stacked_train_x = train_x
        stacked_test_x = test_x

        # Create model
        symbol_model = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=110, base_score=0.35,
                                        colsample_bylevel=0.8, colsample_bytree = 0.8, gamma = 0, learning_rate = 0.01,
                                        max_delta_step = 0, min_child_weight = 0)

        eval_set = [(stacked_test_x, test_y)]
        symbol_model.fit(stacked_train_x, train_y, early_stopping_rounds=250, eval_metric='mae', eval_set=eval_set,
        #symbol_model.fit(train_x, train_y, early_stopping_rounds=250, eval_metric=mape_log_y_eval, eval_set=eval_set,
                         verbose=False)

        gc.collect()

        # Create model
        symbol_model_log = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=110, base_score=0.35,
                                            colsample_bylevel=0.8, colsample_bytree = 0.8, gamma = 0, learning_rate = 0.01,
                                            max_delta_step = 0, min_child_weight = 0)

        eval_set = [(stacked_test_x, safe_log(test_y))]
        symbol_model_log.fit(stacked_train_x, safe_log(train_y), early_stopping_rounds=250, eval_metric='mae', eval_set=eval_set,
        #symbol_model.fit(train_x, train_y, early_stopping_rounds=250, eval_metric=mape_log_y_eval, eval_set=eval_set,
                         verbose=False)

        # Add model to models dictionary
        symbol_models[symbol + 'log_y'] = symbol_model
        symbol_models[symbol + 'log_log_y'] = symbol_model_log

        # Run predictions and prepare results
        predictions = symbol_model.predict(stacked_test_x)
        inverse_scaled_predictions = safe_exp(predictions)

        # Run predictions and prepare results
        log_predictions = symbol_model_log.predict(stacked_test_x)
        log_inverse_scaled_predictions = safe_exp(safe_exp(predictions))


        gen_predictions = gen_model.predict(test_x)
        #### Double exp #######
        gen_inverse_scaled_predictions = safe_exp(gen_predictions)

        lgbm_predictions = lgbm_models['log_y'].predict(test_x)
        lgbm_inverse_scaled_predictions = safe_exp(lgbm_predictions)

        log_lgbm_predictions = lgbm_models['log_log_y'].predict(test_x)
        log_lgbm_inverse_scaled_predictions = safe_exp(safe_exp(lgbm_predictions))

        mape_keras_predictions = keras_models['mape_model'].predict(test_x)

        mae_keras_predictions = keras_models['mae_model'].predict(test_x)
        mae_keras_inverse_scaled_predictions = safe_exp(mae_keras_predictions)



        #Evaluate results
        print('Results for', symbol)


        result_eval = eval_results({
            'symbol_results': {
            'log_y': test_y,
            'actual_y': test_actuals,
            'log_y_predict': predictions,
            'y_predict': inverse_scaled_predictions
            }
        })


        range_results({
            'symbol_': inverse_scaled_predictions,
            'symobl_log': log_inverse_scaled_predictions,
            'general': gen_inverse_scaled_predictions,
            'lgbm': lgbm_inverse_scaled_predictions,
            'lgbm_log':log_lgbm_inverse_scaled_predictions,
            'keras_mae': mae_keras_inverse_scaled_predictions,
            'keras_mape': mape_keras_predictions
         }, test_actuals)

        # Make bagged predictions - for most weight the symbol prediction
        #    if actual for any of the values is >= 0 and <= 2
        #      - average: xgboost log of log of y & lgbm log of log of y
        #    if actual for any of the values is ( >= -5 and < 0) or ( > 2 and <= 5)
        #      - average: xgboost log of y & lgbm log of log of y
        #    Others
        #      - xgboost log of y

        # bagged_predictions = predictions
        #
        # # values in the 0 to 2 range (allow for error to be -0.5 to 2.5
        # mask_lgbm = ((log_inverse_scaled_predictions >= -0.5) & (log_inverse_scaled_predictions <= 2.5))
        # mask_lgbm_log = ((log_lgbm_inverse_scaled_predictions >= -0.5) & (log_lgbm_inverse_scaled_predictions <= 2.5))
        # mask_gen = ((gen_inverse_scaled_predictions >= -0.5) & (gen_inverse_scaled_predictions <= 2.5))
        # mask_symbol = ((inverse_scaled_predictions >= -0.5) & (inverse_scaled_predictions <= 2.5))
        # mask_symbol_log = ((log_inverse_scaled_predictions >= -0.5) & (log_inverse_scaled_predictions <= 2.5))
        #
        # combined_mask = ((mask_lgbm) | (mask_lgbm_log) | (mask_gen) | (mask_symbol) | (mask_symbol_log))
        #
        # bagged_predictions[combined_mask] = (predictions[combined_mask] +
        #                                      safe_exp(log_lgbm_predictions[combined_mask])) / 2
        #
        # # values in the -5 to 5 range -- (allow for error to be -5.5 to 5.5
        # mask_lgbm = ((log_inverse_scaled_predictions >= -5.5) & (log_inverse_scaled_predictions <= 5.5))
        # mask_lgbm_log = ((log_lgbm_inverse_scaled_predictions >= -5.5) & (log_lgbm_inverse_scaled_predictions <= 5.5))
        # mask_gen = ((gen_inverse_scaled_predictions >= -5.5) & (gen_inverse_scaled_predictions <= 5.5))
        # mask_symbol = ((inverse_scaled_predictions >= -5.5) & (inverse_scaled_predictions <= 5.5))
        # mask_symbol_log = ((log_inverse_scaled_predictions >= -5.5) & (log_inverse_scaled_predictions <= 5.5))
        #
        # combined_mask = ((mask_lgbm) | (mask_lgbm_log) | (mask_gen) | (mask_symbol) | (mask_symbol_log))
        #
        # bagged_predictions[combined_mask] = (safe_exp(log_predictions[combined_mask]) +
        #                                      safe_exp(log_lgbm_predictions[combined_mask])) / 2
        #
        # bagged_inverse_scaled_predictions = safe_exp(bagged_predictions)
        #
        # bagged_results = eval_results({
        #     'bagged_results': {
        #             'log_y': test_y,
        #             'actual_y': test_actuals,
        #             'log_y_predict': ((gen_predictions + predictions +
        #                            lgbm_predictions) / 3),
        #             'y_predict': ((gen_inverse_scaled_predictions + inverse_scaled_predictions +
        #                            lgbm_inverse_scaled_predictions) / 3)
        #         }
        # })
        #
        #
        # all_results = all_results.append(pd.DataFrame.from_dict({'actuals': test_actuals,
        #                                                         'gen_predictions': gen_inverse_scaled_predictions,
        #                                                         'lgbm_predictions': lgbm_inverse_scaled_predictions,
        #                                                         'symbol_predictions': inverse_scaled_predictions,
        #                                                         'bagged_predictions': bagged_inverse_scaled_predictions,
        #                                                         }))
        #
        # results_output = results_output.append(pd.DataFrame.from_dict({'symbol': [symbol],
        #                                                                'general_mle': [general_results.err],
        #                                                                'lgbm_mle': [lgbm_results.err],
        #                                                                'symbol_mle': [symbol_results.err],
        #                                                                'fifty_fifty_mle': [fifty_results.err],
        #                                                                'sixty_thirty_mle': [sixty_results.err],
        #                                                                'thirds_mle': [thirds_results.err],
        #                                                                'bagged_mle': [bagged_results.err],
        #                                                                'general_mae': [general_results.mae],
        #                                                                'lgbm_mae': [lgbm_results.mae],
        #                                                                'symbol_mae': [symbol_results.mae],
        #                                                                'fifty_fifty_mae': [fifty_results.mae],
        #                                                                'sixty_thirty_mae': [sixty_results.mae],
        #                                                                'thirds_mae': [thirds_results.mae],
        #                                                                'bagged_mae': [bagged_results.mae],
        #                                                                'general_mape': [general_results.mape],
        #                                                                'lgbm_mape': [lgbm_results.mape],
        #                                                                'symbol_mape': [symbol_results.mape],
        #                                                                'fifty_fifty_mape': [fifty_results.mape],
        #                                                                'sixty_thirty_mape': [sixty_results.mape],
        #                                                                'thirds_mape': [thirds_results.mape],
        #                                                                'bagged_mape': [bagged_results.mape],
        #                                                                'general_r2': [general_results.r2],
        #                                                                'lgbm_r2': [lgbm_results.r2],
        #                                                                'symbol_r2': [symbol_results.r2],
        #                                                                'fifty_fifty_r2': [fifty_results.r2],
        #                                                                'sixty_thirty_r2': [sixty_results.r2],
        #                                                                'thirds_r2': [thirds_results.r2],
        #                                                                'bagged_r2': [bagged_results.r2]
        #                                                                }))


    return symbol_models, all_results, results_output

# @profile
def train_lgbm(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x):
    params = {
        'objective': 'regression',
        #'objective': 'huber',
        #'objective': 'regression_l1',
        #'metric': 'huber',
        'metric': 'mae',
        'num_leaves': 16384,
        'max_bin': 250000,
        'boosting_type': 'dart'#,
        #'learning_rate': 0.05
    }

    gbms = {}

    train_x = df_all_train_x.as_matrix()
    train_y = df_all_train_y[0].values
    train_log_y = safe_log(train_y)
    test_x = df_all_test_x.as_matrix()
    test_actuals = df_all_test_actuals.as_matrix()
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)

    # mape_vals_train = keras_models['mape_model'].predict(train_x)
    # mape_vals_test = keras_models['mape_model'].predict(test_x)
    # mae_vals_train = keras_models['mae_model'].predict(train_x)
    # mae_vals_test = keras_models['mae_model'].predict(test_x)
    #
    # # Use keras models to generate extra outputs
    # train_x = np.column_stack([train_x, mape_vals_train])
    # train_x = np.column_stack([train_x, mae_vals_train])
    # test_x = np.column_stack([test_x, mape_vals_test])
    # test_x = np.column_stack([test_x, mae_vals_test])

    # Set-up lightgbm
    train_set = lgb.Dataset(train_x, label=train_y)
    eval_set = lgb.Dataset(test_x, reference=train_set, label=test_y)


    # feature_name and categorical_feature
    gbms['log_y'] = lgb.train(params,
                    train_set,
                    valid_sets=eval_set,
                    # feval=mae_eval,
                    learning_rates=lambda iter: 0.5 * (0.9995 ** iter),
                    num_boost_round=2000,
                    early_stopping_rounds=50)

    gc.collect()

    iteration_number = 2000

    if gbms['log_y'].best_iteration:
        iteration_number = gbms['log_y'].best_iteration


    # Make predictions
    log_predictions = gbms['log_y'].predict(test_x, num_iteration=iteration_number)
    log_inverse_scaled_predictions = safe_exp(log_predictions)

    eval_results({'lgbm_log_y': {
                        'log_y': test_y,
                        'actual_y': test_actuals,
                        'log_y_predict': log_predictions,
                        'y_predict': log_inverse_scaled_predictions
                }
    })

    gc.collect()

    # Set-up lightgbm
    train_set = lgb.Dataset(train_x, label=train_log_y)
    eval_set = lgb.Dataset(test_x, reference=train_set, label=test_log_y)


    # feature_name and categorical_feature
    gbms['log_log_y'] = lgb.train(params,
                    train_set,
                    valid_sets=eval_set,
                    # feval=mae_eval,
                    learning_rates=lambda iter: 0.5 * (0.9995 ** iter),
                    num_boost_round=2000,
                    early_stopping_rounds=50)

    gc.collect()

    iteration_number = 2000

    if gbms['log_log_y'].best_iteration:
        iteration_number = gbms['log_log_y'].best_iteration


    # Make predictions
    log_log_predictions = gbms['log_log_y'].predict(test_x, num_iteration=iteration_number)
    predictions_log_y = safe_exp(log_log_predictions)
    log_log_inverse_scaled_predictions = safe_exp(predictions_log_y)


    eval_results({'lgbm_log_log_y': {
                        'log_y': test_y,
                        'actual_y': test_actuals,
                        'log_y_predict': predictions_log_y,
                        'y_predict': log_log_inverse_scaled_predictions
                }
    })


    range_results({
        'lgbm_log_y':log_inverse_scaled_predictions,
        'lgbm_log_log_y': log_log_inverse_scaled_predictions
        }, test_actuals)


    return gbms


def train_sklearn_models(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x):
    sklearn_models = {}

    estimators = {
        # 'gradient_boosted': (GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=100,
        #                                                          max_depth=30, criterion='mae', verbose=2)),
        'random_forest': (RandomForestRegressor(n_estimators=10, criterion='mae', n_jobs=-1, verbose=2)),
        'extra_trees': (ExtraTreesRegressor(n_estimators=15, criterion='mae', n_jobs=-1, verbose=2))
    }


    train_y = df_all_train_y[0].values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.as_matrix()
    test_actuals = df_all_test_actuals.as_matrix()
    test_y = df_all_test_y[0].values
    test_x = df_all_test_x.as_matrix()

    for estimator in estimators:
        print('Fitting',  estimator,  'regressor...')

        model_ref = estimator + '_log_y'
        log_model_ref = estimator + '_log_log_y'

        sklearn_models[model_ref] = estimators[estimator].fit(train_x, train_y)
        gc.collect()

        print('Executing', estimator, 'regressor predictions...')

        # Make predictions
        log_predictions = sklearn_models[model_ref].predict(test_x)
        log_inverse_scaled_predictions = safe_exp(log_predictions)


        eval_results({model_ref: {
                            'log_y': test_y,
                            'actual_y': test_actuals,
                            'log_y_predict': log_predictions,
                            'y_predict': log_inverse_scaled_predictions
                    }
        })

        # # Log log y
        # sklearn_models[log_model_ref] = estimators[estimator].fit(train_x, train_y)
        #
        # gc.collect()
        #
        # sklearn_models[log_model_ref].fit(train_x, train_log_y)
        # gc.collect()
        #
        # # Make predictions
        # log_log_predictions = sklearn_models[log_model_ref].predict(test_x)
        # predictions_log_y = safe_exp(log_log_predictions)
        # log_log_inverse_scaled_predictions = safe_exp(predictions_log_y)
        #
        # eval_results({sklearn_models: {
        #                     'log_y': test_y,
        #                     'actual_y': test_actuals,
        #                     'log_y_predict': predictions_log_y,
        #                     'y_predict': log_log_inverse_scaled_predictions
        #             }
        # })


        range_results({
            model_ref:log_inverse_scaled_predictions #,
            # log_model_ref: log_log_inverse_scaled_predictions
            }, test_actuals)

    return sklearn_models

# @profile
def train_keras_nn(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals, df_all_test_y,
                   df_all_test_x):
    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.as_matrix()
    test_actuals = df_all_test_actuals.as_matrix()
    test_y = df_all_test_y[0].values
    test_x = df_all_test_x.as_matrix()

    print('Scaling y values')
    actuals_scaler = MinMaxScaler(feature_range=(0,1)) #StandardScaler()
    actuals_scaler.fit(train_actuals)
    scaled_train_actuals = actuals_scaler.transform(train_actuals)
    scaled_test_actuals = actuals_scaler.transform(test_actuals)

    log_scaler = MinMaxScaler(feature_range=(0,1)) #StandardScaler()
    log_scaler.fit(train_y)
    scaled_train_y = log_scaler.transform(train_y)
    scaled_test_y = log_scaler.transform(test_y)



    print('Building Keras mape model...')

    n_layer1 = int(train_x.shape[1])
    n_layer2 = int(n_layer1 / 2 + 2)

    p_model = Sequential()
    p_model.add(Dense(n_layer1, input_shape=(train_x.shape[1],)))
    p_model.add(Activation('selu'))
    p_model.add(Dropout(0.05))
    p_model.add(Dense(n_layer2))
    # p_model.add(BatchNormalization())
    p_model.add(Activation('selu'))
    p_model.add(Dropout(0.05))
    p_model.add(Dense(12, name="mape_twelve"))
    p_model.add(Activation('selu'))
    p_model.add(Dense(1))
    p_model.add(Activation('linear'))

    p_model.compile(optimizer='adam', loss=sc_mean_absolute_percentage_error, metrics = ['mae'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=15)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    csv_logger = CSVLogger('./logs/scaled-mape-training.log')

    print('Fitting Keras mape scaled actuals model...')

    history = p_model.fit(train_x,
                          scaled_train_actuals,
                          validation_data=(test_x, scaled_test_actuals),
                          epochs=20000,
                          batch_size=512,
                          callbacks=[reduce_lr, early_stopping, csv_logger],
                          verbose=0)

    gc.collect()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Keras mape scaled model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.ion()
    # plt.show()
    #
    # gc.collect()

    predictions = actuals_scaler.inverse_transform(p_model.predict(test_x))

    eval_results({'keras_mape_y': {
                        'actual_y': test_actuals,
                        'y_predict': predictions
                }
    })

    range_results({
        'Keras_mape_scaled_nn': predictions
    }, test_actuals)

    print('Fitting Keras mape model...')

    csv_logger = CSVLogger('./logs/actual-mape-training.log')

    history = p_model.fit(train_x,
                          train_actuals,
                          validation_data=(test_x, test_actuals),
                          epochs=20000,
                          batch_size=512,
                          callbacks=[reduce_lr, early_stopping, csv_logger],
                          verbose=0)

    gc.collect()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Keras mape model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.ion()
    # plt.show()
    #
    # gc.collect()

    predictions = p_model.predict(test_x)

    eval_results({'keras_mape_y': {
                        'actual_y': test_actuals,
                        'y_predict': predictions
                }
    })

    range_results({
        'Keras_mape_nn': predictions
    }, test_actuals)


    model = Sequential()
    model.add(Dense(n_layer1, input_shape=(train_x.shape[1],)))
    # model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.15))
    model.add(Dense(n_layer2))
    # model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.15))
    model.add(Dense(n_layer1))
    # model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.15))
    model.add(Dense(n_layer2))
    # model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.15))
    model.add(Dense(12, name="mae_twelve"))
    model.add(Activation('selu'))
    # model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('linear'))


    model.compile(optimizer='adam', loss='mae') #, metrics = ['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    csv_logger = CSVLogger('./logs/scaled-log-training.log')


    print('Fitting Keras scaled log y model...')


    history = model.fit(train_x,
                        scaled_train_y,
                        validation_data=(test_x, scaled_test_y),
                        epochs=20000,
                        batch_size=512,
                        callbacks=[reduce_lr, early_stopping, csv_logger],
                        verbose=0)

    gc.collect()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Keras mae scaled y model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.ion()
    # plt.show()
    #
    # gc.collect()


    print('Executing keras predictions...')


    log_predictions = log_scaler.inverse_transform(model.predict(test_x))
    exp_predictions = safe_exp(log_predictions)

    eval_results({'keras_log_y': {
                        'log_y': test_y,
                        'actual_y': test_actuals,
                        'log_y_predict': log_predictions,
                        'y_predict': exp_predictions
                }
    })

    print('Fitting Keras log y model...')

    csv_logger = CSVLogger('./logs/log-training.log')

    history = model.fit(train_x,
                        train_y,
                        validation_data=(test_x, test_y),
                        epochs=20000,
                        batch_size=512,
                        callbacks=[reduce_lr, early_stopping, csv_logger],
                        verbose=0)

    gc.collect()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Keras mae model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.ion()
    # plt.show()
    #
    # gc.collect()


    print('Executing keras predictions...')

    log_predictions = model.predict(test_x)
    exp_predictions = safe_exp(log_predictions)

    eval_results({'keras_log_y': {
                        'log_y': test_y,
                        'actual_y': test_actuals,
                        'log_y_predict': log_predictions,
                        'y_predict': exp_predictions
                }
    })


    # Construct models which output final twelve weights as predictions
    # mape_intermediate_model = Model(inputs=p_model.input,
    #                                  outputs=p_model.get_layer('mape_twelve').output)
    #
    # mae_intermediate_model = Model(inputs=model.input,
    #                                  outputs=model.get_layer('mae_twelve').output)

    return {
        'mape_model': p_model,
        'mae_model': model
        # 'mape_intermediate_model': mape_intermediate_model,
        # 'mae_intermediate_model': mae_intermediate_model
        }

def train_keras_linear(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals, df_all_test_y,
                   df_all_test_x):
    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.as_matrix()
    test_actuals = df_all_test_actuals.as_matrix()
    test_y = df_all_test_y[0].values
    test_x = df_all_test_x.as_matrix()

    # reshape arrays
    train_y = train_y.reshape(train_y.shape[0], 1)
    test_y = test_y.reshape(test_y.shape[0], 1)
    train_actuals = train_actuals.reshape(train_actuals.shape[0], 1)
    test_actuals = test_actuals.reshape(test_actuals.shape[0], 1)

    print('Scaling y values')
    actuals_scaler = MinMaxScaler(feature_range=(0,1)) #StandardScaler()
    actuals_scaler.fit(np.row_stack([train_actuals, test_actuals]))
    scaled_train_actuals = actuals_scaler.transform(train_actuals)
    scaled_test_actuals = actuals_scaler.transform(test_actuals)

    log_scaler = MinMaxScaler(feature_range=(0,1)) #StandardScaler()
    log_scaler.fit(np.row_stack([train_y, test_y]))
    scaled_train_y = log_scaler.transform(train_y)
    scaled_test_y = log_scaler.transform(test_y)



    print('Building Keras linear model...')


    linear_model = Sequential()
    linear_model.add(Dense(train_x.shape[1], input_shape=(train_x.shape[1],)))
    linear_model.add(Dense(1))
    linear_model.add(Activation('linear'))

    linear_model.compile(optimizer='adam', loss='mae', metrics = ['mae'])

    print('Fitting Keras linear model to y...')

    history = linear_model.fit(train_x,
                               train_actuals,
                               validation_data=(test_x, test_actuals),
                               epochs=50,
                               batch_size=512,
                               verbose=1)

    gc.collect()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Keras linear model loss y')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.ion()
    # plt.show()
    #
    # gc.collect()

    actuals_predictions = linear_model.predict(test_x)

    eval_results({'keras_linear_actuals': {
                        'actual_y': test_actuals,
                        'y_predict': actuals_predictions
                }
    })

    print('Fitting Keras linear model to log of y...')

    history = linear_model.fit(train_x,
                               train_y,
                               validation_data=(test_x, test_y),
                               epochs=50,
                               batch_size=512,
                               verbose=1)

    gc.collect()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Keras linear model loss log of y')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.ion()
    # plt.show()
    #
    # gc.collect()

    log_predictions = linear_model.predict(test_x)
    scaled_log_predictions = safe_exp(log_predictions)

    eval_results({'keras_linear_log_y': {
        'actual_y': test_actuals,
        'y_predict': scaled_log_predictions
    }
    })

    print('Fitting Keras linear model to scaled y...')

    history = linear_model.fit(train_x,
                               scaled_train_actuals,
                               validation_data=(test_x, scaled_test_actuals),
                               epochs=50,
                               batch_size=512,
                               verbose=1)

    gc.collect()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Keras linear model loss scaled y')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.ion()
    # plt.show()
    #
    # gc.collect()

    scaled_predictions = linear_model.predict(test_x)
    transformed_scaled_predictions = actuals_scaler.inverse_transform(scaled_predictions)

    eval_results({'keras_linear_scaled_y': {
        'actual_y': test_actuals,
        'y_predict': transformed_scaled_predictions
    }
    })

    print('Fitting Keras linear model to scaled log y...')

    history = linear_model.fit(train_x,
                               scaled_train_y,
                               validation_data=(test_x, scaled_test_y),
                               epochs=50,
                               batch_size=512,
                               verbose=1)

    gc.collect()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Keras linear model loss scaled log of y')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.ion()
    # plt.show()
    #
    # gc.collect()

    scaled_log_predictions = linear_model.predict(test_x)
    transformed_scaled_log_predictions = safe_exp(log_scaler.inverse_transform(scaled_log_predictions))

    eval_results({'keras_linear_scaled_log_y': {
        'actual_y': test_actuals,
        'y_predict': transformed_scaled_log_predictions
    }
    })

    range_results({
        'keras_linear_y': actuals_predictions,
        'keras_linear_log_y': scaled_log_predictions,
        'keras_linear_scaled_y': transformed_scaled_predictions,
        'keras_linear_scaled_log_y': transformed_scaled_log_predictions
    }, test_actuals)


# @profile
def bagging(df_all_test_x, df_all_test_actuals, gen_model, lgbm_models, keras_models):
    test_actuals = df_all_test_actuals.as_matrix()
    test_x = df_all_test_x.as_matrix()

    print('Running model predictions')
    gen = gen_model.predict(test_x)
    gen = safe_exp(gen)
    lgbm = lgbm_models['log_y'].predict(test_x)
    lgbm = safe_exp(lgbm)
    log_lgbm = lgbm_models['log_log_y'].predict(test_x)
    log_lgbm = safe_exp(safe_exp(log_lgbm))
    keras_mae = keras_models['mae_model'].predict(test_x)
    keras_mae = safe_exp(keras_mae)
    keras_mape = keras_models['mape_model'].predict(test_x)

    # Reshape arrays
    gen = gen.reshape(gen.shape[0], 1)
    lgbm = lgbm.reshape(lgbm.shape[0], 1)
    log_lgbm = log_lgbm.reshape(log_lgbm.shape[0], 1)

    print('Bagging predictions')

    # Set default value to mape
    bagged_predictions = np.copy(keras_mape)

    # <-22.5 use keras mae - set default
    mask_0 = ((keras_mae <= -10))
    bagged_predictions[mask_0] = keras_mae[mask_0]

    #### - Change to keras ######
    # >-25 and <-10 average keras mae & gen xgboost
    # mask_1 = ((keras_mae > -25) & (keras_mae < -10))
    # bagged_predictions[mask_1] = (keras_mae[mask_1] + gen[mask_1]) / 2

    # >-11 and <-5 xgboost gen
    mask_2 = ((gen > -10) & (gen < -5.5))
    bagged_predictions[mask_2] = gen[mask_2]

    # > 2 and < 6 average gen and lgbm
    mask_3 = ((gen > 2.5) & (gen <= 5.5))
    bagged_predictions[mask_3] = (lgbm[mask_3] + gen[mask_3]) / 2

    # > -6 and < 2.5 keras mape & log lgbm
    mask_4 = ((log_lgbm > -6) & (log_lgbm <= 2.5))
    bagged_predictions[mask_4] = (keras_mape[mask_4] + log_lgbm[mask_4]) / 2

    # > 5.5 and < 21 gen xgboost
    mask_5 = ((gen > 5.5) & (gen <= 21))
    bagged_predictions[mask_5] = gen[mask_5]

    # > 20 and < 55 average gen and keras mae
    mask_6 = ((keras_mae > 21) & (keras_mae < 55))
    bagged_predictions[mask_6] = (keras_mae[mask_6] + gen[mask_6]) / 2

    # >= 55 and < 110 keras mae
    mask_7 = ((keras_mae >= 55) & (keras_mae < 110))
    bagged_predictions[mask_7] = keras_mae[mask_7]

    # >= 110 gen
    mask_8 = ((gen >= 110))
    bagged_predictions[mask_8] = gen[mask_8]



    eval_results({'bagged_predictions': {
                        'actual_y': test_actuals,
                        'y_predict': bagged_predictions
                }
    })

    range_results({
        'gen': gen,
        'lgbm': lgbm,
        'log_lgbm': log_lgbm,
        'keras_mae': keras_mae,
        'keras_mape': keras_mape,
        'bagged_predictions': bagged_predictions
    }, test_actuals)


if __name__ == "__main__":

    # Prepare run_str
    run_str = datetime.datetime.now().strftime('%Y%m%d%H%M')

    log = open("logs/execution-" + run_str + ".log", "a")
    sys.stdout = log

    # Retrieve and run combined prep on data
    share_data  = prep_data()
    gc.collect()

    # Divide data into symbol sand general data for training an testing
    symbol_map, symbols_train_y, symbols_train_actuals, symbols_train_x, symbols_test_actuals, \
    symbols_test_y, symbols_test_x, df_all_train_y, df_all_train_actuals, df_all_train_x, \
    df_all_test_actuals, df_all_test_y, df_all_test_x = divide_data(share_data)

    del share_data
    gc.collect()

    # train_keras_linear(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals,
    #                    df_all_test_y, df_all_test_x)

    lgbm_models = train_lgbm(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x)

    # Train the general model
    gen_model = train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x)

    keras_models = train_keras_nn(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals,
                                  df_all_test_y, df_all_test_x)

    bagging(df_all_test_x, df_all_test_actuals, gen_model, lgbm_models, keras_models)

    quit()

    # Remove combined data, only required for general model
    del df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x
    gc.collect()

    symbol_models, all_results, results_output = train_symbol_models(symbol_map, symbols_train_y, symbols_train_x,
                                                                     symbols_test_actuals, symbols_test_y,
                                                                     symbols_test_x, gen_model, lgbm_models,
                                                                     keras_models)

    gc.collect()

    # Save results as csv
    results_output.to_csv('xgboost-models/results-' + run_str + '.csv', columns = ['symbol',
                                                                                   'general_mle',
                                                                                   'lgbm_mle',
                                                                                   'symbol_mle',
                                                                                   'fifty_fifty_mle',
                                                                                   'sixty_thirty_mle',
                                                                                   'thirds_mle',
                                                                                   'bagged_mle',
                                                                                   'general_mae',
                                                                                   'lgbm_mae',
                                                                                   'symbol_mae',
                                                                                   'fifty_fifty_mae',
                                                                                   'sixty_thirty_mae',
                                                                                   'thirds_mae',
                                                                                   'bagged_mae',
                                                                                   'general_mape',
                                                                                   'lgbm_mape',
                                                                                   'symbol_mape',
                                                                                   'fifty_fifty_mape',
                                                                                   'sixty_thirty_mape',
                                                                                   'thirds_mape',
                                                                                   'bagged_mape',
                                                                                   'general_r2',
                                                                                   'lgbm_r2',
                                                                                   'symbol_r2',
                                                                                   'fifty_fifty_r2',
                                                                                   'sixty_thirty_r2',
                                                                                   'thirds_r2',
                                                                                   'bagged_r2'])


    # Save models to file
    #save(symbol_map, 'xgboost-models/symbol-map.dat.gz')
    #save(gen_model, 'xgboost-models/general.dat.gz')
    #save(symbol_models, 'xgboost-models/symbols.dat.gz')


    # Generate final and combined results
    gen_err = mean_absolute_error(all_results['actuals'].values, all_results['gen_predictions'].values)
    symbol_err = mean_absolute_error(all_results['actuals'].values, all_results['symbol_predictions'].values)

    gen_mape = safe_mape(all_results['actuals'].values, all_results['gen_predictions'].values)
    symbol_mape = safe_mape(all_results['actuals'].values, all_results['symbol_predictions'].values)

    gen_r2 = r2_score(all_results['actuals'].values, all_results['gen_predictions'].values)
    symbol_r2 = r2_score(all_results['actuals'].values, all_results['symbol_predictions'].values)

    fifty_err = mean_absolute_error(all_results['actuals'].values, ((all_results['gen_predictions'].values + all_results['symbol_predictions'].values) / 2))
    fifty_mape = safe_mape(all_results['actuals'].values, ((all_results['gen_predictions'].values + all_results['symbol_predictions'].values) / 2))
    fifty_r2 = r2_score(all_results['actuals'].values, ((all_results['gen_predictions'].values + all_results['symbol_predictions'].values) / 2))

    sixty_err = mean_absolute_error(all_results['actuals'].values, ((all_results['gen_predictions'].values + (all_results['symbol_predictions'].values * 2)) / 3))
    sixty_mape = safe_mape(all_results['actuals'].values, ((all_results['gen_predictions'].values + (all_results['symbol_predictions'].values * 2)) / 3))
    sixty_r2 = r2_score(all_results['actuals'].values, ((all_results['gen_predictions'].values + (all_results['symbol_predictions'].values * 2)) / 3))

    thirds_err = mean_absolute_error(all_results['actuals'].values, ((all_results['gen_predictions'].values +
                                                                      all_results['symbol_predictions'].values +
                                                                      all_results['lgbm_predictions'].values) / 3))
    thirds_mape = safe_mape(all_results['actuals'].values, ((all_results['gen_predictions'].values +
                                                             all_results['symbol_predictions'].values +
                                                             all_results['lgbm_predictions'].values) / 3))
    thirds_r2 = r2_score(all_results['actuals'].values, ((all_results['gen_predictions'].values +
                                                          all_results['symbol_predictions'].values +
                                                          all_results['lgbm_predictions'].values) / 3))

    bagged_err = mean_absolute_error(all_results['actuals'].values, all_results['bagged_predictions'].values)
    bagged_mape = safe_mape(all_results['actuals'].values, all_results['bagged_predictions'].values)
    bagged_r2 = r2_score(all_results['actuals'].values, all_results['bagged_predictions'].values)



    # Print results
    print('Overall results')
    print('-------------------')
    print('Mean absolute error')
    print('     gen:', gen_err)
    print('  symbol:', symbol_err)
    print('   50/50:', fifty_err)
    print('   66/33:', sixty_err)
    print('  thirds:', thirds_err)
    print('  bagged:', bagged_err)

    print('Mean absolute percentage error')
    print('     gen:', gen_mape)
    print('  symbol:', symbol_mape)
    print('   50/50:', fifty_mape)
    print('   66/33:', sixty_mape)
    print('  thirds:', thirds_mape)
    print('  bagged:', bagged_mape)

    print('r2')
    print('     gen:', gen_r2)
    print('  symbol:', symbol_r2)
    print('   50/50:', fifty_r2)
    print('   66/33:', sixty_r2)
    print('  thirds:', thirds_r2)
    print('  bagged:', bagged_r2)

    overall_results_output = pd.DataFrame()


    result_ranges = [-50, -25, -10, -5, 0, 2, 5, 10, 20, 50, 100, 1001]
    lower_range = -100

    for upper_range in result_ranges:
        range_results = all_results.loc[(all_results['actuals'] >= lower_range) &
                                        (all_results['actuals'] < upper_range)]
        # Generate final and combined results
        range_actuals = range_results['actuals'].values
        range_gen_predictions = range_results['gen_predictions'].values
        range_lgbm_predictions = range_results['lgbm_predictions'].values
        range_symbol_predictions = range_results['symbol_predictions'].values
        range_bagged_predictions = range_results['bagged_predictions'].values

        gen_mae = mean_absolute_error(range_actuals, range_gen_predictions)
        lgbm_mae = mean_absolute_error(range_actuals, range_lgbm_predictions)
        symbol_mae = mean_absolute_error(range_actuals, range_symbol_predictions)
        bagged_mae = mean_absolute_error(range_actuals, range_bagged_predictions)
        gen_mape = safe_mape(range_actuals, range_gen_predictions)
        lgbm_mape = safe_mape(range_actuals, range_lgbm_predictions)
        symbol_mape = safe_mape(range_actuals, range_symbol_predictions)
        bagged_mape = safe_mape(range_actuals, range_bagged_predictions)

        fifty_mae = mean_absolute_error(range_actuals, ((range_gen_predictions + range_symbol_predictions) / 2))
        fifty_mape = safe_mape(range_actuals, (range_gen_predictions + range_symbol_predictions) / 2)

        sixty_mae = mean_absolute_error(range_actuals, ((range_gen_predictions + (range_symbol_predictions * 2)) / 3))
        sixty_mape = safe_mape(range_actuals, ((range_gen_predictions + (range_symbol_predictions * 2)) / 3))

        thirds_mae = mean_absolute_error(range_actuals, ((range_gen_predictions + range_symbol_predictions +
                                                          range_lgbm_predictions) / 3))
        thirds_mape = safe_mape(range_actuals, (range_gen_predictions + range_symbol_predictions +
                                                range_lgbm_predictions) / 3)


        # Print results
        print('Results:', lower_range, 'to', upper_range)
        print('-------------------')
        print('Mean absolute error')
        print('     gen:', gen_mae)
        print('    lgbm:', lgbm_mae)
        print('  symbol:', symbol_mae)
        print('   50/50:', fifty_mae)
        print('   66/33:', sixty_mae)
        print('  thirds:', thirds_mae)
        print('  bagged:', bagged_mae)

        print('Mean absolute percentage error')
        print('     gen:', gen_mape)
        print('    lgbm:', lgbm_mape)
        print('  symbol:', symbol_mape)
        print('   50/50:', fifty_mape)
        print('   66/33:', sixty_mape)
        print('  thirds:', thirds_mape)
        print('  bagged:', bagged_mape)



        overall_results_output = overall_results_output.append(pd.DataFrame.from_dict(
            {'lower_range': [lower_range],
             'upper_range': [upper_range],
             'gen_mae': [gen_mae],
             'lgbm_mae': [lgbm_mae],
             'symbol_mae': [symbol_mae],
             'fifty_mae': [fifty_mae],
             'sixty_mae': [sixty_mae],
             'thirds_mae': [thirds_mae],
             'bagged_mae': [bagged_mae],
             'gen_mape': [gen_mape],
             'lgbm_mape': [lgbm_mape],
             'symbol_mape': [symbol_mape],
             'fifty_mape': [fifty_mape],
             'sixty_mape': [sixty_mape],
             'thirds_mape': [thirds_mape],
             'bagged_mape': [bagged_mape]
             }))


        lower_range = upper_range

    # Output range results as csv
    overall_results_output.to_csv('xgboost-models/range-results-' + run_str+ '.csv', columns=['lower_range',
                                                                                              'upper_range',
                                                                                              'gen_mae',
                                                                                              'lgbm_mae',
                                                                                              'symbol_mae',
                                                                                              'fifty_mae',
                                                                                              'sixty_mae',
                                                                                              'thirds_mae',
                                                                                              'bagged_mae',
                                                                                              'gen_mape',
                                                                                              'lgbm_mape',
                                                                                              'symbol_mape',
                                                                                              'fifty_mape',
                                                                                              'sixty_mape',
                                                                                              'thirds_mape',
                                                                                              'bagged_mape'])