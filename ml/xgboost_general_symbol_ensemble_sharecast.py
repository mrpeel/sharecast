
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# import pickle
# import gzip
import joblib
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor, HuberRegressor

from categorical_encoder import *
from eval_results import *
#from autoencoder import *

from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger, ModelCheckpoint
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
    # with gzip.open(filename, 'wb') as f:
    #     f.write(pickle.dumps(object, protocol))
    joblib.dump(object, filename)


# @profile
def load(filename):
    """Loads a compressed object from disk
    """
    # with gzip.open(filename, 'rb') as f:
    #     file_content = f.read()
    #
    # object = pickle.loads(file_content)
    # return object
    joblib.load(filename)

def round_down(num, divisor):
    return num - (num%divisor)

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
    # Ensure data shape is correct
    actual_y = actual_y.reshape(actual_y.shape[0], )
    prediction_y = prediction_y.reshape(prediction_y.shape[0], )
    # Calculate MAPE
    diff = np.absolute((actual_y - prediction_y) / np.clip(np.absolute(actual_y), 1., None))
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


def flatten_array(np_array):
    if np_array.ndim > 1:
        new_array = np.concatenate(np_array)
    else:
        new_array = np_array

    return new_array


# @profile
def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
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
    df = pd.read_pickle('data/ml-aug-sample.pkl.gz', compression='gzip')
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

    # Fill categrical vals with phrase 'NA'
    for col in CATEGORICAL_COLUMNS:
        df[col].fillna('NA', inplace=True)

    # Fill N/A vals with dummy number
    df.fillna(-999, inplace=True)

    return df

# @profile
def prep_data():
    df = load_data()

    return df


# @profile
def divide_data(share_data):
    # Use pandas dummy columns for categorical columns other than symbol
    # share_data = pd.get_dummies(data=share_data, columns=['4WeekBollingerPrediction',
    #                                                        '4WeekBollingerType',
    #                                                        '12WeekBollingerPrediction',
    #                                                        '12WeekBollingerType',
    #                                                        'quoteDate_DAY',
    #                                                        'quoteDate_DAYOFWEEK',
    #                                                        'quoteDate_MONTH',
    #                                                        'quoteDate_YEAR'])


    # Run one-hot encoding and keep original values for entity embedding
    # share_data = pd.concat([share_data, pd.get_dummies(share_data['4WeekBollingerPrediction'])], axis=1)
    # share_data = pd.concat([share_data, pd.get_dummies(share_data['4WeekBollingerType'])], axis=1)
    # share_data = pd.concat([share_data, pd.get_dummies(share_data['12WeekBollingerPrediction'])], axis=1)
    # share_data = pd.concat([share_data, pd.get_dummies(share_data['12WeekBollingerType'])], axis=1)
    # share_data = pd.concat([share_data, pd.get_dummies(share_data['quoteDate_DAY'])], axis=1)
    # share_data = pd.concat([share_data, pd.get_dummies(share_data['quoteDate_DAYOFWEEK'])], axis=1)
    # share_data = pd.concat([share_data, pd.get_dummies(share_data['quoteDate_MONTH'])], axis=1)
    # share_data = pd.concat([share_data, pd.get_dummies(share_data['quoteDate_YEAR'])], axis=1)


    symbol_models = {}
    symbols = share_data['symbol'].unique()
    # For testing only take the first 10 elements
    # symbols = symbols[:10]
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
        model_data = share_data.loc[share_data['symbol'] == symbol]

        # Remove symbol as it has now been encoded separately
        #model_data.drop(['symbol'], axis=1, inplace=True)

        msk = np.random.rand(len(model_data)) < 0.75

        # Prep dataframes and reset index for appending
        df_train = model_data[msk]
        df_test = model_data[~msk]
        df_train.reset_index()
        df_test.reset_index()

        # MAke sure a minimum number of rows are present in sample for symbol
        if (len(df_train) > 150 & len(df_test) > 50):
            # symbols_train_y[symbol] = df_train[LABEL_COLUMN + '_scaled'].values
            # symbols_train_actuals[symbol] = df_train[LABEL_COLUMN].values
            # symbols_train_x[symbol] = df_train.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1).values
            #
            # symbols_test_actuals[symbol] = df_test[LABEL_COLUMN].values
            # symbols_test_y[symbol] = df_test[LABEL_COLUMN + '_scaled'].values
            # symbols_test_x[symbol] = df_test.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1).values

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

def preprocess_train_data(train_x_df, train_y_df):
    scaler = MinMaxScaler(feature_range=(0,1)) #StandardScaler()
    train_x_df[CONTINUOUS_COLUMNS] = scaler.fit_transform(train_x_df[CONTINUOUS_COLUMNS].values)

    # Save data fo use in genetic algorithm
    train_x_df.to_pickle('data/pp_train_x_df.pkl.gz', compression='gzip')
    train_y_df.to_pickle('data/pp_train_y_df.pkl.gz', compression='gzip')

    # Use categorical entity embedding encoder
    ce = Categorical_encoder(strategy="entity_embedding", verbose=False)
    df_train_transform = ce.fit_transform(train_x_df, train_y_df[0])

    # df_train_transform['symbol'] = train_x_df['symbol']

    return df_train_transform, scaler, ce


def preprocess_test_data(test_x_df, scaler, ce):
    test_x_df[CONTINUOUS_COLUMNS] = scaler.transform(test_x_df[CONTINUOUS_COLUMNS].values)

    test_x_df.to_pickle('data/pp_test_x_df.pkl.gz', compression='gzip')

    # Use categorical entity embedding encoder
    df_test_transform = ce.transform(test_x_df)

    # df_test_transform['symbol'] = test_x_df['symbol']

    return df_test_transform

# @profile
def train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x, keras_models):
    #Train general model
    models = {}
    # Create model
    models['log_y'] = xgb.XGBRegressor(nthread=-1, n_estimators=500, max_depth=70, base_score=0.1, colsample_bylevel=0.7,
                                           colsample_bytree=1.0, gamma=0, learning_rate=0.025, min_child_weight=3)

    all_train_y = df_all_train_y.values
    all_train_log_y = safe_log(all_train_y)
    all_train_x = df_all_train_x.values
    all_test_actuals = df_all_test_actuals.values
    all_test_y = df_all_test_y.values
    all_test_x = df_all_test_x.values
    all_test_log_y = safe_log(all_test_y)


    # mape_vals_train = keras_models['mape_model'].predict(all_train_x)
    # mape_vals_test = keras_models['mape_model'].predict(all_test_x)
    mae_vals_train = keras_models['mae_intermediate_model'].predict(all_train_x)
    mae_vals_test = keras_models['mae_intermediate_model'].predict(all_test_x)
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
    models['log_y'].fit(all_train_x, all_train_y, early_stopping_rounds=25, eval_metric='mae', eval_set=eval_set,
    #model.fit(all_train_x, all_train_y, early_stopping_rounds=250, eval_metric=mape_log_y_eval, eval_set=eval_set,
    #model.fit(stacked_train_x, all_train_y, early_stopping_rounds=250, eval_metric=huber_loss_eval, eval_set=eval_set,
                verbose=True)


    gc.collect()

    predictions = models['log_y'].predict(all_test_x)
    #### Double exp #######
    inverse_scaled_predictions = safe_exp(predictions)

    eval_results({'xgboost_mae': {
                        'log_y': all_test_y,
                        'actual_y': all_test_actuals,
                        'log_y_predict': predictions,
                        'y_predict': inverse_scaled_predictions
                }
    })


    models['log_log_y'] = xgb.XGBRegressor(nthread=-1, n_estimators=500,
                                           max_depth=130,
                                           base_score=0.7,
                                           colsample_bylevel=0.55,
                                           colsample_bytree=0.85,
                                           gamma=0.15,
                                           min_child_weight=2,
                                           learning_rate=0.025)

    eval_set = [(all_test_x, all_test_log_y)]
    models['log_log_y'].fit(all_train_x, all_train_log_y, early_stopping_rounds=25, eval_metric='mae', eval_set=eval_set,
                verbose=True)


    gc.collect()

    log_predictions = models['log_log_y'].predict(all_test_x)
    #### Double exp #######
    log_inverse_scaled_predictions = safe_exp(safe_exp(log_predictions))

    eval_results({'xgboost_log_log_mae': {
                        'actual_y': all_test_actuals,
                        'y_predict': log_inverse_scaled_predictions
                }
    })

    models['keras_mae'] = xgb.XGBRegressor(nthread=-1, n_estimators=500, max_depth=70, learning_rate=0.025,
                                           base_score=0.25, colsample_bylevel=0.4, colsample_bytree=0.55,
                                           gamma=0, min_child_weight=0)

    eval_set = [(mae_vals_test, all_test_y)]
    models['keras_mae'].fit(mae_vals_train, all_train_y, early_stopping_rounds=25, eval_metric='mae',
                                            eval_set=eval_set,verbose=True)
    gc.collect()

    keras_log_predictions = models['keras_mae'].predict(mae_vals_test)
    #### Double exp #######
    keras_inverse_scaled_predictions = safe_exp(keras_log_predictions)

    eval_results({'xgboost_keras': {
                            'log_y': all_test_y,
                            'actual_y': all_test_actuals,
                            'log_y_predict': keras_log_predictions,
                            'y_predict': keras_inverse_scaled_predictions
                    }
        })

    models['keras_log_mae'] = xgb.XGBRegressor(nthread=-1, n_estimators=500,
                                               max_depth=130,
                                               base_score=0.4,
                                               colsample_bylevel=0.4,
                                               colsample_bytree=0.4,
                                               gamma=0,
                                               min_child_weight=0,
                                               learning_rate=0.025)

    eval_set = [(mae_vals_test, all_test_log_y)]
    models['keras_log_mae'].fit(mae_vals_train, all_train_log_y, early_stopping_rounds=25, eval_metric='mae',
                                            eval_set=eval_set,verbose=True)
    gc.collect()

    keras_log_log_predictions = models['keras_log_mae'].predict(mae_vals_test)
    #### Double exp #######
    keras_log_inverse_scaled_predictions = safe_exp(safe_exp(keras_log_log_predictions))

    eval_results({'xgboost_keras_log_y': {
                            'actual_y': all_test_actuals,
                            'y_predict': keras_log_inverse_scaled_predictions
                    }
        })


    range_results({
        'xgboost_mae': inverse_scaled_predictions,
        'xgboost_log_mae': log_inverse_scaled_predictions,
        'xgboost_keras_mae': keras_inverse_scaled_predictions,
        'xgboost_keras_log_mae': keras_log_inverse_scaled_predictions
        }, all_test_actuals)

    for key in models:
        save(models[key], 'models/xgb-' + key + '.model.gz')

    return models

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
        symbol_model = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=130, base_score=0.35,
                                        colsample_bylevel=0.8, colsample_bytree = 0.8, gamma = 0, learning_rate = 0.01,
                                        max_delta_step = 0, min_child_weight = 0)

        eval_set = [(stacked_test_x, test_y)]
        symbol_model.fit(stacked_train_x, train_y, early_stopping_rounds=250, eval_metric='mae', eval_set=eval_set,
        #symbol_model.fit(train_x, train_y, early_stopping_rounds=250, eval_metric=mape_log_y_eval, eval_set=eval_set,
                         verbose=False)

        gc.collect()

        # Create model
        symbol_model_log = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=130, base_score=0.35,
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
        'num_leaves': 65536,
        'max_bin': 5000000,
        'boosting_type': "gbdt",
        'feature_fraction': 0.7,
        'min_split_gain': 0,
        'boost_from_average': True,
    }

    gbms = {}

    train_x = df_all_train_x.values
    train_y = df_all_train_y[0].values
    train_log_y = safe_log(train_y)
    test_x = df_all_test_x.values
    test_actuals = df_all_test_actuals.values
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)


    train_set = lgb.Dataset(df_all_train_x, label=train_y)
    eval_set = lgb.Dataset(df_all_test_x, reference=train_set, label=test_y)

    params['histogram_pool_size'] = 8192
    params['metric'] = ['mae', 'huber']
    params['metric_freq'] = 10

    # feature_name and categorical_feature
    gbms['log_y'] = lgb.train(params,
                    train_set,
                    valid_sets=eval_set,  # eval training data
                    # feval=mle_eval,
                    # Set learning rate to reduce every 10 iterations
                    learning_rates=lambda iter: 0.125 * (0.999 ** round_down(iter, 10)),
                    num_boost_round=500,
                    early_stopping_rounds=5)

    iteration_number = 500

    if gbms['log_y'].best_iteration:
        iteration_number = gbms['log_y'].best_iteration

    predictions = gbms['log_y'].predict(df_all_test_x, num_iteration=iteration_number)
    eval_predictions = safe_exp(predictions)

    eval_results({'lgbm_log_y': {
                        'log_y': test_y,
                        'actual_y': test_actuals,
                        'log_y_predict': predictions,
                        'y_predict': eval_predictions
                }
    })


    # feature_name and categorical_feature
    gbms['log_log_y'] = lgb.train(params,
                    train_set,
                    valid_sets=eval_set,
                    # feval=mae_eval,
                    # learning_rates=lambda iter: 0.75 * (0.9995 ** iter),
                    num_boost_round=2000,
                    early_stopping_rounds=50,
                    verbose_eval=10)

    gc.collect()

    iteration_number = 2000

    if gbms['log_log_y'].best_iteration:
        iteration_number = gbms['log_log_y'].best_iteration


    # Make predictions
    log_log_predictions = gbms['log_log_y'].predict(test_x, num_iteration=iteration_number)
    predictions_log_y = safe_exp(log_log_predictions)
    log_log_inverse_scaled_predictions = safe_exp(predictions_log_y)




    range_results({
        # 'lgbm_log_y':log_inverse_scaled_predictions,
        'lgbm_log_log_y': log_log_inverse_scaled_predictions
        }, test_actuals)

    for key in gbms:
        save(gbms[key], 'models/lgbm-' + key + '.model.gz')

    return gbms


def compile_keras_model(network, input_shape):
    """Compile a sequential model.
    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.
    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout = network['dropout']
    model_type = network['model_type']
    if 'int_layer' in network:
            int_layer = network['int_layer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(dropout))

    if 'int_layer' in network:
        model.add(Dense(int_layer, activation=activation, name="int_layer"))
        model.add(Dropout(dropout))

    # Output layer.
    model.add(Dense(1, activation='linear'))

    if model_type == "mape":
        model.compile(loss=sc_mean_absolute_percentage_error, optimizer=optimizer, metrics=['mae'])
    else:
        model.compile(loss='mae', optimizer=optimizer, metrics=[sc_mean_absolute_percentage_error])

    return model

# @profile
def train_keras_nn(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals, df_all_test_y,
                   df_all_test_x):
    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.values
    test_actuals = df_all_test_actuals.values
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)
    test_x = df_all_test_x.values


    print('Fitting Keras mape model...')

    network = {
        'nb_neurons': 512,
        'nb_layers': 3,
        'activation': "relu",
        'optimizer': "adagrad",
        'batch_size': 256,
        'dropout': 0.05,
        'model_type': "mape"
    }

    input_shape = (train_x.shape[1],)

    p_model = compile_keras_model(network, input_shape)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    csv_logger = CSVLogger('./logs/actual-mape-training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    history = p_model.fit(train_x,
                          train_actuals,
                          validation_data=(test_x, test_actuals),
                          epochs=20000,
                          batch_size=network['batch_size'],
                          callbacks=[reduce_lr, early_stopping, csv_logger, checkpointer],
                          verbose=0)

    p_model.load_weights('weights.hdf5')

    predictions = p_model.predict(test_x)

    eval_results({'keras_mape': {
                        'actual_y': test_actuals,
                        'y_predict': predictions
                }
    })

    gc.collect()

    print('Building Keras mae model...')

    network = {
        'nb_layers': 4,
        'nb_neurons': 768,
        'activation': "relu",
        'optimizer': "adamax",
        'dropout': 0.05,
        'batch_size': 256,
        'model_type': "mae",
        'int_layer': 30
    }

    input_shape = (train_x.shape[1],)

    model = compile_keras_model(network, input_shape)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=26)
    csv_logger = CSVLogger('./logs/log-training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    print('Fitting Keras mae model...')

    history = model.fit(train_x,
                        train_y,
                        validation_data=(test_x, test_y),
                        epochs=20000,
                        batch_size=network['batch_size'],
                        callbacks=[reduce_lr, early_stopping, checkpointer, csv_logger],
                        verbose=0)

    model.load_weights('weights.hdf5')


    print('Executing keras predictions...')

    log_y_predictions = model.predict(test_x)
    exp_predictions = safe_exp(log_y_predictions)

    eval_results({'keras_log_y': {
                        #'log_y': test_y,
                        'actual_y': test_actuals,
                        #'log_y_predict': log_predictions,
                        'y_predict': exp_predictions
                }
    })

    range_results({
        'keras_mape': predictions,
        'keras_log_y': exp_predictions,
    }, test_actuals)

    gc.collect()


    # Construct models which output final twelve weights as predictions
    # mape_intermediate_model = Model(inputs=p_model.input,
    #                                  outputs=p_model.get_layer('mape_twelve').output)
    #
    mae_intermediate_model = Model(inputs=model.input,
                                      outputs=model.get_layer('int_layer').output)

    # save models
    p_model.save('models/keras-mape-model.h5')
    model.save('models/keras-mae-model.h5')
    mae_intermediate_model.save('models/keras-mae-intermediate-model.h5')

    return {
        'mape_model': p_model,
        'mae_model': model,
        'mae_intermediate_model': mae_intermediate_model
        }

def deep_bagging(train_predictions, train_actuals, test_predictions, test_actuals):

    train_x = train_predictions.values
    train_y = train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_x = test_predictions.values
    test_y = test_actuals[0].values
    test_log_y = safe_log(test_y)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    input_shape = (train_x.shape[1],)

    network = {
        'nb_neurons': 64,
        'nb_layers': 2,
        'activation': "selu",
        'optimizer': "adagrad",
        'batch_size': 32,
        'dropout': 0.5,
        'model_type': "mape"
    }

    model = compile_keras_model(network, input_shape)

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])

    history = model.fit(train_x, train_log_y,
                        batch_size=network['batch_size'],
                        epochs=20000,
                        verbose=0,
                        validation_data=(test_x, test_log_y),
                        callbacks=[csv_logger, reduce_lr, early_stopping, checkpointer])

    print('\rResults')

    # hist_epochs = len(history.history['val_loss'])

    model.load_weights('weights.hdf5')
    predictions = model.predict(test_x)
    prediction_results = predictions.reshape(predictions.shape[0], )
    prediction_results = safe_exp(prediction_results)


    eval_results({'deep_bagged_predictions': {
                        'actual_y': test_y,
                        'y_predict': prediction_results
                }
    })

    range_results({
        'deep_bagged_predictions': prediction_results
    }, test_y)

    return model, prediction_results

# @profile
def bagging(df_all_test_x, df_all_test_actuals, gen_models, lgbm_models, keras_models, deep_bagged_predictions):

    test_actuals = df_all_test_actuals.values
    test_x = df_all_test_x.values

    print('Running model predictions')
    gen = gen_models['log_y'].predict(test_x)
    gen = safe_exp(gen)
    log_gen = gen_models['log_log_y'].predict(test_x)
    log_gen = safe_exp(safe_exp(log_gen))

    lgbm = lgbm_models['log_log_y'].predict(test_x)
    lgbm = safe_exp(safe_exp(lgbm))

    log_lgbm = lgbm_models['log_log_y'].predict(test_x)
    log_lgbm = safe_exp(safe_exp(log_lgbm))

    keras_mape = keras_models['mape_model'].predict(test_x)

    keras_mae = keras_models['mae_model'].predict(test_x)
    keras_mae = safe_exp(keras_mae)

    # Generate values required for keras -> xgboost model
    keras_mae_intermediate = keras_models['mae_intermediate_model'].predict(test_x)
    keras_gen = gen_models['keras_mae'].predict(keras_mae_intermediate)
    keras_gen = safe_exp(keras_gen)

    keras_log_gen = gen_models['keras_log_mae'].predict(keras_mae_intermediate)
    keras_log_gen = safe_exp(safe_exp(keras_log_gen))


    # Call deep bagging
    # train_predictions = pd.DataFrame.from_dict({
    #     'xgboost_log': gen,
    #     'xgboost_log_log': log_gen,
    #     'lgbm_log_log': log_lgbm,
    #     'keras_mape': keras_mape,
    #     'keras_log': keras_mae,
    #     'xgboost_keras_log': keras_gen,
    #     'xgboost_keras_log_log': keras_log_gen,
    # })


    # Reshape arrays
    gen = gen.reshape(gen.shape[0], 1)
    lgbm = lgbm.reshape(log_lgbm.shape[0], 1)
    log_lgbm = log_lgbm.reshape(log_lgbm.shape[0], 1)
    log_gen = log_gen.reshape(log_gen.shape[0], 1)
    keras_mape = keras_mape.reshape(keras_mape.shape[0], 1)
    keras_mae = keras_mae.reshape(keras_mape.shape[0], 1)
    keras_gen = keras_gen.reshape(keras_gen.shape[0], 1)
    keras_log_gen = keras_log_gen.reshape(keras_gen.shape[0], 1)
    deep_bagged_predictions = deep_bagged_predictions.reshape(deep_bagged_predictions.shape[0], 1)

    small_pred_average = (keras_log_gen + log_gen) / 2.

    print('gen shape', gen.shape)
    print('log_lgbm shape', log_lgbm.shape)
    print('lgbm shape', lgbm.shape)
    print('log_gen shape', log_gen.shape)
    print('keras_mape shape', keras_mape.shape)
    print('keras_mae shape', keras_mae.shape)
    print('keras_gen shape', keras_gen.shape)
    print('keras_log_gen shape', keras_log_gen.shape)
    print('small_pred_average shape', small_pred_average.shape)



    print('Bagging predictions')

    # Set default value to mape
    bagged_predictions = np.copy(keras_gen)

    # -10 - -5 should should use gen
    mask_neg_10_5 = ((gen > -10) & (gen <= 5))
    bagged_predictions[mask_neg_10_5] = lgbm[mask_neg_10_5]

    # 2 - 20 should should use gen
    mask_2_20 = ((gen > 0) & (gen <= 22))
    bagged_predictions[mask_2_20] = gen[mask_2_20]

    # 2 > and < 5 use average of gen and log_gen
    mask_2_5 = ((small_pred_average > 1.5) & (small_pred_average <= 6))
    bagged_predictions[mask_2_5] = ((gen + log_gen) / 2.)[mask_2_5]

    # -5 - 0 should should average of keras_mape and log_gen
    mask_neg_5_0 = ((small_pred_average > -6.5) & (small_pred_average <= 0))
    bagged_predictions[mask_neg_5_0] = ((keras_mape + log_gen) / 2.)[mask_neg_5_0]

    # 0 > and < 2 use keras_mape only
    mask_neg_0_2 = ((small_pred_average > -0.5) & (small_pred_average <= 3.5) & (keras_mape > -0.5) &
                    (keras_mape <= 2.5))
    bagged_predictions[mask_neg_0_2] = keras_mape[mask_neg_0_2]


    eval_results({'bagged_predictions': {
                        'actual_y': test_actuals,
                        'y_predict': bagged_predictions
                }
    })

    range_results({
        'gen': gen,
        'lgbm': lgbm,
        'log_lgbm': log_lgbm,
        'log_gen': log_gen,
        'xgboost_keras': keras_gen,
        'xgboost_log_keras': keras_log_gen,
        'keras_mape': keras_mape,
        'keras_mae': keras_mae,
        'bagged_predictions': bagged_predictions,
        'deep_bagged_predictions': deep_bagged_predictions
    }, test_actuals)

def export_final_data(df_all_train_x, df_all_train_actuals, df_all_test_x, df_all_test_actuals,
                gen_models, lgbm_models, keras_models):

    # export results
    df_all_test_actuals.to_pickle('data/test_actuals.pkl.gz', compression='gzip')
    df_all_train_actuals.to_pickle('data/train_actuals.pkl.gz', compression='gzip')


    test_x = df_all_test_x.values
    train_x = df_all_train_x.values

    print('Exporting individual predictions')
    gen_train = gen_models['log_y'].predict(train_x)
    gen_train = safe_exp(gen_train)

    gen_test = gen_models['log_y'].predict(test_x)
    gen_test = safe_exp(gen_test)

    log_gen_train = gen_models['log_log_y'].predict(train_x)
    log_gen_train = safe_exp(safe_exp(log_gen_train))

    log_gen_test = gen_models['log_log_y'].predict(test_x)
    log_gen_test = safe_exp(safe_exp(log_gen_test))

    lgbm_train = lgbm_models['log_y'].predict(train_x)
    lgbm_train = safe_exp(lgbm_train)

    lgbm_test = lgbm_models['log_y'].predict(test_x)
    lgbm_test = safe_exp(lgbm_test)

    log_lgbm_train = lgbm_models['log_log_y'].predict(train_x)
    log_lgbm_train = safe_exp(safe_exp(log_lgbm_train))

    log_lgbm_test = lgbm_models['log_log_y'].predict(test_x)
    log_lgbm_test = safe_exp(safe_exp(log_lgbm_test))

    keras_mape_train = keras_models['mape_model'].predict(train_x)

    keras_mape_test = keras_models['mape_model'].predict(test_x)

    keras_log_train = keras_models['mae_model'].predict(train_x)
    keras_log_train = safe_exp(keras_log_train)

    keras_log_test = keras_models['mae_model'].predict(test_x)
    keras_log_test = safe_exp(keras_log_test)

    # Generate values required for keras -> xgboost model
    keras_mae_intermediate_train = keras_models['mae_intermediate_model'].predict(train_x)
    keras_mae_intermediate_test = keras_models['mae_intermediate_model'].predict(test_x)

    xgboost_keras_gen_train = gen_models['keras_mae'].predict(keras_mae_intermediate_train)
    xgboost_keras_gen_train = safe_exp(xgboost_keras_gen_train)

    xgboost_keras_gen_test = gen_models['keras_mae'].predict(keras_mae_intermediate_test)
    xgboost_keras_gen_test = safe_exp(xgboost_keras_gen_test)

    xgboost_keras_log_gen_train = gen_models['keras_log_mae'].predict(keras_mae_intermediate_train)
    xgboost_keras_log_gen_train = safe_exp(safe_exp(xgboost_keras_log_gen_train))

    xgboost_keras_log_gen_test = gen_models['keras_log_mae'].predict(keras_mae_intermediate_test)
    xgboost_keras_log_gen_test = safe_exp(safe_exp(xgboost_keras_log_gen_test))

    # Make consistent shape for outputs from keras
    keras_mape_train = keras_mape_train.reshape(keras_mape_train.shape[0], )
    keras_log_train = keras_log_train.reshape(keras_log_train.shape[0], )


    train_predictions = pd.DataFrame.from_dict({
        'xgboost_log': flatten_array(gen_train),
        'xgboost_log_log': flatten_array(log_gen_train),
        'lgbmlog': flatten_array(lgbm_train),
        'lgbm_log_log': flatten_array(log_lgbm_train),
        'keras_mape': flatten_array(keras_mape_train),
        'keras_log': flatten_array(keras_log_train),
        'xgboost_keras_log': flatten_array(xgboost_keras_gen_train),
        'xgboost_keras_log_log': flatten_array(xgboost_keras_log_gen_train),
    })
    train_predictions.to_pickle('data/train_predictions.pkl.gz', compression='gzip')

    # Make consistent shape for outputs from keras
    keras_mape_test = keras_mape_test.reshape(keras_mape_test.shape[0], )
    keras_log_test = keras_log_test.reshape(keras_log_test.shape[0], )

    test_predictions = pd.DataFrame.from_dict({
        'xgboost_log': flatten_array(gen_test),
        'xgboost_log_log': flatten_array(log_gen_test),
        'lgb': flatten_array(lgbm_test),
        'lgbm_log_log': flatten_array(log_lgbm_test),
        'keras_mape': flatten_array(keras_mape_test),
        'keras_log': flatten_array(keras_log_test),
        'xgboost_keras_log': flatten_array(xgboost_keras_gen_test),
        'xgboost_keras_log_log': flatten_array(xgboost_keras_log_gen_test),

    })
    test_predictions.to_pickle('data/test_predictions.pkl.gz', compression='gzip')

    return train_predictions, test_predictions

def main():
    # Prepare run_str
    run_str = datetime.datetime.now().strftime('%Y%m%d%H%M')

    # log = open("logs/execution-" + run_str + ".log", "a")
    # sys.stdout = log

    # Retrieve and run combined prep on data
    share_data  = prep_data()
    gc.collect()

    # Divide data into symbol sand general data for training an testing
    symbol_map, symbols_train_y, symbols_train_actuals, symbols_train_x, symbols_test_actuals, \
    symbols_test_y, symbols_test_x, df_all_train_y, df_all_train_actuals, df_all_train_x, \
    df_all_test_actuals, df_all_test_y, df_all_test_x = divide_data(share_data)

    del share_data
    gc.collect()

    # Run pre-processing steps
    df_all_train_x, scaler, ce = preprocess_train_data(df_all_train_x, df_all_train_y)
    df_all_test_x  = preprocess_test_data(df_all_test_x, scaler, ce)

    # train_keras_linear(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals,
    #                    df_all_test_y, df_all_test_x)

    # sklearn_models = train_sklearn_models(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y,
    #                                       df_all_test_x)

    # Write scaler and categorical encoder to files
    save(scaler, 'models/scaler.pkl.gz')
    save(ce, 'models/ce.pkl.gz')

    # Write data to files
    df_all_train_x.to_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_train_y.to_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
    df_all_train_actuals.to_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    df_all_test_x.to_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    df_all_test_y.to_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
    df_all_test_actuals.to_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')


    keras_models = train_keras_nn(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals,
                                  df_all_test_y, df_all_test_x)


    # Train the general model
    gen_models = train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x,
                                    keras_models)

    lgbm_models = train_lgbm(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x)



    train_predictions, test_predictions = export_final_data(df_all_train_x, df_all_train_actuals, df_all_test_x,
                                                            df_all_test_actuals, gen_models, lgbm_models, keras_models)

    bagging_model, deep_bagged_predictions = deep_bagging(train_predictions, df_all_train_actuals, test_predictions,
                                                         df_all_test_actuals)

    bagging(df_all_test_x, df_all_test_actuals, gen_models, lgbm_models, keras_models, deep_bagged_predictions)


if __name__ == "__main__":
    main()