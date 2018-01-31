
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import joblib
import sys
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split

from categorical_encoder import *
from eval_results import *
#from autoencoder import *
from compile_keras import *
from keras.models import load_model
import logging


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

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def print(*log_line_vals):
    """
        Prints and logs a string with an arbitrary set of values
    """
    log_line = ''
    for arg in log_line_vals:
        log_line = log_line + str(arg) + ' '

    sys.stdout.write(log_line.strip() + '\n')
    logging.info(log_line.strip())


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
    model_object = joblib.load(filename)
    return model_object

def round_down(num, divisor):
    return num - (num%divisor)

# @profile
def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log1p(np.absolute(return_vals))
    return_vals[neg_mask] *= -1.
    return return_vals

# @profile
def safe_exp(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.expm1(np.clip(np.absolute(return_vals), -7, 7))
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


def mae_mape(actual_y, prediction_y):
    mape = safe_mape(actual_y, prediction_y)
    mae = mean_absolute_error(actual_y, prediction_y)
    return mape * mae

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
    df = pd.read_pickle('data/ml-dec-data.pkl.gz', compression='gzip')
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

        msk = np.random.rand(len(model_data)) < 0.8

        # Prep dataframes and reset index for appending
        df_train = model_data[msk]
        df_test = model_data[~msk]
        df_train.reset_index()
        df_test.reset_index()

        # Make sure a minimum number of rows are present in sample for symbol
        if (len(df_train) > 50 & len(df_test) > 10):
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


def train_preprocessor(train_x_df, train_y_df):
    print('Training pre-processor...')

    print('Scaling data...')
    scaler = MinMaxScaler(feature_range=(0,1)) #StandardScaler()
    train_x_df[CONTINUOUS_COLUMNS] = scaler.fit_transform(train_x_df[CONTINUOUS_COLUMNS].values)


    print('Encoding categorical data...')
    # Use categorical entity embedding encoder
    ce = Categorical_encoder(strategy="entity_embedding", verbose=False)
    df_train_transform = ce.fit_transform(train_x_df, train_y_df[0])

    # Write scaler and categorical encoder to files
    save(scaler, 'models/scaler.pkl.gz')
    save(ce, 'models/ce.pkl.gz')

    return df_train_transform, scaler, ce


def execute_preprocessor(transform_df, scaler, ce):
    print('Executing pre-processor on supplied data...')

    print('Scaling data...')
    transform_df[CONTINUOUS_COLUMNS] = scaler.transform(transform_df[CONTINUOUS_COLUMNS].values)

    print('Encoding categorical data...')
    # Use categorical entity embedding encoder
    transform_df = ce.transform(transform_df)

    return transform_df


# @profile
def train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x, keras_models):
    #Train general model
    models = {}

    tree_method = 'auto'
    nthread = -1
    # Create model
    models['log_y'] = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method,
                                       n_estimators=1000, max_depth=70, base_score=0.1,
                                       colsample_bylevel=0.7, colsample_bytree=1.0, gamma=0, learning_rate=0.05,
                                       min_child_weight=3)

    all_train_y = df_all_train_y.values
    all_train_log_y = safe_log(all_train_y)
    all_train_x = df_all_train_x.values
    all_test_actuals = df_all_test_actuals.values
    all_test_y = df_all_test_y.values
    all_test_x = df_all_test_x.values
    all_test_log_y = safe_log(all_test_y)

    print('Training xgboost log of y model...')
    x_train, x_test, y_train, y_test = train_test_split(all_train_x, all_train_y, test_size = 0.15)

    eval_set = [(x_test, y_test)]
    models['log_y'].fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae', eval_set=eval_set,
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

    print('Retrieving keras intermediate model vals...')
    mae_vals_train = keras_models['mae_intermediate_model'].predict(all_train_x)
    mae_vals_test = keras_models['mae_intermediate_model'].predict(all_test_x)

    print('Training xgboost log of y with keras outputs model...')
    models['keras_mae'] = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method, n_estimators=1000, max_depth=70,
                                           learning_rate=0.025, base_score=0.25, colsample_bylevel=0.4,
                                           colsample_bytree=0.55, gamma=0, min_child_weight=0)

    x_train, x_test, y_train, y_test = train_test_split(mae_vals_train, all_train_y, test_size=0.15)

    eval_set = [(x_test, y_test)]
    models['keras_mae'].fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae',
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

    print('Training xgboost log of log of y with keras outputs model...')
    models['keras_log_mae'] = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method, n_estimators=1000,
                                               max_depth=130,
                                               base_score=0.4,
                                               colsample_bylevel=0.4,
                                               colsample_bytree=0.4,
                                               gamma=0,
                                               min_child_weight=0,
                                               learning_rate=0.025)

    x_train, x_test, y_train, y_test = train_test_split(mae_vals_train, all_train_log_y, test_size=0.15)

    eval_set = [(x_test, y_test)]
    models['keras_log_mae'].fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae',
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
        'xgboost_keras_mae': keras_inverse_scaled_predictions,
        'xgboost_keras_log_mae': keras_log_inverse_scaled_predictions
        }, all_test_actuals)

    save(models['log_y'], 'models/xgb-log_y.model.gz')
    save(models['keras_mae'], 'models/xgb-keras_mae.model.gz')
    save(models['keras_log_mae'], 'models/xgb-keras_log_mae.model.gz')

    return models


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


    print('Training keras mape model...')

    network = {
        'hidden_layers': [5, 5, 5],
        'activation': 'relu',
        'optimizer': 'Adagrad',
        'kernel_initializer': 'glorot_uniform',
        'batch_size': 256,
        'dropout': 0.05,
        'model_type': 'mape',
    }

    dimensions = train_x.shape[1]

    p_model = compile_keras_model(network, dimensions)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=12)
    csv_logger = CSVLogger('./logs/actual-mape-training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    # Reorder array - get array index
    s = np.arange(train_x.shape[0])
    # Reshuffle index
    np.random.shuffle(s)

    # Create array using new index
    x_shuffled_train = train_x[s]
    y_shuffled_train = train_actuals[s]

    history = p_model.fit(x_shuffled_train,
                          y_shuffled_train,
                          validation_split=0.15,
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


    network = {
        'hidden_layers': [7, 7, 7, 7],
        'activation': 'relu',
        'optimizer': 'Adamax',
        'kernel_initializer': 'normal',
        'dropout': 0,
        'batch_size': 512,
        'model_type': 'mae',
        'int_layer': 30,
    }

    dimensions = train_x.shape[1]

    model = compile_keras_model(network, dimensions)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=12)
    csv_logger = CSVLogger('./logs/log-training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    print('Training keras mae model...')

    # Reorder array - get array index
    s = np.arange(train_x.shape[0])
    # Reshuffle index
    np.random.shuffle(s)

    # Create array using new index
    x_shuffled_train = train_x[s]
    y_shuffled_train = train_y[s]

    history = model.fit(x_shuffled_train,
                        y_shuffled_train,
                        validation_split=0.15,
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

def train_deep_bagging(train_predictions, train_actuals, test_predictions, test_actuals):
    print('Training keras based bagging...')
    train_x = train_predictions.values
    train_y = train_actuals[0].values
    test_x = test_predictions.values
    test_y = test_actuals[0].values

    scaler = MinMaxScaler(feature_range=(0,1))
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    dimensions = train_x.shape[1]

    network = {
        'activation': 'PReLU',
        'optimizer': 'Nadam',
        'batch_size': 1024,
        'dropout': 0,
        'model_type': 'mae_mape',
        'kernel_initializer': 'normal',
        'hidden_layers': [5],
    }

    model = compile_keras_model(network, dimensions)

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])

    history = model.fit(train_x_scaled, train_y,
                        batch_size=network['batch_size'],
                        epochs=20000,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=[csv_logger, reduce_lr, early_stopping, checkpointer])

    print('\rResults')

    model.load_weights('weights.hdf5')
    predictions = model.predict(test_x_scaled)
    prediction_results = predictions.reshape(predictions.shape[0], )

    eval_results({'deep_bagged_predictions': {
                        'actual_y': test_y,
                        'y_predict': prediction_results
                }
    })

    range_results({
        'deep_bagged_predictions': prediction_results
    }, test_y)

    # save models
    model.save('models/keras-bagging-model.h5')
    save(scaler, 'models/deep-bagging-scaler.pkl.gz')

    return model, scaler, prediction_results

def execute_deep_bagging(model, scaler, bagging_data):
    print('Executing keras based bagging...')
    test_x = bagging_data.values

    test_x_scaled = scaler.transform(test_x)

    predictions = model.predict(test_x_scaled)
    prediction_results = predictions.reshape(predictions.shape[0], )

    return prediction_results

# @profile
def final_bagging(df_all_test_x, df_all_test_actuals, gen_models, keras_models, deep_bagged_predictions):
    print('Executing manual bagging...')
    test_actuals = df_all_test_actuals.values
    test_x = df_all_test_x.values

    print('Running model predictions')
    gen = gen_models['log_y'].predict(test_x)
    gen = safe_exp(gen)

    keras_mape = keras_models['mape_model'].predict(test_x)

    keras_mae = keras_models['mae_model'].predict(test_x)
    keras_mae = safe_exp(keras_mae)

    # Generate values required for keras -> xgboost model
    keras_mae_intermediate = keras_models['mae_intermediate_model'].predict(test_x)
    keras_gen = gen_models['keras_mae'].predict(keras_mae_intermediate)
    keras_gen = safe_exp(keras_gen)


    # Reshape arrays
    gen = gen.reshape(gen.shape[0], 1)
    keras_mape = keras_mape.reshape(keras_mape.shape[0], 1)
    keras_mae = keras_mae.reshape(keras_mape.shape[0], 1)
    keras_gen = keras_gen.reshape(keras_gen.shape[0], 1)
    deep_bagged_predictions = deep_bagged_predictions.reshape(deep_bagged_predictions.shape[0], 1)

    small_pred_average = keras_mape

    print('gen shape', gen.shape)
    print('keras_mape shape', keras_mape.shape)
    print('keras_mae shape', keras_mae.shape)
    print('keras_gen shape', keras_gen.shape)
    print('small_pred_average shape', small_pred_average.shape)

    print('Bagging predictions')

    # Set default value to mape
    bagged_predictions = np.copy(keras_gen)

    # -10 - -5 should should use gen
    mask_neg_10_5 = ((gen > -10) & (gen <= 5))
    bagged_predictions[mask_neg_10_5] = gen[mask_neg_10_5]

    # 2 - 20 should should use gen
    mask_2_20 = ((gen > 0) & (gen <= 22))
    bagged_predictions[mask_2_20] = gen[mask_2_20]

    # > 2 and < 5 use average of gen and log_gen
    mask_2_5 = ((small_pred_average > 1.5) & (small_pred_average <= 6))
    bagged_predictions[mask_2_5] = gen[mask_2_5]

    # -5 - 0 should should average of keras_mape
    mask_neg_5_0 = ((small_pred_average > -6.5) & (small_pred_average <= 0))
    bagged_predictions[mask_neg_5_0] = keras_mape[mask_neg_5_0]

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
        'xgboost_keras': keras_gen,
        'keras_mape': keras_mape,
        'keras_mae': keras_mae,
        'bagged_predictions': bagged_predictions,
        'deep_bagged_predictions': deep_bagged_predictions
    }, test_actuals)

def export_final_data(df_all_train_x, df_all_train_actuals, df_all_test_x, df_all_test_actuals,
                gen_models, keras_models):
    print('Exporting predictions data...')
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

    # Make consistent shape for outputs from keras
    keras_mape_train = keras_mape_train.reshape(keras_mape_train.shape[0], )
    keras_log_train = keras_log_train.reshape(keras_log_train.shape[0], )


    train_predictions = pd.DataFrame.from_dict({
        'xgboost_log': flatten_array(gen_train),
        'keras_mape': flatten_array(keras_mape_train),
        'keras_log': flatten_array(keras_log_train),
        'xgboost_keras_log': flatten_array(xgboost_keras_gen_train),
    })
    train_predictions.to_pickle('data/train_predictions.pkl.gz', compression='gzip')

    # Make consistent shape for outputs from keras
    keras_mape_test = keras_mape_test.reshape(keras_mape_test.shape[0], )
    keras_log_test = keras_log_test.reshape(keras_log_test.shape[0], )

    test_predictions = pd.DataFrame.from_dict({
        'xgboost_log': flatten_array(gen_test),
        'keras_mape': flatten_array(keras_mape_test),
        'keras_log': flatten_array(keras_log_test),
        'xgboost_keras_log': flatten_array(xgboost_keras_gen_test),
    })
    test_predictions.to_pickle('data/test_predictions.pkl.gz', compression='gzip')

    return train_predictions, test_predictions

def main(run_config):
    # Prepare run_str
    run_str = datetime.datetime.now().strftime('%Y%m%d%H%M')

    print('Starting sharecast run:', run_str)

    # Retrieve and divide data
    if 'load_data' in run_config and run_config['load_data'] == True:
        # Load and divide data
        share_data  = prep_data()
        gc.collect()

        # Divide data into symbol sand general data for training an testing
        symbol_map, symbols_train_y, symbols_train_actuals, symbols_train_x, symbols_test_actuals, \
        symbols_test_y, symbols_test_x, df_all_train_y, df_all_train_actuals, df_all_train_x, \
        df_all_test_actuals, df_all_test_y, df_all_test_x = divide_data(share_data)

        del share_data
        gc.collect()

        # Save data after dividing
        df_all_train_x.to_pickle('data/pp_train_x_df.pkl.gz', compression='gzip')
        df_all_train_y.to_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
        df_all_train_actuals.to_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
        df_all_test_x.to_pickle('data/pp_test_x_df.pkl.gz', compression='gzip')
        df_all_test_y.to_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
        df_all_test_actuals.to_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')
    else:
        # Data already divided
        print('Loading divided data')
        df_all_train_x = pd.read_pickle('data/pp_train_x_df.pkl.gz', compression='gzip')
        df_all_train_y = pd.read_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
        df_all_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
        df_all_test_x = pd.read_pickle('data/pp_test_x_df.pkl.gz', compression='gzip')
        df_all_test_y = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
        df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    if 'train_pre_process' in run_config and run_config['train_pre_process'] == True:
        # Execute pre-processing trainer
        df_all_train_x, scaler, ce = train_preprocessor(df_all_train_x, df_all_train_y)
        df_all_test_x = execute_preprocessor(df_all_test_x, scaler, ce)

    else:
        print('Loading pre-processing models')
        # Load pre-processing models
        scaler = load('models/scaler.pkl.gz')
        ce = load('models/ce.pkl.gz')

    if 'execute_pre_process' in run_config and run_config['execute_pre_process'] == True:
        print('Executing pre-processing')
        # Execute pre-processing
        df_all_train_x = execute_preprocessor(df_all_train_x, scaler, ce)
        df_all_test_x = execute_preprocessor(df_all_test_x, scaler, ce)

        # Write processed data to files
        df_all_train_x.to_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
        df_all_test_x.to_pickle('data/df_all_test_x.pkl.gz', compression='gzip')

    else:
        print('Loading pre-processed data')
        # Write processed data to files
        df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
        df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')

    if 'train_keras' in run_config and run_config['train_keras'] == True:
        # Train keras models
        keras_models = train_keras_nn(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals,
                                      df_all_test_y, df_all_test_x)
    else:
        print('Loading keras models')
        # Load keras models
        keras_models =  {
            'mape_model': load_model('models/keras-mape-model.h5', custom_objects={
                'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
                'k_mae_mape': k_mae_mape,
            }),
            'mae_model': load_model('models/keras-mae-model.h5', custom_objects={
                'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
                'k_mae_mape': k_mae_mape,
            }),
            'mae_intermediate_model': load_model('models/keras-mae-intermediate-model.h5', custom_objects={
                'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
                'k_mae_mape': k_mae_mape,
            }),
        }

    if 'train_xgb' in run_config and run_config['train_xgb'] == True:
        # Train the general models
        gen_models = train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y,
                                         df_all_test_x, keras_models)
    else:
        print('Loading xgboost models')
        # Load the general models
        gen_models = {
            'log_y': load('models/xgb-log_y.model.gz'),
            'keras_mae': load('models/xgb-keras_mae.model.gz'),
            'keras_log_mae': load('models/xgb-keras_log_mae.model.gz'),
        }


    # Export data prior to bagging
    train_predictions, test_predictions = export_final_data(df_all_train_x, df_all_train_actuals, df_all_test_x,
                                                            df_all_test_actuals, gen_models, keras_models)

    if 'train_deep_bagging' in run_config and run_config['train_deep_bagging'] == True:
        bagging_model, bagging_scaler, deep_bagged_predictions = train_deep_bagging(train_predictions,
                                                                                    df_all_train_actuals,
                                                                                    test_predictions,
                                                                                    df_all_test_actuals)
    else:
        print('Loading bagging models')
        bagging_model = load_model('models/keras-bagging-model.h5')
        bagging_scaler = load('models/deep-bagging-scaler.pkl.gz')
        deep_bagged_predictions = execute_deep_bagging(bagging_model, bagging_scaler, test_predictions)


    final_bagging(df_all_test_x, df_all_test_actuals, gen_models, keras_models, deep_bagged_predictions)


if __name__ == "__main__":
    run_config = {
        'load_data': False,
        'train_pre_process': False,
        'execute_pre_process': False,
        'train_keras': False,
        'train_xgb': True,
        'train_deep_bagging': True,
    }

    main(run_config)