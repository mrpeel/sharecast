from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
import sys
import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import numba
import xgboost as xgb
import gc
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# from categorical_encoder import *
from eval_results import eval_results, range_results
from print_logger import *
from clr_callback import *
# from autoencoder import *
from compile_keras import *
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
import os
from pathlib import Path
import matplotlib.pyplot as plt
import math

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

LABEL_COLUMN = 'future_eight_week_return'
RETURN_COLUMN = 'eight_week_total_return'

CATEGORICAL_COLUMNS = ['symbol_encoded', 'quoteDate_YEAR',
                       'quoteDate_MONTH', 'quoteDate_DAY', 'quoteDate_DAYOFWEEK']

CONTINUOUS_COLUMNS = ['lastTradePriceOnly', 'adjustedPrice', 'quoteDate_TIMESTAMP', 'volume', 'previousClose',
                      'change', 'changeInPercent',
                      '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
                      'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'allordpreviousclose',
                      'allordchange', 'allorddayshigh', 'allorddayslow', 'allordpercebtChangeFrom52WeekHigh',
                      'allordpercentChangeFrom52WeekLow', 'asxpreviousclose', 'asxchange', 'asxdayshigh',
                      'asxdayslow', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow',
                      'exDividendRelative', 'exDividendPayout', '640106_A3597525W', 'AINTCOV', 'Beta',
                      'BookValuePerShareYear', 'CashPerShareYear', 'DPSRecentYear', 'EPS', 'FIRMMCRT', 'FXRUSD',
                      'Float', 'GRCPAIAD', 'GRCPAISAD', 'GRCPBCAD', 'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD',
                      'H01_GGDPCVGDP', 'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP', 'MarketCap', 'OperatingMargin', 'PE',
                      'QuoteLast', 'ReturnOnEquityYear', 'TotalDebtToEquityYear', 'daysHigh', 'daysLow']


PAST_RESULTS_CONTINUOUS_COLUMNS = ['one_week_min', 'one_week_max', 'one_week_mean', 'one_week_median', 'one_week_std',
                                   'one_week_bollinger_upper', 'one_week_bollinger_lower',
                                   'one_week_comparison_adjustedPrice', 'one_week_price_change',
                                   'one_week_price_return',
                                   'one_week_dividend_value', 'one_week_dividend_return',
                                   'one_week_total_return', 'two_week_min', 'two_week_max',
                                   'two_week_mean', 'two_week_median', 'two_week_std',
                                   'two_week_bollinger_upper', 'two_week_bollinger_lower',
                                   'two_week_comparison_adjustedPrice',
                                   'two_week_price_change', 'two_week_price_return',
                                   'two_week_dividend_value', 'two_week_dividend_return',
                                   'two_week_total_return', 'four_week_min', 'four_week_max',
                                   'four_week_mean', 'four_week_median', 'four_week_std',
                                   'four_week_bollinger_upper', 'four_week_bollinger_lower',
                                   'four_week_comparison_adjustedPrice',
                                   'four_week_price_change', 'four_week_price_return',
                                   'four_week_dividend_value', 'four_week_dividend_return',
                                   'four_week_total_return', 'eight_week_min', 'eight_week_max',
                                   'eight_week_mean', 'eight_week_median', 'eight_week_std',
                                   'eight_week_bollinger_upper', 'eight_week_bollinger_lower',
                                   'eight_week_comparison_adjustedPrice',
                                   'eight_week_price_change',
                                   'eight_week_price_return', 'eight_week_dividend_value',
                                   'eight_week_dividend_return', 'eight_week_total_return',
                                   'twelve_week_min', 'twelve_week_max', 'twelve_week_mean',
                                   'twelve_week_median', 'twelve_week_std',
                                   'twelve_week_bollinger_upper', 'twelve_week_bollinger_lower',
                                   'twelve_week_comparison_adjustedPrice',
                                   'twelve_week_price_change',
                                   'twelve_week_price_return', 'twelve_week_dividend_value',
                                   'twelve_week_dividend_return', 'twelve_week_total_return',
                                   'twenty_six_week_min', 'twenty_six_week_max',
                                   'twenty_six_week_mean', 'twenty_six_week_median',
                                   'twenty_six_week_std', 'twenty_six_week_bollinger_upper',
                                   'twenty_six_week_bollinger_lower',
                                   'twenty_six_week_comparison_adjustedPrice',
                                   'twenty_six_week_price_change',
                                   'twenty_six_week_price_return', 'twenty_six_week_dividend_value',
                                   'twenty_six_week_dividend_return', 'twenty_six_week_total_return',
                                   'fifty_two_week_min', 'fifty_two_week_max', 'fifty_two_week_mean',
                                   'fifty_two_week_median', 'fifty_two_week_std',
                                   'fifty_two_week_bollinger_upper', 'fifty_two_week_bollinger_lower',
                                   'fifty_two_week_comparison_adjustedPrice',
                                   'fifty_two_week_price_change',
                                   'fifty_two_week_price_return', 'fifty_two_week_dividend_value',
                                   'fifty_two_week_dividend_return', 'fifty_two_week_total_return']

PAST_RESULTS_CATEGORICAL_COLUMNS = ['one_week_bollinger_type', 'one_week_bollinger_prediction',
                                    'two_week_bollinger_type', 'two_week_bollinger_prediction',
                                    'four_week_bollinger_type', 'four_week_bollinger_prediction',
                                    'eight_week_bollinger_type', 'eight_week_bollinger_prediction',
                                    'twelve_week_bollinger_type', 'twelve_week_bollinger_prediction',
                                    'twenty_six_week_bollinger_type', 'twenty_six_week_bollinger_prediction',
                                    'fifty_two_week_bollinger_type', 'fifty_two_week_bollinger_prediction']

RECURRENT_COLUMNS = ['asxpreviousclose_T11_20P', 'asxpreviousclose_T1P', 'asxpreviousclose_T2_5P',
                     'asxpreviousclose_T6_10P', 'asxpreviousclose_T11_20P', 'asxpreviousclose_T1P',
                     'asxpreviousclose_T2_5P', 'asxpreviousclose_T6_10P', 'allordpreviousclose_T11_20P',
                     'allordpreviousclose_T1P', 'allordpreviousclose_T2_5P', 'allordpreviousclose_T6_10P',
                     'adjustedPrice_T11_20P', 'adjustedPrice_T1P', 'adjustedPrice_T2_5P', 'adjustedPrice_T6_10P',
                     'FIRMMCRT_T11_20P', 'FIRMMCRT_T1P', 'FIRMMCRT_T2_5P', 'FIRMMCRT_T6_10P', 'FXRUSD_T11_20P',
                     'FXRUSD_T1P', 'FXRUSD_T2_5P', 'FXRUSD_T6_10P', 'GRCPAIAD_T11_20P', 'GRCPAIAD_T1P',
                     'GRCPAIAD_T2_5P', 'GRCPAIAD_T6_10P', 'GRCPAISAD_T1P', 'GRCPAISAD_T2_5P', 'GRCPAISAD_T6_10P',
                     'GRCPAISAD_T11_20P', 'GRCPBCAD_T1P', 'GRCPBCAD_T2_5P', 'GRCPBCAD_T6_10P', 'GRCPBCAD_T11_20P',
                     'GRCPBCSAD_T1P', 'GRCPBCSAD_T2_5P', 'GRCPBCSAD_T6_10P', 'GRCPBCSAD_T11_20P',
                     'GRCPBMAD_T1P', 'GRCPBMAD_T2_5P', 'GRCPBMAD_T6_10P', 'GRCPBMAD_T11_20P', 'GRCPNRAD_T1P',
                     'GRCPNRAD_T2_5P', 'GRCPNRAD_T6_10P', 'GRCPNRAD_T11_20P', 'GRCPRCAD_T1P', 'GRCPRCAD_T2_5P',
                     'GRCPRCAD_T6_10P', 'GRCPRCAD_T11_20P', 'H01_GGDPCVGDPFY_T1P', 'H01_GGDPCVGDPFY_T2_5P',
                     'H01_GGDPCVGDPFY_T6_10P', 'H01_GGDPCVGDPFY_T11_20P', 'H05_GLFSEPTPOP_T1P', 'H05_GLFSEPTPOP_T2_5P',
                     'H05_GLFSEPTPOP_T6_10P', 'H05_GLFSEPTPOP_T11_20P']

HIGH_NAN_COLUMNS = ['Price200DayAverage', 'Price52WeekPercChange', 'AverageVolume', 'EBITDMargin',
                    'EPSGrowthRate10Years', 'EPSGrowthRate5Years', 'IAD', 'LTDebtToEquityQuarter',
                    'LTDebtToEquityYear', 'NetIncomeGrowthRate5Years', 'NetProfitMarginPercent',
                    'PriceToBook', 'ReturnOnAssets5Years', 'ReturnOnAssetsTTM', 'ReturnOnAssetsYear',
                    'ReturnOnEquity5Years', 'ReturnOnEquityTTM', 'RevenueGrowthRate10Years',
                    'RevenueGrowthRate5Years', 'TotalDebtToAssetsQuarter', 'TotalDebtToAssetsYear',
                    'TotalDebtToEquityQuarter', 'bookValue', 'earningsPerShare', 'ebitda',
                    'epsEstimateCurrentYear', 'marketCapitalization', 'peRatio', 'pegRatio', 'pricePerBook',
                    'pricePerEpsEstimateCurrentYear', 'pricePerEpsEstimateNextYear', 'pricePerSales']

PAST_RESULTS_DATE_REF_COLUMNS = ['one_week_comparison_date', 'two_week_comparison_date', 'four_week_comparison_date',
                                 'eight_week_comparison_date', 'twelve_week_comparison_date',
                                 'twenty_six_week_comparison_date', 'fifty_two_week_comparison_date']

ALL_CONTINUOUS_COLUMNS = []
ALL_CONTINUOUS_COLUMNS.extend(CONTINUOUS_COLUMNS)
ALL_CONTINUOUS_COLUMNS.extend(PAST_RESULTS_CONTINUOUS_COLUMNS)
ALL_CONTINUOUS_COLUMNS.extend(RECURRENT_COLUMNS)

ALL_CATEGORICAL_COLUMNS = []
ALL_CATEGORICAL_COLUMNS.extend(CATEGORICAL_COLUMNS)
ALL_CATEGORICAL_COLUMNS.extend(PAST_RESULTS_CATEGORICAL_COLUMNS)

COLUMNS_TO_REMOVE = []
COLUMNS_TO_REMOVE.extend(HIGH_NAN_COLUMNS)
COLUMNS_TO_REMOVE.extend(PAST_RESULTS_DATE_REF_COLUMNS)

column_types = {
    'symbol': 'category',
    'one_week_bollinger_type': 'category',
    'one_week_bollinger_prediction': 'category',
    'two_week_bollinger_type': 'category',
    'two_week_bollinger_prediction': 'category',
    'four_week_bollinger_type': 'category',
    'four_week_bollinger_prediction': 'category',
    'eight_week_bollinger_type': 'category',
    'eight_week_bollinger_prediction': 'category',
    'twelve_week_bollinger_type': 'category',
    'twelve_week_bollinger_prediction': 'category',
    'twenty_six_week_bollinger_type': 'category',
    'twenty_six_week_bollinger_prediction': 'category',
    'fifty_two_week_bollinger_type': 'category',
    'fifty_two_week_bollinger_prediction': 'category',
}


xgb_set_path = './models/xgb-sets/'


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
    return num - (num % divisor)


# @profile
@numba.jit
def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log1p(np.absolute(return_vals))
    return_vals[neg_mask] *= -1.
    return return_vals

# @profile


@numba.jit
def safe_exp(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.expm1(np.clip(np.absolute(return_vals), -7, 7))
    return_vals[neg_mask] *= -1.
    return return_vals


@numba.jit
def y_scaler(input_array):
    transformed_array = safe_log(input_array)
    scaler = MaxAbsScaler()
    # transformed_array = scaler.fit_transform(transformed_array)
    return transformed_array, scaler


@numba.jit
def y_inverse_scaler(prediction_array):
    # scaler.inverse_transform(prediction_array)
    transformed_array = prediction_array
    transformed_array = safe_exp(transformed_array)
    return transformed_array


# @profile
@numba.jit
def mle(actual_y, prediction_y):
    """
    Compute the Root Mean  Log Error

    Args:
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
    """

    return np.mean(np.absolute(safe_log(prediction_y) - safe_log(actual_y)))


# @profile
@numba.jit
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
@numba.jit
def mae_eval(y, y0):
    y0 = y0.get_label()
    assert len(y) == len(y0)
    # return 'error', np.sqrt(np.mean(np.square(np.log(y + 1) - np.log(y0 + 1))))
    return 'error', np.mean(np.absolute(y - y0)), False


# @profile
@numba.jit
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
    diff = np.absolute((actual_y - prediction_y) /
                       np.clip(np.absolute(actual_y), 1., None))
    return 100. * np.mean(diff)


# @profile
@numba.jit
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
@numba.jit
def mape_log_y(actual_y, prediction_y):
    inverse_actual = actual_y.copy()
    inverse_actual = y_inverse_scaler(inverse_actual)

    inverse_prediction = prediction_y.copy()
    inverse_prediction = y_inverse_scaler(inverse_prediction)

    return safe_mape(inverse_actual, inverse_prediction)


# @profile
@numba.jit
def mape_log_y_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', mape_log_y(actual_y, prediction_y)


@numba.jit
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
@numba.jit
def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    return 100. * K.mean(diff, axis=-1)


# @profile
@numba.jit
def drop_unused_columns(df, data_cols):
    # Check for columns to drop
    print('Keeping columns:', list(data_cols))
    cols_to_drop = []
    for col in df.columns:
        if col not in data_cols:
            cols_to_drop.append(col)

    print('Dropping columns:', list(cols_to_drop))
    df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

    return df


# @profile
@numba.jit
def convert_date(df, column_name):
    df[column_name + "_TIMESTAMP"] = (pd.DatetimeIndex(
        df[column_name]) - pd.datetime(2007, 1, 1)).total_seconds()
    df[column_name + "_TIMESTAMP"] = df[column_name +
                                        "_TIMESTAMP"].astype('int32', errors='ignore')

    df[column_name +
        "_YEAR"] = pd.DatetimeIndex(df[column_name]).year.astype('str')
    df[column_name + "_YEAR"] = df[column_name +
                                   "_YEAR"].astype('int32', errors='ignore')

    df[column_name +
        "_MONTH"] = pd.DatetimeIndex(df[column_name]).month.astype('str')
    df[column_name + "_MONTH"] = df[column_name +
                                    "_MONTH"].astype('int32', errors='ignore')

    df[column_name +
        "_DAY"] = pd.DatetimeIndex(df[column_name]).day.astype('str')
    df[column_name + "_DAY"] = df[column_name +
                                  "_DAY"].astype('int32', errors='ignore')

    df[column_name +
        "_DAYOFWEEK"] = pd.DatetimeIndex(df[column_name]).dayofweek.astype('str')
    df[column_name + "_DAYOFWEEK"] = df[column_name +
                                        "_DAYOFWEEK"].astype('int32', errors='ignore')


# @profile
# def setup_data_columns(df):
#     # # Remove columns not referenced in either algorithm
#     # columns_to_keep = [LABEL_COLUMN, 'quoteDate', 'exDividendDate']
#     #
#     # # Add continuous and categorical columns
#     # columns_to_keep.extend(CONTINUOUS_COLUMNS)
#     # columns_to_keep.extend(CATEGORICAL_COLUMNS)
#     # columns_to_keep.extend(FUTURE_RESULTS_COLUMNS)
#     # Drop columns not in keep list
#
#     columns_to_keep = []
#     # Keep columns not in high nan list
#     for col in df.columns.values:
#         if col not in HIGH_NAN_COLUMNS:
#             columns_to_keep.append(col)
#
#
#     return_df = drop_unused_columns(df, columns_to_keep)
#     return return_df

@numba.jit
def generate_label_column(df, num_weeks, reference_date, date_col):
    """ Generate a label column for each record num_weeks in the future
        then reduce data set to values which occurr <= (reference_date - num_weeks)
     """

    # Retrieve unique symbols
    symbols = df['symbol'].drop_duplicates()
    symbol_counter = 0

    date_ref_col = date_col + '_ref'

    # Array to hold completed dataframes
    symbol_dfs = []

    # ### TEMP DEBUG ##############
    # symbols = symbols.head(10)
    # ####################

    # Create empty data frame
    # output_df = pd.DataFrame()

    for symbol in symbols:
        gc.collect()
        symbol_counter = symbol_counter + 1
        # print(80 * '-')
        print('Generating labels for:', symbol, 'num:', symbol_counter)
        # Retrieve data for symbol and sort by date
        symbol_df = df.loc[df['symbol'] == symbol, :]
        symbol_df.sort_values(by=['quoteDate'], inplace=True)

        # Set the date index for the data frame
        symbol_df[date_ref_col] = symbol_df[date_col]
        symbol_df = symbol_df.set_index(date_ref_col)

        comparison_date = symbol_df.index + pd.DateOffset(weeks=num_weeks)

        # Get the future result values for number of weeks
        ref_vals_df = pd.DataFrame()
        # Create offset for number of weeks (this sets the index forwards as well)
        ref_vals_df[LABEL_COLUMN] = symbol_df[RETURN_COLUMN].asof(
            comparison_date)
        ref_vals_df[LABEL_COLUMN + '_date'] = comparison_date

        # Reset the index value back to original dates
        ref_vals_df.index = symbol_df.index

        # concatenate the offset values with the original values
        combined_vals = pd.concat([symbol_df, ref_vals_df], axis=1)

        # Append dataframe to dataframe array
        symbol_dfs.append(combined_vals)

    # Generate concatenated dataframe
    output_df = pd.concat(symbol_dfs)

    # Process data set to remove records which are too recent to have a future value

    # Ensure reference is a date/time
    converted_ref_date = datetime.strptime(reference_date, '%Y-%m-%d')
    # filter dataframe
    output_df = output_df.loc[output_df[LABEL_COLUMN +
                                        '_date'] <= converted_ref_date]
    # Remove extra column with future date reference
    output_df.drop([LABEL_COLUMN + '_date'], axis=1, inplace=True)

    # Use most efficient storage for memory
    output_df.loc[:, LABEL_COLUMN] = output_df[LABEL_COLUMN].astype(
        'float32', errors='ignore')

    return output_df


@numba.jit
def remove_labelled_data(df, num_weeks, reference_date):
    """
        Remove all data which is before the referenece period.
        Keep quotes > (reference_date - num_weeks)
    """

    print('Removing labelled data, reference_date:',
          reference_date, ', num_weeks:', num_weeks)

    output_df = df
    weeks_delta = timedelta(weeks=num_weeks)
    converted_ref_date = datetime.strptime(reference_date, '%Y-%m-%d')
    comparison_date = converted_ref_date - weeks_delta
    print('Calculated comparison date:', comparison_date)

    output_df = output_df.loc[output_df.index > comparison_date]
    print('Retaining', len(output_df.index), 'records')

    return output_df


# @profile
def load_data(file_name, **kwargs):
    """Load pickled data and run combined prep
        Arguments:
        file_name
        drop_unlabelled=True
        drop_labelled=False
        generate_labels=False
        label_weeks=None
        reference_date=None
        labelled_file_name=None
        unlabelled_file_name=None
    """
    print('Loading file:', file_name)

    drop_unlabelled = kwargs.get('drop_unlabelled', True)
    generate_labels = kwargs.get('generate_labels', False)
    drop_labelled = kwargs.get('drop_labelled', False)
    label_weeks = kwargs.get('label_weeks', None)
    reference_date = kwargs.get('reference_date', None)
    labelled_file_name = kwargs.get('labelled_file_name', None)
    unlabelled_file_name = kwargs.get('unlabelled_file_name', None)

    df = pd.read_pickle(file_name, compression='gzip')
    gc.collect()

    print(df.info(memory_usage='deep'))

    # Convert dates to correct type
    print('Converting dates to datetime types')
    df['quoteDate'] = pd.to_datetime(df['quoteDate'])
    df['exDividendDate'] = pd.to_datetime(
        df['exDividendDate'], errors='coerce')

    # Remove columns which should not be used for calculations (ignore errors if already removed)
    print('Dropping columns:', COLUMNS_TO_REMOVE)
    df.drop(COLUMNS_TO_REMOVE, axis=1, inplace=True, errors='ignore')

    # Reset dividend date as a number
    print('Making ex-dividend date a relative number')
    df['exDividendRelative'] = df['exDividendDate'] - df['quoteDate']

    # convert string difference value to integer
    df['exDividendRelative'] = df['exDividendRelative'].apply(
        lambda x: np.nan if pd.isnull(x) else x.days)
    # Make sure it is the minimum data type size
    df.loc[:, 'exDividendRelative'] = df['exDividendRelative'].astype(
        'int32', errors='ignore')

    # generate labels if required
    if generate_labels and label_weeks and reference_date:
        df = generate_label_column(
            df, label_weeks, reference_date, 'quoteDate')
        gc.collect()
        # save as new file
        if labelled_file_name:
            df.to_pickle(labelled_file_name, compression='gzip')

    # Remobe labelled data
    if drop_labelled and label_weeks and reference_date:
        df = remove_labelled_data(
            df, label_weeks, reference_date)
        gc.collect()

        # save as new file
        if unlabelled_file_name:
            df.to_pickle(unlabelled_file_name, compression='gzip')

    print('Converting quoteDate to numeric types')
    convert_date(df, 'quoteDate')

    # Remove date columns
    print('Removing date columns')
    df.drop(['quoteDate', 'exDividendDate'], axis=1, inplace=True)

    print(df.info(memory_usage='deep'))

    if drop_unlabelled is True:
        # Drop any row which does not have the label columns
        print('Dropping rows missing the label column')
        # df.dropna(subset=[LABEL_COLUMN], inplace=True)
        df = df[np.isfinite(df[LABEL_COLUMN].values)]

        # Clip to -99 to 1000 range
        print('Clipping label column to (-99, 1000)')
        df[LABEL_COLUMN] = df[LABEL_COLUMN].clip(-99, 1000)

        # Add scaled value for y - using log of y
        print('Creating scaled log label column')
        df[LABEL_COLUMN + '_scaled'] = safe_log(df[LABEL_COLUMN].values)

    return df


def column_stats(df):
    # Get number of vals in dataframe
    num_vals = df.shape[0]
    interpolate_cols = []

    print('Column stats')

    for col in df.columns.values:
        if df[col].dtype == 'float64':
            # Perform stats
            num_null_vals = df[col].isnull().sum()
            perc_null_vals = num_null_vals / num_vals * 100
            if perc_null_vals > 20:
                interpolate_cols.append(col)
                print(col)
                print('Percentage of null vals:', perc_null_vals)
                print('Min val:', df[col].min())
                print('Max val:', df[col].max())
                print('Mean:', df[col].mean())
                print('Median:', df[col].median())
                print('Std dev:', df[col].std())
                print(20 * '-')

    df.interpolate(method='slinear', inplace=True)
    print(50 * '-')
    for col in interpolate_cols:
        num_null_vals = df[col].isnull().sum()
        perc_null_vals = num_null_vals / num_vals * 100
        print(col)
        print('Percentage of null vals:', perc_null_vals)
        print('Min val:', df[col].min())
        print('Max val:', df[col].max())
        print('Mean:', df[col].mean())
        print('Median:', df[col].median())
        print('Std dev:', df[col].std())
        print(20 * '-')


# @profile
def divide_data(share_data):
    symbols = share_data['symbol'].unique()
    # For testing only take the first 10 elements
    # symbols = symbols[:10]
    ##########
    symbol_map = {}
    symbol_num = 0

    symbol_models = []

    print('No of symbols:', len(symbols))

    # Array to hold completed dataframes
    train_x_dfs = []
    train_y_dfs = []
    train_actuals_dfs = []
    test_x_dfs = []
    test_y_dfs = []
    test_actuals_dfs = []

    # symbols_train_x = {}
    # symbols_train_y = {}
    # symbols_train_actuals = {}
    # symbols_test_x = {}
    # symbols_test_y = {}
    # symbols_test_actuals = {}

    # Shuffle symbols into random order
    np.random.shuffle(symbols)

    # prep data for fitting into both model types
    for symbol in symbols:
        gc.collect()

        # Filter to model data for this symbol and re-set the pandas indexes
        model_data = share_data.loc[share_data['symbol'] == symbol]

        print('Symbol:', symbol, 'num:', symbol_num,
              'number of records:', len(model_data))

        msk = np.random.rand(len(model_data)) < 0.8

        # Prep dataframes and reset index for appending
        df_train = model_data[msk]
        df_test = model_data[~msk]
        # df_train.reset_index()
        # df_test.reset_index()

        # Make sure a minimum number of rows are present in sample for symbol
        if (len(df_train) > 50):
            # Check whether this will have its own model or the generic model
            if (len(df_train) > 150):
                df_train.loc[:, 'model'] = df_train.loc[:, 'symbol']
                df_test.loc[:, 'model'] = df_test.loc[:, 'symbol']
                symbol_models.append(symbol)
            else:
                df_train['model'] = 'generic'
                df_test['model'] = 'generic'

            train_y_dfs.append(df_train[LABEL_COLUMN + '_scaled'])
            train_actuals_dfs.append(df_train[LABEL_COLUMN])
            train_x_dfs.append(df_train.drop(
                [LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

            test_y_dfs.append(df_test[LABEL_COLUMN + '_scaled'])
            test_actuals_dfs.append(df_test[LABEL_COLUMN])
            test_x_dfs.append(df_test.drop(
                [LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

            # Set up map of symbol name to number
            symbol_map[symbol] = symbol_num

        symbol_num += 1

        # ### TEMP DEBUG ##############
        # if symbol_num >= 100:
        #     break

    # Create concatenated dataframes with all data
    print('Creating concatenated dataframes')

    df_all_test_y = pd.concat(test_y_dfs)
    del test_y_dfs
    gc.collect()

    df_all_test_actuals = pd.concat(test_actuals_dfs)
    del test_actuals_dfs
    gc.collect()

    df_all_train_y = pd.concat(train_y_dfs)
    del train_y_dfs
    gc.collect()

    df_all_train_actuals = pd.concat(train_actuals_dfs)
    del train_actuals_dfs
    gc.collect()

    df_all_test_x = pd.concat(test_x_dfs)
    del test_x_dfs
    gc.collect()

    df_all_train_x = pd.concat(train_x_dfs)
    del train_x_dfs
    gc.collect()

    print(symbol_map)

    # Write symbol models array to file
    print('Saving symbol-models-list')
    save(symbol_models, 'models/symbol-models-list.pkl.gz')

    # return symbol_map, symbols_train_y, symbols_train_actuals, symbols_train_x, symbols_test_actuals, \
    #        symbols_test_y, symbols_test_x, df_all_train_y, df_all_train_actuals, df_all_train_x, \
    #        df_all_test_actuals, df_all_test_y, df_all_test_x

    return symbol_map, df_all_train_y, df_all_train_actuals, df_all_train_x, df_all_test_actuals, df_all_test_y, df_all_test_x


# def train_one_hot_string_encoder(df, cols):
#     gs = Smarties()

#     # Create one hot encoded columns
#     new_cols = gs.fit_transform(df[cols])

#     # Drop original value columns
#     df.drop(cols, axis=1, inplace=True, errors='ignore')
#     df = pd.concat([df, new_cols])

#     # Return dataframe and encoder
#     return df, gs


def execute_one_hot_string_encoder(df, cols):  # , gs):
    # Create one hot encoded columns
    new_cols = pd.get_dummies(df[cols])
    new_cols = new_cols.astype('int8', errors='ignore')
    col_names = new_cols.columns.values

    # Drop original value columns
    df.drop(cols, axis=1, inplace=True, errors='ignore')

    for col in col_names:
        df[col] = new_cols[col].astype('int8', errors='ignore')

    # Return dataframe
    return df


def train_symbol_encoder(df):
    # Calculate risk adjusted return
    temp_df = pd.DataFrame()

    # Copy values to new dataframe
    print('Copying return data')
    temp_df[
        ['symbol', 'adjustedPrice', 'eight_week_price_return', 'eight_week_dividend_return', 'eight_week_total_return',
         'eight_week_std']] = df[
        ['symbol', 'adjustedPrice', 'eight_week_price_return', 'eight_week_dividend_return', 'eight_week_total_return',
         'eight_week_std']].dropna()
    less_than_zero = temp_df['eight_week_total_return'] < 0

    # Calculate percentage volatility
    print('Calculating volatility')
    temp_df['eight_week_volatility_perc'] = temp_df['eight_week_std'] / \
        temp_df['adjustedPrice'] * 100

    # Convert to multiplication factor to adjust return
    print('Calculating multiplication factors')
    temp_df['mult_factor'] = 1 - \
        (temp_df['eight_week_std'] / temp_df['adjustedPrice'])
    temp_df.loc[less_than_zero, 'mult_factor'] = 1 + (
        temp_df.loc[less_than_zero, 'eight_week_std'] / temp_df.loc[less_than_zero, 'adjustedPrice'])

    # Calulate risk adjusted returns
    print('Calculating risk adjusted returns')
    temp_df['ra_price_return'] = temp_df['eight_week_price_return'] * \
        temp_df['mult_factor']
    temp_df['ra_total_return'] = temp_df['ra_price_return'] + \
        temp_df['eight_week_dividend_return']

    se_lookup = pd.DataFrame(temp_df.groupby(['symbol'])[
        'ra_total_return'].mean().reset_index(name="symbol_encoded"))

    ret_df = execute_symbol_encoder(df, se_lookup)

    return ret_df, se_lookup


def execute_symbol_encoder(df, se_df):
    # Merge symbol encodeded values
    print('Merging encoded symbol with dataframe')
    ret_df = df.merge(se_df, left_on='symbol', right_on='symbol')

    # Remove symbol column
    print('Dropping symbol column')
    ret_df.drop(['symbol'], axis=1, inplace=True)

    # Impute any missing values for symbol (can happen when predictions include new symbols)
    ret_df['symbol_encoded'].fillna(
        (ret_df['symbol_encoded'].median()), inplace=True)

    return ret_df


def train_imputer(df):
    print('Training imputer')
    imputer = [Imputer(strategy='median'), Imputer(
        strategy='median'), Imputer(strategy='median')]
    print(' continous columns')
    imputer[0].fit(df[CONTINUOUS_COLUMNS].values)
    print(' past results continous columns')
    imputer[1].fit(df[PAST_RESULTS_CONTINUOUS_COLUMNS].values)
    print(' recurrent columns')
    imputer[2].fit(df[RECURRENT_COLUMNS].values)

    ret_df = execute_imputer(df, imputer)

    return ret_df, imputer


def execute_imputer(df, imputer):
    print('Executing imputer')
    ret_df = df
    print(' continuous columns')
    ret_df[CONTINUOUS_COLUMNS] = imputer[0].transform(
        ret_df[CONTINUOUS_COLUMNS].values)
    ret_df[CONTINUOUS_COLUMNS] = ret_df[CONTINUOUS_COLUMNS].astype(
        'float32', errors='ignore')
    print(' past results continous columns')
    ret_df[PAST_RESULTS_CONTINUOUS_COLUMNS] = imputer[1].transform(
        ret_df[PAST_RESULTS_CONTINUOUS_COLUMNS].values)
    ret_df[PAST_RESULTS_CONTINUOUS_COLUMNS] = ret_df[PAST_RESULTS_CONTINUOUS_COLUMNS].astype(
        'float32', errors='ignore')
    print(' recurrent columns')
    ret_df[RECURRENT_COLUMNS] = imputer[2].transform(
        ret_df[RECURRENT_COLUMNS].values)
    ret_df[RECURRENT_COLUMNS] = ret_df[RECURRENT_COLUMNS].astype(
        'float32', errors='ignore')

    return ret_df


def train_scaler(df):
    print('Training scaler')
    scaler = [MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1)),
              MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))]
    print(' continuous columns')
    scaler[0].fit(df[CONTINUOUS_COLUMNS].values)
    print(' past results continuous columns')
    scaler[1].fit(df[PAST_RESULTS_CONTINUOUS_COLUMNS].values)
    print(' recurrent columns')
    scaler[2].fit(df[RECURRENT_COLUMNS].values)
    print(' categorical columns')
    scaler[3].fit(df[CATEGORICAL_COLUMNS].values)

    ret_df = execute_scaler(df, scaler)

    return ret_df, scaler


def execute_scaler(df, scaler):
    print('Executing scaler')
    print(' continuous columns')
    df[CONTINUOUS_COLUMNS] = scaler[0].transform(df[CONTINUOUS_COLUMNS].values)
    df[CONTINUOUS_COLUMNS] = df[CONTINUOUS_COLUMNS].astype(
        'float32', errors='ignore')

    print(' past results continuous columns')
    df[PAST_RESULTS_CONTINUOUS_COLUMNS] = scaler[1].transform(
        df[PAST_RESULTS_CONTINUOUS_COLUMNS].values)
    df[PAST_RESULTS_CONTINUOUS_COLUMNS] = df[PAST_RESULTS_CONTINUOUS_COLUMNS].astype(
        'float32', errors='ignore')

    print(' recurrent columns')
    df[RECURRENT_COLUMNS] = scaler[2].transform(df[RECURRENT_COLUMNS].values)
    df[RECURRENT_COLUMNS] = df[RECURRENT_COLUMNS].astype(
        'float32', errors='ignore')

    print(' categorical columns')
    df[CATEGORICAL_COLUMNS] = scaler[3].transform(
        df[CATEGORICAL_COLUMNS].values)
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].astype(
        'float32', errors='ignore')

    return df


def train_preprocessor(train_x_df, train_y_df):
    print('Training pre-processor...')

    print('One hot encoding past results categorical columns')
    train_x_df = execute_one_hot_string_encoder(
        train_x_df, PAST_RESULTS_CATEGORICAL_COLUMNS)
    gc.collect()

    print('Encoding symbol values')
    train_x_df, se = train_symbol_encoder(train_x_df)
    gc.collect()

    print('Imputing missing values')
    train_x_df, imputer = train_imputer(train_x_df)
    gc.collect()

    print('Scaling data')
    train_x_df, scaler = train_scaler(train_x_df)
    gc.collect()

    # print('Encoding categorical data...')
    # # Use categorical entity embedding encoder
    #
    # ce = Categorical_encoder(strategy="entity_embedding", verbose=True)
    # # Transform everything except the model name
    # df_train_transform = ce.fit_transform(train_x_df, train_y_df[0])

    # Write one hot encoder, symbol encoder, scaler and categorical encoder to files
    save(se, 'models/se.pkl.gz')
    save(imputer, 'models/imputer.pkl.gz')
    save(scaler, 'models/scaler.pkl.gz')
    # save(ce, 'models/ce.pkl.gz')

    return train_x_df, se, imputer, scaler


def execute_preprocessor(transform_df, se, imputer, scaler):
    print('Executing pre-processor on supplied data...')

    print('One hot encoding past results categorical columns')
    transform_df = execute_one_hot_string_encoder(
        transform_df, PAST_RESULTS_CATEGORICAL_COLUMNS)
    gc.collect()

    print('Encoding symbol values')
    transform_df = execute_symbol_encoder(transform_df, se)
    gc.collect()

    print('Imputing missing values')
    transform_df = execute_imputer(transform_df, imputer)
    gc.collect()

    print('Remove any remaining columns with nan values')
    transform_df.dropna(inplace=True)

    print('Scaling data...')
    transform_df = execute_scaler(transform_df, scaler)
    gc.collect()

    # print('Encoding categorical data...')
    # # Use categorical entity embedding encoder
    # transform_df = ce.transform(transform_df)

    return transform_df


# @profile
def load_xgb_models():
    all_xgb_models = {}

    print('Loading files from', xgb_set_path)
    # Return files in path
    file_list = glob.glob(xgb_set_path + '*.model.gz')

    # load each model set
    for file_name in file_list:
        # remove leading path and trailing file extension
        model_name = file_name.replace(
            xgb_set_path, '').replace('.model.gz', '')

        # create model property and load model set into it
        all_xgb_models[model_name] = file_name

    return all_xgb_models


# @profile
def train_xgb_models(df_all_train_x, df_all_train_y, train_x_model_names, test_x_model_names,
                     df_all_test_actuals, df_all_test_y, df_all_test_x, keras_models):
    # clear previous models
    files = glob.glob(xgb_set_path + '*.model.gz')
    for f in files:
        os.remove(f)

    # Retrieve model name list
    model_names = np.unique(train_x_model_names)
    print('Number of xgb model sets to train:', len(model_names))

    for model in model_names:
        # Retrieve indices for data values which have this model name
        train_index = np.where(train_x_model_names == model)[0]
        test_index = np.where(test_x_model_names == model)[0]

        # Retrieve only the data matching this model
        model_train_x = df_all_train_x.iloc[train_index, :]
        model_train_y = df_all_train_y.iloc[train_index]
        model_test_x = df_all_test_x.iloc[test_index, :]
        model_test_y = df_all_test_y.iloc[test_index]
        model_test_actuals = df_all_test_actuals.iloc[test_index]

        xgb_model_set = train_xgb_model_set(model, model_train_x, model_train_y, model_test_actuals,
                                            model_test_y, model_test_x, keras_models)

        # Save model set
        print('Saving model set for ', model)
        save(xgb_model_set, xgb_set_path + model + '.model.gz')


# @profile
def train_xgb_model_set(model_set_name, df_all_train_x, df_all_train_y, df_all_test_actuals,
                        df_all_test_y, df_all_test_x, keras_models):
    # Train gxb models for a symbol

    tree_method = 'auto'
    predictor = 'cpu_predictor'
    nthread = 8

    print('-' * 80)
    print('xgboost ' + model_set_name)
    print('-' * 80)

    # Create model
    log_y_model = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method, predictor=predictor,
                                   n_estimators=150, max_depth=70, base_score=0.1, colsample_bylevel=0.7,
                                   colsample_bytree=1.0, gamma=0, learning_rate=0.05, min_child_weight=3)

    all_train_y = df_all_train_y.values
    all_train_log_y = safe_log(all_train_y)
    all_train_x = df_all_train_x.values
    all_test_actuals = df_all_test_actuals.values
    all_test_y = df_all_test_y.values
    all_test_x = df_all_test_x.values
    all_test_log_y = safe_log(all_test_y)

    print('Training xgboost log of y model for', model_set_name)
    x_train, x_test, y_train, y_test = train_test_split(
        all_train_x, all_train_y, test_size=0.15)

    eval_set = [(x_test, y_test)]
    log_y_model.fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae', eval_set=eval_set,
                    verbose=True)

    # Save, delete and reload model to clear memory when using GPU
    print('Saving xgboost log of y model...')
    save(log_y_model, './temp/xgb-log-y.model.gz')
    print('Deleting xgboost log of y model...')
    del log_y_model
    print('Reloading xgboost log of y model...')
    log_y_model = load('./temp/xgb-log-y.model.gz')

    gc.collect()

    # output feature importances
    # print(log_y_model.feature_importances_)

    predictions = log_y_model.predict(all_test_x)
    inverse_scaled_predictions = safe_exp(predictions)

    eval_results({model_set_name + '_xgboost_mae': {
        'log_y': all_test_y,
        'actual_y': all_test_actuals,
        'log_y_predict': predictions,
        'y_predict': inverse_scaled_predictions
    }
    })

    print('Retrieving keras intermediate model vals...')
    mae_vals_train = keras_models['mae_intermediate_model'].predict(
        all_train_x)
    mae_vals_test = keras_models['mae_intermediate_model'].predict(all_test_x)

    stacked_vals_train = np.column_stack([all_train_x, mae_vals_train])
    stacked_vals_test = np.column_stack([all_test_x, mae_vals_test])

    print('Training xgboost log of y with keras outputs model for', model_set_name)
    keras_mae_model = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method, predictor=predictor,
                                       n_estimators=150, max_depth=70, learning_rate=0.05, base_score=0.25,
                                       colsample_bylevel=0.4, colsample_bytree=0.55, gamma=0, min_child_weight=0)

    x_train, x_test, y_train, y_test = train_test_split(
        stacked_vals_train, all_train_y, test_size=0.15)

    eval_set = [(x_test, y_test)]
    keras_mae_model.fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae',
                        eval_set=eval_set, verbose=True)

    # Save, delete and reload model to clear memory when using GPU
    print('Saving xgboost log of y with keras outputs model...')
    save(keras_mae_model, './temp/xgb-keras-mae.model.gz')
    print('Deleting xgboost log of y with keras outputs model...')
    del keras_mae_model
    print('Reloading xgboost log of y with keras outputs model...')
    keras_mae_model = load('./temp/xgb-keras-mae.model.gz')

    gc.collect()

    # output feature importances
    # print(keras_mae_model.feature_importances_)

    keras_log_predictions = keras_mae_model.predict(stacked_vals_test)
    # ### Double exp #######
    keras_inverse_scaled_predictions = safe_exp(keras_log_predictions)

    eval_results({model_set_name + 'xgboost_keras': {
        'log_y': all_test_y,
        'actual_y': all_test_actuals,
        'log_y_predict': keras_log_predictions,
        'y_predict': keras_inverse_scaled_predictions
    }
    })

    print('Training xgboost log of log of y with keras outputs model...')
    keras_log_mae_model = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method, predictor=predictor,
                                           n_estimators=150, max_depth=130, base_score=0.4, colsample_bylevel=0.4,
                                           colsample_bytree=0.4, gamma=0, min_child_weight=0, learning_rate=0.05)

    x_train, x_test, y_train, y_test = train_test_split(
        stacked_vals_train, all_train_log_y, test_size=0.15)

    eval_set = [(x_test, y_test)]
    keras_log_mae_model.fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae',
                            eval_set=eval_set, verbose=True)

    # Save, delete and reload model to clear memory when using GPU
    print('Saving xgboost log of log of y with keras outputs model...')
    save(keras_log_mae_model, './temp/xgb-keras-log-mae.model.gz')
    print('Deleting xgboost log of log of y with keras outputs model...')
    del keras_log_mae_model
    print('Reloading xgboost log of log of y with keras outputs model...')
    keras_log_mae_model = load('./temp/xgb-keras-log-mae.model.gz')

    gc.collect()

    # output feature importances
    # print(keras_log_mae_model.feature_importances_)

    keras_log_log_predictions = keras_log_mae_model.predict(stacked_vals_test)
    # ### Double exp #######
    keras_log_inverse_scaled_predictions = safe_exp(
        safe_exp(keras_log_log_predictions))

    eval_results({model_set_name + 'xgboost_keras_log_y': {
        'actual_y': all_test_actuals,
        'y_predict': keras_log_inverse_scaled_predictions
    }
    })

    range_results({
        'xgboost_mae': inverse_scaled_predictions,
        'xgboost_keras_mae': keras_inverse_scaled_predictions,
        'xgboost_keras_log_mae': keras_log_inverse_scaled_predictions
    }, all_test_actuals)

    return {
        'log_y_model': log_y_model,
        'keras_mae_model': keras_mae_model,
        'keras_log_mae_model': keras_log_mae_model
    }


def execute_xgb_predictions(x_df, x_model_names, xgb_models, keras_models):
    # Determine length of data
    num_values = x_df.shape[0]

    # Set up empty array for values
    print('Create empty y arrays with', num_values, 'values')
    log_y_predictions = np.empty([num_values, ])
    keras_mae_predictions = np.empty([num_values, ])
    keras_log_mae_predictions = np.empty([num_values, ])

    # Determine unique list of models
    model_names = np.unique(x_model_names)

    print('Number of xgb models in x_data:', len(model_names))

    for model in model_names:
        # Retrieve data indices which match model names
        pred_index = np.where(x_model_names == model)[0]

        # Retrieve data which matches model name
        model_x_df = x_df.iloc[pred_index, :]
        x_data = model_x_df.values
        print('Retrieving keras intermediate model vals for', model, '...')

        x_keras_data = keras_models['mae_intermediate_model'].predict(x_data)
        x_stacked_vals = np.column_stack([x_data, x_keras_data])

        # If model name not found ,use the generic model
        if model not in xgb_models:
            print('WARNING.  Model', model, 'not found.  Using generic model')
            model = 'generic'

        xgb_model_set = load(xgb_models[model])

        print('Executing xgb ' + model + ' predictions ...')
        model_log_y_predictions = xgb_model_set['log_y_model'].predict(x_data)
        model_log_y_predictions = safe_exp(model_log_y_predictions)

        print('Executing keras mae predictions ...')
        model_keras_mae_predictions = xgb_model_set['keras_mae_model'].predict(
            x_stacked_vals)
        model_keras_mae_predictions = safe_exp(model_keras_mae_predictions)

        print('Executing keras log mae predictions ...')
        model_keras_log_mae_predictions = xgb_model_set['keras_log_mae_model'].predict(
            x_stacked_vals)
        model_keras_log_mae_predictions = safe_exp(
            safe_exp(model_keras_log_mae_predictions))

        # Update overall arrays
        print('Updating prediction results ...')
        np.put(log_y_predictions, pred_index, model_log_y_predictions)
        np.put(keras_mae_predictions, pred_index, model_keras_mae_predictions)
        np.put(keras_log_mae_predictions, pred_index,
               model_keras_log_mae_predictions)

    # Return array values
    print('Returning xgb log, keras mae and keras log results')
    return {
        'log_y_predictions': log_y_predictions,
        'keras_mae_predictions': keras_mae_predictions,
        'keras_log_mae_predictions': keras_log_mae_predictions,
    }


# @profile
def train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x,
                        keras_models):
    # Train general model
    models = {}

    tree_method = 'auto'
    predictor = 'gpu_predictor'
    nthread = 8
    # Create model
    models['log_y'] = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method, predictor=predictor,
                                       n_estimators=250, max_depth=70, base_score=0.1,
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
    x_train, x_test, y_train, y_test = train_test_split(
        all_train_x, all_train_y, test_size=0.15)

    eval_set = [(x_test, y_test)]
    models['log_y'].fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae', eval_set=eval_set,
                        verbose=True)

    # Save, delete and reload model to clear memory when using GPU
    print('Saving xgboost log of y model...')
    save(models['log_y'], 'models/xgb-log-y.model.gz')
    print('Deleting xgboost log of y model...')
    del models['log_y']
    print('Reloading xgboost log of y model...')
    models['log_y'] = load('models/xgb-log-y.model.gz')

    gc.collect()

    # output feature importances
    print(models['log_y'].feature_importances_)

    predictions = models['log_y'].predict(all_test_x)
    inverse_scaled_predictions = safe_exp(predictions)

    eval_results({'xgboost_mae': {
        'log_y': all_test_y,
        'actual_y': all_test_actuals,
        'log_y_predict': predictions,
        'y_predict': inverse_scaled_predictions
    }
    })

    print('Retrieving keras intermediate model vals...')
    mae_vals_train = keras_models['mae_intermediate_model'].predict(
        all_train_x)
    mae_vals_test = keras_models['mae_intermediate_model'].predict(all_test_x)

    print('Training xgboost log of y with keras outputs model...')
    models['keras_mae'] = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method, predictor=predictor,
                                           n_estimators=250, max_depth=70, learning_rate=0.05, base_score=0.25,
                                           colsample_bylevel=0.4, colsample_bytree=0.55, gamma=0, min_child_weight=0)

    x_train, x_test, y_train, y_test = train_test_split(
        mae_vals_train, all_train_y, test_size=0.15)

    eval_set = [(x_test, y_test)]
    models['keras_mae'].fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae',
                            eval_set=eval_set, verbose=True)

    # Save, delete and reload model to clear memory when using GPU
    print('Saving xgboost log of y with keras outputs model...')
    save(models['keras_mae'], 'models/xgb-keras-mae.model.gz')
    print('Deleting xgboost log of y with keras outputs model...')
    del models['keras_mae']
    print('Reloading xgboost log of y with keras outputs model...')
    models['keras_mae'] = load('models/xgb-keras-mae.model.gz')

    gc.collect()

    # output feature importances
    print(models['keras_mae'].feature_importances_)

    keras_log_predictions = models['keras_mae'].predict(mae_vals_test)
    # ### Double exp #######
    keras_inverse_scaled_predictions = safe_exp(keras_log_predictions)

    eval_results({'xgboost_keras': {
        'log_y': all_test_y,
        'actual_y': all_test_actuals,
        'log_y_predict': keras_log_predictions,
        'y_predict': keras_inverse_scaled_predictions
    }
    })

    print('Training xgboost log of log of y with keras outputs model...')
    models['keras_log_mae'] = xgb.XGBRegressor(nthread=nthread, tree_method=tree_method, predictor=predictor,
                                               n_estimators=250,
                                               max_depth=130,
                                               base_score=0.4,
                                               colsample_bylevel=0.4,
                                               colsample_bytree=0.4,
                                               gamma=0,
                                               min_child_weight=0,
                                               learning_rate=0.05)

    x_train, x_test, y_train, y_test = train_test_split(
        mae_vals_train, all_train_log_y, test_size=0.15)

    eval_set = [(x_test, y_test)]
    models['keras_log_mae'].fit(x_train, y_train, early_stopping_rounds=25, eval_metric='mae',
                                eval_set=eval_set, verbose=True)

    # Save, delete and reload model to clear memory when using GPU
    print('Saving xgboost log of log of y with keras outputs model...')
    save(models['keras_log_mae'], 'models/xgb-keras-log-mae.model.gz')
    print('Deleting xgboost log of log of y with keras outputs model...')
    del models['keras_log_mae']
    print('Reloading xgboost log of log of y with keras outputs model...')
    models['keras_log_mae'] = load('models/xgb-keras-log-mae.model.gz')

    gc.collect()

    # output feature importances
    print(models['keras_log_mae'].feature_importances_)

    keras_log_log_predictions = models['keras_log_mae'].predict(mae_vals_test)
    # ### Double exp #######
    keras_log_inverse_scaled_predictions = safe_exp(
        safe_exp(keras_log_log_predictions))

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

    return models


# @profile
def train_keras_nn(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals, df_all_test_y,
                   df_all_test_x, use_previous_training_weights):
    # Load values into numpy arrays - drop the model name for use with xgb
    train_y = df_all_train_y.values
    train_actuals = df_all_train_actuals.values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.values
    test_actuals = df_all_test_actuals.values
    test_y = df_all_test_y.values
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

    # See if we should load previous weights
    if use_previous_training_weights and Path('./weights/weights-1.hdf5').exists():
        p_model.load_weights('./weights/weights-1.hdf5')

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, verbose=1, patience=3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=12)
    csv_logger = CSVLogger('./logs/actual-mape-training.log')
    checkpointer = ModelCheckpoint(
        filepath='./weights/weights-1.hdf5', verbose=0, save_best_only=True)

    # Reorder array - get array index
    s = np.arange(train_x.shape[0])

    # Vals *.85 (train / test split) / batch size * num epochs for cycle
    step_size = math.ceil(s.shape[0] * .85 / 256) * 4
    clr = CyclicLR(base_lr=0.001, max_lr=0.04, step_size=step_size)
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
                          callbacks=[  # reduce_lr,#
                              early_stopping, csv_logger, checkpointer, clr],
                          verbose=0)

    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.title("CLR Min Max Learning - MAPE")
    # plt.plot(clr.history['lr'], clr.history['loss'], )

    p_model.load_weights('./weights/weights-1.hdf5')

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

    # See if we should load previous weights
    if use_previous_training_weights and Path('./weights/weights-2.hdf5').exists():
        model.load_weights('./weights/weights-2.hdf5')

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=12)
    csv_logger = CSVLogger('./logs/log-training.log')
    checkpointer = ModelCheckpoint(
        filepath='./weights/weights-2.hdf5', verbose=0, save_best_only=True)

    print('Training keras mae model...')

    # Reorder array - get array index
    s = np.arange(train_x.shape[0])

    # Temporary get less data
    # s = s[0:20000]

    step_size = math.ceil(s.shape[0] * .85 / 512) * 100
    clr = CyclicLR(base_lr=0.001, max_lr=0.04,
                   step_size=step_size, mode='exp_range', gamma=0.96)

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
                        callbacks=[  # reduce_lr,
                            early_stopping, checkpointer, csv_logger, clr],
                        verbose=0)

    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.title("CLR Min Max Learning - MAE")
    # plt.plot(clr.history['lr'], clr.history['loss'])

    model.load_weights('./weights/weights-2.hdf5')

    print('Executing keras predictions...')

    log_y_predictions = model.predict(test_x)
    exp_predictions = safe_exp(log_y_predictions)

    eval_results({'keras_log_y': {
        # 'log_y': test_y,
        'actual_y': test_actuals,
        # 'log_y_predict': log_predictions,
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


def train_deep_bagging(train_predictions, train_actuals, test_predictions,
                       test_actuals, use_previous_training_weights):
    print('Training keras based bagging...')
    train_x = train_predictions.values
    train_y = train_actuals.values
    test_x = test_predictions.values
    test_y = test_actuals.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, verbose=1, patience=2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(
        filepath='./weights/weights-3.hdf5', verbose=0, save_best_only=True)
    # Vals *.8 (train / test split) / batch size * num epochs for cycle
    step_size = math.ceil(train_x_scaled.shape[0] * 0.8 / 1024) * 4
    clr = CyclicLR(base_lr=0.001, max_lr=0.03, step_size=step_size)

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

    # See if we should load previous weights
    if use_previous_training_weights and Path('./weights/weights-3.hdf5').exists():
        model.load_weights('./weights/weights-3.hdf5')

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])

    history = model.fit(train_x_scaled, train_y,
                        batch_size=network['batch_size'],
                        epochs=2000,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=[csv_logger, clr,  # reduce_lr,
                                   early_stopping, checkpointer])

    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.title("CLR Min Max Learning - deep bagging")
    # plt.plot(clr.history['lr'], clr.history['loss'], )

    print('\rResults')

    model.load_weights('./weights/weights-3.hdf5')
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


def execute_deep_bagging(model, scaler, bagging_df):
    print('Executing keras based bagging...')
    test_x = bagging_df.values

    test_x_scaled = scaler.transform(test_x)

    predictions = model.predict(test_x_scaled)
    prediction_results = predictions.reshape(predictions.shape[0], )

    return prediction_results


# @profile
def assess_results(df_predictions, model_names, df_actuals, run_str):
    test_actuals = df_actuals.values

    range_predictions = {}

    for column in df_predictions.columns.values:
        range_predictions[column] = df_predictions[column].values

    range_results(range_predictions, test_actuals)

    symbol_results(
        model_names, df_predictions['deep_bagged_predictions'].values, test_actuals, run_str)


def symbol_results(symbols_x, predictions, actuals, run_str):
    # Determine unique list of symbols
    symbols = np.unique(symbols_x)

    print('Executing symbol results, number of symbols in prediction data:', len(symbols))

    df_results = pd.DataFrame()

    for symbol in symbols:
        # Retrieve data indices which match symbols
        pred_index = np.where(symbols_x == symbol)[0]

        # Retrieve data which matches symbol
        symbol_predictions = predictions[pred_index]
        symbol_actuals = actuals[pred_index]

        # execute val for symbol
        symbol_results = eval_results({symbol: {
            'actual_y': symbol_actuals,
            'y_predict': symbol_predictions
        }
        })

        mean_actual_val = np.mean(symbol_actuals)
        median_actual_val = np.median(symbol_actuals)

        mean_predicted_val = np.mean(symbol_predictions)
        median_predicted_val = np.median(symbol_predictions)

        symbol_dict = {
            'symbol': [symbol],
            'mean_actual_val': [mean_actual_val],
            'median_actual_val': [median_actual_val],
            'mean_predicted_val': [mean_predicted_val],
            'median_predicted_val': [median_predicted_val],
        }
        # Add results values
        for key in symbol_results[symbol]:
            symbol_dict[key] = [symbol_results[symbol][key]]

        # create data frame from results
        df_symbol_result = pd.DataFrame.from_dict(symbol_dict)

        # Add data frame into all results
        df_results = pd.concat([df_results, df_symbol_result])

    # When all symbols are done, write the results as a csv
    df_results.to_csv('./results/' + run_str + '.csv')


def execute_train_test_predictions(df_all_train_x, train_x_model_names, df_all_train_actuals,
                                   df_all_test_x, test_x_model_names, df_all_test_actuals,
                                   xgb_models, keras_models):
    print('Executing and exporting predictions data...')
    # export results
    df_all_test_actuals.to_pickle(
        'data/test_actuals.pkl.gz', compression='gzip')
    df_all_train_actuals.to_pickle(
        'data/train_actuals.pkl.gz', compression='gzip')

    train_y_predictions = execute_model_predictions(
        df_all_train_x, train_x_model_names, xgb_models, keras_models)
    test_y_predictions = execute_model_predictions(
        df_all_test_x, test_x_model_names, xgb_models, keras_models)

    train_predictions = pd.DataFrame.from_dict({
        'xgboost_log': train_y_predictions['xgboost_log'],
        'keras_mape': train_y_predictions['keras_mape'],
        'keras_log': train_y_predictions['keras_log'],
        'xgboost_keras_log': train_y_predictions['xgboost_keras_log'],
        'xgboost_keras_log_log': train_y_predictions['xgboost_keras_log_log'],
    })
    train_predictions.to_pickle(
        'data/train_predictions.pkl.gz', compression='gzip')

    test_predictions = pd.DataFrame.from_dict({
        'xgboost_log': test_y_predictions['xgboost_log'],
        'keras_mape': test_y_predictions['keras_mape'],
        'keras_log': test_y_predictions['keras_log'],
        'xgboost_keras_log': test_y_predictions['xgboost_keras_log'],
        'xgboost_keras_log_log': test_y_predictions['xgboost_keras_log_log'],
    })
    test_predictions.to_pickle(
        'data/test_predictions.pkl.gz', compression='gzip')

    return train_predictions, test_predictions


def execute_model_predictions(df_x, x_model_names, xgb_models, keras_models):
    print('Executing xgb predictions')
    xgb_predictions = execute_xgb_predictions(
        df_x, x_model_names, xgb_models, keras_models)

    gen_predictions = xgb_predictions['log_y_predictions']
    xgboost_keras_gen_predictions = xgb_predictions['keras_mae_predictions']
    xgboost_keras_log_predictions = xgb_predictions['keras_log_mae_predictions']

    print('Executing keras predictions')
    data_x = df_x.values

    keras_mape_predictions = keras_models['mape_model'].predict(data_x)

    keras_log_predictions = keras_models['mae_model'].predict(data_x)
    keras_log_predictions = safe_exp(keras_log_predictions)

    # Make consistent shape for outputs from keras
    keras_mape_predictions = keras_mape_predictions.reshape(
        keras_mape_predictions.shape[0], )
    keras_log_predictions = keras_log_predictions.reshape(
        keras_log_predictions.shape[0], )

    predictions_df = pd.DataFrame.from_dict({
        'xgboost_log': flatten_array(gen_predictions),
        'keras_mape': flatten_array(keras_mape_predictions),
        'keras_log': flatten_array(keras_log_predictions),
        'xgboost_keras_log': flatten_array(xgboost_keras_gen_predictions),
        'xgboost_keras_log_log': flatten_array(xgboost_keras_log_predictions),
    })

    return predictions_df


def main(run_config):
    # Prepare run_str
    run_str = datetime.now().strftime('%Y%m%d%H%M')

    initialise_print_logger('logs/training-' + run_str + '.log')

    print('Starting sharecast run:', run_str)

    # Check whether we can skip all preprocessing steps
    needs_preprocessing = False

    if run_config.get('use_previous_training_weights') is True:
        use_previous_training_weights = True
    else:
        use_previous_training_weights = False

    if run_config.get('load_data') is True:
        needs_preprocessing = True

    if run_config.get('train_pre_process') is True:
        needs_preprocessing = True

    # Retrieve and divide data
    if needs_preprocessing:
        if run_config.get('load_data') is True:
            # Load and divide data
            if run_config.get('generate_labels') is True:
                share_data = load_data(run_config['unlabelled_data_file'],
                                       drop_unlabelled=True,
                                       drop_labelled=False,
                                       generate_labels=True,
                                       label_weeks=run_config['generate_label_weeks'],
                                       reference_date=run_config['reference_date'],
                                       labelled_file_name=run_config['labelled_data_file']
                                       )
            else:
                share_data = load_data(run_config['labelled_data_file'])
            gc.collect()

            # Divide data into symbol sand general data for training an testing
            symbol_map, df_all_train_y, df_all_train_actuals, df_all_train_x, df_all_test_actuals, \
                df_all_test_y, df_all_test_x = divide_data(share_data)

            del share_data
            gc.collect()

            # Save data after dividing
            df_all_train_x.to_pickle(
                'data/pp_train_x_df.pkl.gz', compression='gzip')
            df_all_train_y.to_pickle(
                'data/df_all_train_y.pkl.gz', compression='gzip')
            df_all_train_actuals.to_pickle(
                'data/df_all_train_actuals.pkl.gz', compression='gzip')
            df_all_test_x.to_pickle(
                'data/pp_test_x_df.pkl.gz', compression='gzip')
            df_all_test_y.to_pickle(
                'data/df_all_test_y.pkl.gz', compression='gzip')
            df_all_test_actuals.to_pickle(
                'data/df_all_test_actuals.pkl.gz', compression='gzip')

        else:
            # Data already divided
            print('Loading divided data')
            df_all_train_x = pd.read_pickle(
                'data/pp_train_x_df.pkl.gz', compression='gzip')
            df_all_train_y = pd.read_pickle(
                'data/df_all_train_y.pkl.gz', compression='gzip')
            df_all_train_actuals = pd.read_pickle(
                'data/df_all_train_actuals.pkl.gz', compression='gzip')
            df_all_test_x = pd.read_pickle(
                'data/pp_test_x_df.pkl.gz', compression='gzip')
            df_all_test_y = pd.read_pickle(
                'data/df_all_test_y.pkl.gz', compression='gzip')
            df_all_test_actuals = pd.read_pickle(
                'data/df_all_test_actuals.pkl.gz', compression='gzip')

        # Retain model names for train and test
        print('Retaining model name data')
        train_x_model_names = df_all_train_x['model'].values
        test_x_model_names = df_all_test_x['model'].values

        save(train_x_model_names, 'data/train_x_model_names.pkl.gz')
        save(test_x_model_names, 'data/test_x_model_names.pkl.gz')

        # Drop model names
        df_all_train_x = df_all_train_x.drop(['model'], axis=1)
        df_all_test_x = df_all_test_x.drop(['model'], axis=1)

        df_all_train_x.info()
        df_all_test_x.info()

        if run_config.get('train_pre_process') is True:
            # Execute pre-processing trainer
            df_all_train_x, se, imputer, scaler = train_preprocessor(
                df_all_train_x, df_all_train_y)
            df_all_test_x = execute_preprocessor(
                df_all_test_x, se, imputer, scaler)

            # Write processed data to files
            df_all_train_x.to_pickle(
                'data/df_all_train_x.pkl.gz', compression='gzip')
            df_all_test_x.to_pickle(
                'data/df_all_test_x.pkl.gz', compression='gzip')

        if run_config.get('load_and_execute_pre_process') is True:
            print('Loading pre-processing models')
            # Load pre-processing models
            se = load('models/se.pkl.gz')
            imputer = load('models/imputer.pkl.gz')
            scaler = load('models/scaler.pkl.gz')

            print('Executing pre-processing')
            # Execute pre-processing
            df_all_train_x = execute_preprocessor(
                df_all_train_x, se, imputer, scaler)
            df_all_test_x = execute_preprocessor(
                df_all_test_x, se, imputer, scaler)

            # Write processed data to files
            df_all_train_x.to_pickle(
                'data/df_all_train_x.pkl.gz', compression='gzip')
            df_all_test_x.to_pickle(
                'data/df_all_test_x.pkl.gz', compression='gzip')

    else:
        print('Load model name data')
        train_x_model_names = load('data/train_x_model_names.pkl.gz')
        test_x_model_names = load('data/test_x_model_names.pkl.gz')

    if run_config.get('load_processed_data') is True:
        print('Loading pre-processed data')
        df_all_train_x = pd.read_pickle(
            'data/df_all_train_x.pkl.gz', compression='gzip')
        df_all_train_y = pd.read_pickle(
            'data/df_all_train_y.pkl.gz', compression='gzip')
        df_all_train_actuals = pd.read_pickle(
            'data/df_all_train_actuals.pkl.gz', compression='gzip')
        df_all_test_x = pd.read_pickle(
            'data/df_all_test_x.pkl.gz', compression='gzip')
        df_all_test_y = pd.read_pickle(
            'data/df_all_test_y.pkl.gz', compression='gzip')
        df_all_test_actuals = pd.read_pickle(
            'data/df_all_test_actuals.pkl.gz', compression='gzip')

    if run_config.get('train_keras') is True:
        # Train keras models
        keras_models = train_keras_nn(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals,
                                      df_all_test_y, df_all_test_x, use_previous_training_weights)
    else:
        print('Loading keras models')
        # Load keras models
        keras_models = {
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

    if run_config.get('train_xgb') is True:
        # Train the general models
        # gen_models = train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y,
        #                                  df_all_test_x, keras_models)
        train_xgb_models(df_all_train_x, df_all_train_y, train_x_model_names, test_x_model_names,
                         df_all_test_actuals, df_all_test_y, df_all_test_x, keras_models)

    print('Loading xgboost model list')
    xgb_models = load_xgb_models()

    # Export data prior to bagging
    train_predictions, test_predictions = execute_train_test_predictions(df_all_train_x, train_x_model_names,
                                                                         df_all_train_actuals, df_all_test_x,
                                                                         test_x_model_names, df_all_test_actuals,
                                                                         xgb_models, keras_models)

    if run_config.get('train_deep_bagging') is True:
        bagging_model, bagging_scaler, deep_bagged_predictions = train_deep_bagging(train_predictions,
                                                                                    df_all_train_actuals,
                                                                                    test_predictions,
                                                                                    df_all_test_actuals,
                                                                                    use_previous_training_weights)
    else:
        print('Loading bagging models')
        bagging_model = load_model('models/keras-bagging-model.h5')
        bagging_scaler = load('models/deep-bagging-scaler.pkl.gz')
        deep_bagged_predictions = execute_deep_bagging(
            bagging_model, bagging_scaler, test_predictions)

    # Add deep bagged predictions to set
    test_predictions['deep_bagged_predictions'] = deep_bagged_predictions

    assess_results(test_predictions, test_x_model_names,
                   df_all_test_actuals, run_str)

    print('Execution completed')


if __name__ == "__main__":
    run_config = {
        'load_data': False,
        'generate_labels': False,
        'generate_label_weeks': 8,
        'reference_date': '2018-05-12',
        'unlabelled_data_file': './data/ml-20180512-processed.pkl.gz',
        'labelled_data_file': './data/ml-20180512-labelled.pkl.gz',
        'train_pre_process': False,
        'load_and_execute_pre_process': False,
        'load_processed_data': True,
        'train_keras': False,
        'use_previous_training_weights': True,
        'train_xgb': False,
        'train_deep_bagging': True,
    }

    main(run_config)
