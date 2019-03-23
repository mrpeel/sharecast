import glob
import gc
import os
from pathlib import Path
import math

import joblib
import pandas as pd
import numpy as np
import numba
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.model_selection import train_test_split

from keras import Model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from eval_results import eval_results, range_results
from print_logger import print
from stats_operations import safe_log, safe_exp
from stats_operations import flatten_array

# from clr_callback import CyclicLR
from compile_keras import compile_keras_model
# import matplotlib.pyplot as plt

from processing_constants import LABEL_COLUMN
from processing_constants import CONTINUOUS_COLUMNS, PAST_RESULTS_CONTINUOUS_COLUMNS
from processing_constants import CATEGORICAL_COLUMNS, PAST_RESULTS_CATEGORICAL_COLUMNS
from processing_constants import COLUMNS_TO_REMOVE, RECURRENT_COLUMNS
from processing_constants import XGB_SET_PATH, INDUSTRY_XGB_SET_PATH


def save(save_object, filename):
    """Saves a compressed object to disk
       """
    joblib.dump(save_object, filename)


# @profile
def load(filename):
    """Loads a compressed object from disk
    """
    model_object = joblib.load(filename)
    return model_object


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
def load_data(file_name, **kwargs):
    """Load pickled data and run combined prep
        Arguments:
        file_name
        drop_unlabelled=True
        drop_labelled=False
        generate_labels=False
        generate_tminus_labels=False
        label_weeks=None
        reference_date=None
        labelled_file_name=None
        unlabelled_file_name=None
    """
    print('Loading file:', file_name)

    drop_unlabelled = kwargs.get('drop_unlabelled', True)

    df = pd.read_pickle(file_name, compression='gzip')
    gc.collect()

    print('Number of "NA" symbols:',
        df[df['symbol'] == 'NA'].shape[0])

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
        if len(df_train) > 50:
            # Check whether this will have its own model or the generic model
            if len(df_train) > 150:
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

    return symbol_map, df_all_train_y, df_all_train_actuals, df_all_train_x, df_all_test_actuals, df_all_test_y, df_all_test_x


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


def train_rar_encoder(df, column_name):
    # Calculate risk adjusted return
    temp_df = pd.DataFrame()

    # Copy values to new dataframe
    print('Copying return data')
    temp_df[
        [column_name, 'adjustedPrice', 'eight_week_price_return', 'eight_week_dividend_return', 'eight_week_total_return',
         'eight_week_std']] = df[
        [column_name, 'adjustedPrice', 'eight_week_price_return', 'eight_week_dividend_return', 'eight_week_total_return',
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

    rar_lookup = pd.DataFrame(temp_df.groupby([column_name])[
        'ra_total_return'].mean().reset_index(name=column_name + '_encoded'))

    ret_df = execute_rar_encoder(df, rar_lookup, column_name)

    return ret_df, rar_lookup


def execute_rar_encoder(df, rar_df, column_name):
    # Merge encodeded values
    print('Merging encoded', column_name, 'with dataframe')
    ret_df = df.merge(rar_df, left_on=column_name,
                      right_on=column_name, how='left')

    # Remove symbol column
    print('Dropping', column_name, 'column')
    ret_df.drop([column_name], axis=1, inplace=True)

    # Impute any missing values for encoded column (can happen when predictions include new values)
    encoded_column = column_name + '_encoded'
    ret_df[encoded_column].fillna(
        (ret_df[encoded_column].median()), inplace=True)

    return ret_df


def train_imputer(df):
    print('Training imputer')
    imputer = [Imputer(strategy='median'), Imputer(
        strategy='median'), Imputer(strategy='median')]
    print('-- continous columns')
    imputer[0].fit(df[CONTINUOUS_COLUMNS].values)
    print('-- past results continous columns')
    imputer[1].fit(df[PAST_RESULTS_CONTINUOUS_COLUMNS].values)
    print('-- recurrent columns')
    imputer[2].fit(df[RECURRENT_COLUMNS].values)

    ret_df = execute_imputer(df, imputer)

    return ret_df, imputer


def execute_imputer(df, imputer):
    print('Executing imputer')
    ret_df = df
    print('-- continuous columns')
    ret_df[CONTINUOUS_COLUMNS] = imputer[0].transform(
        ret_df[CONTINUOUS_COLUMNS].values)
    ret_df[CONTINUOUS_COLUMNS] = ret_df[CONTINUOUS_COLUMNS].astype(
        'float32', errors='ignore')
    print('-- past results continous columns')
    ret_df[PAST_RESULTS_CONTINUOUS_COLUMNS] = imputer[1].transform(
        ret_df[PAST_RESULTS_CONTINUOUS_COLUMNS].values)
    ret_df[PAST_RESULTS_CONTINUOUS_COLUMNS] = ret_df[PAST_RESULTS_CONTINUOUS_COLUMNS].astype(
        'float32', errors='ignore')
    print('-- recurrent columns')
    ret_df[RECURRENT_COLUMNS] = imputer[2].transform(
        ret_df[RECURRENT_COLUMNS].values)
    ret_df[RECURRENT_COLUMNS] = ret_df[RECURRENT_COLUMNS].astype(
        'float32', errors='ignore')

    return ret_df


def train_scaler(df):
    print('Training scaler')
    scaler = [MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1)),
              MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))]
    print('-- continuous columns')
    scaler[0].fit(df[CONTINUOUS_COLUMNS].values)
    print('-- past results continuous columns')
    scaler[1].fit(df[PAST_RESULTS_CONTINUOUS_COLUMNS].values)
    print('-- recurrent columns')
    scaler[2].fit(df[RECURRENT_COLUMNS].values)
    print('-- categorical columns')
    scaler[3].fit(df[CATEGORICAL_COLUMNS].values)

    ret_df = execute_scaler(df, scaler)

    return ret_df, scaler


def execute_scaler(df, scaler):
    print('Executing scaler')
    print('-- continuous columns')
    df[CONTINUOUS_COLUMNS] = scaler[0].transform(df[CONTINUOUS_COLUMNS].values)
    df[CONTINUOUS_COLUMNS] = df[CONTINUOUS_COLUMNS].astype(
        'float32', errors='ignore')

    print('-- past results continuous columns')
    df[PAST_RESULTS_CONTINUOUS_COLUMNS] = scaler[1].transform(
        df[PAST_RESULTS_CONTINUOUS_COLUMNS].values)
    df[PAST_RESULTS_CONTINUOUS_COLUMNS] = df[PAST_RESULTS_CONTINUOUS_COLUMNS].astype(
        'float32', errors='ignore')

    print('-- recurrent columns')
    df[RECURRENT_COLUMNS] = scaler[2].transform(df[RECURRENT_COLUMNS].values)
    df[RECURRENT_COLUMNS] = df[RECURRENT_COLUMNS].astype(
        'float32', errors='ignore')

    print('-- categorical columns')
    df[CATEGORICAL_COLUMNS] = scaler[3].transform(
        df[CATEGORICAL_COLUMNS].values)
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].astype(
        'float32', errors='ignore')

    return df


def train_preprocessor(train_x_df):
    print('Training pre-processor...')

    print('One hot encoding past results categorical columns')
    train_x_df = execute_one_hot_string_encoder(
        train_x_df, PAST_RESULTS_CATEGORICAL_COLUMNS)
    gc.collect()

    print('Encoding symbol values')
    train_x_df, symbol_encoder = train_rar_encoder(train_x_df, 'symbol')
    gc.collect()

    print('Imputing missing values')
    train_x_df, imputer = train_imputer(train_x_df)
    gc.collect()

    print('Scaling data')
    train_x_df, scaler = train_scaler(train_x_df)
    gc.collect()

    # Write one hot encoder, symbol encoder, scaler and categorical encoder to files
    save(symbol_encoder, 'models/se.pkl.gz')
    save(imputer, 'models/imputer.pkl.gz')
    save(scaler, 'models/scaler.pkl.gz')
    # save(ce, 'models/ce.pkl.gz')

    return train_x_df, symbol_encoder, imputer, scaler


def execute_preprocessor(transform_df, symbol_encoder, imputer, scaler):
    print('Executing pre-processor on supplied data...')

    print('One hot encoding past results categorical columns')
    transform_df = execute_one_hot_string_encoder(
        transform_df, PAST_RESULTS_CATEGORICAL_COLUMNS)
    gc.collect()

    print('Encoding symbol values')
    transform_df = execute_rar_encoder(transform_df, symbol_encoder, 'symbol')
    gc.collect()

    print('Imputing missing values')
    transform_df = execute_imputer(transform_df, imputer)
    gc.collect()

    print('Remove any remaining columns with nan values')
    transform_df.dropna(inplace=True)

    print('Scaling data...')
    transform_df = execute_scaler(transform_df, scaler)
    gc.collect()

    return transform_df


# @profile
def load_xgb_models(model_type='symbol'):
    all_xgb_models = {}

    if model_type == 'symbol':
        model_path = XGB_SET_PATH
    else:
        model_path = INDUSTRY_XGB_SET_PATH

    print('Loading files from', model_path)
    # Return files in path
    file_list = glob.glob(model_path + '*.model.gz')

    # load each model set
    for file_name in file_list:
        # remove leading path and trailing file extension
        model_name = file_name.replace(
            model_path, '').replace('.model.gz', '')

        # create model property and load model set into it
        all_xgb_models[model_name] = file_name

    return all_xgb_models


# @profile
def train_xgb_models(df_all_train_x, df_all_train_y, train_x_model_names, test_x_model_names,
                     df_all_test_actuals, df_all_test_y, df_all_test_x, keras_models,
                     model_type='symbol'):

    if model_type == 'symbol':
        model_path = XGB_SET_PATH
    else:
        model_path = INDUSTRY_XGB_SET_PATH

    # clear previous models
    files = glob.glob(model_path + '*.model.gz')
    for file in files:
        os.remove(file)

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
        save(xgb_model_set, model_path + model + '.model.gz')


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
    # all_test_log_y = safe_log(all_test_y)

    print('Training xgboost log of y model for', model_set_name)
    print('Number of training instances:', len(all_train_x))
    print('Number of test instances:', len(all_test_x))
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
    # all_test_log_y = safe_log(all_test_y)

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
    # train_log_y = safe_log(train_y)
    train_x = df_all_train_x.values
    test_actuals = df_all_test_actuals.values
    # test_y = df_all_test_y.values
    # test_log_y = safe_log(test_y)
    test_x = df_all_test_x.values

    print('Training keras mape model...')

    network = {
        'hidden_layers': [5, 5, 5],
        'activation': 'relu',
        'optimizer': 'AdamW',
        'kernel_initializer': 'glorot_uniform',
        'batch_size': 256,
        'dropout': 0.05,
        'model_type': 'mape',
    }

    num_samples = train_x.shape[0]
    dimensions = train_x.shape[1]
    num_epochs = 500
    
    network['weight_decay'] = 0.005 * (network['batch_size'] / num_samples / num_epochs )**0.5

    p_model = compile_keras_model(network, dimensions)

    # See if we should load previous weights
    if use_previous_training_weights and Path('./weights/weights-1.hdf5').exists():
        p_model.load_weights('./weights/weights-1.hdf5')

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, verbose=1, patience=15, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    csv_logger = CSVLogger('./logs/actual-mape-training.log')
    checkpointer = ModelCheckpoint(
        filepath='./weights/weights-1.hdf5', verbose=0, save_best_only=True)

    # Reorder array - get array index
    array_index = np.arange(train_x.shape[0])

    # Vals *.85 (train / test split) / batch size * num epochs for cycle
    # step_size = math.ceil(array_index.shape[0] * .85 / 256) * 4
    # clr = CyclicLR(base_lr=0.001, max_lr=0.04, step_size=step_size)
    # Reshuffle index
    np.random.shuffle(array_index)

    # Create array using new index
    x_shuffled_train = train_x[array_index]
    y_shuffled_train = train_actuals[array_index]

    # history = p_model.fit(x_shuffled_train,
    p_model.fit(x_shuffled_train,
                y_shuffled_train,
                validation_split=0.15,
                epochs=num_epochs,
                batch_size=network['batch_size'],
                callbacks=[reduce_lr, early_stopping, csv_logger, checkpointer],
                verbose=0)

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
        'optimizer': 'AdamW',
        'kernel_initializer': 'normal',
        'dropout': 0.1,
        'batch_size': 512,
        'model_type': 'mae',
        'int_layer': 30,
    }

    num_samples = train_x.shape[0]
    dimensions = train_x.shape[1]
    num_epochs = 500
    
    network['weight_decay'] = 0.005 * (network['batch_size'] / num_samples / num_epochs )**0.5

    model = compile_keras_model(network, dimensions)

    # See if we should load previous weights
    if use_previous_training_weights and Path('./weights/weights-2.hdf5').exists():
        model.load_weights('./weights/weights-2.hdf5')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1, patience=15, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    csv_logger = CSVLogger('./logs/log-training.log')
    checkpointer = ModelCheckpoint(
        filepath='./weights/weights-2.hdf5', verbose=0, save_best_only=True)

    print('Training keras mae model...')

    # Reorder array - get array index
    array_index = np.arange(train_x.shape[0])

    # step_size = math.ceil(array_index.shape[0] * .85 / 512) * 100
    # clr = CyclicLR(base_lr=0.001, max_lr=0.04,
    #                step_size=step_size, mode='exp_range', gamma=0.96)

    # Reshuffle index
    np.random.shuffle(array_index)

    # Create array using new index
    x_shuffled_train = train_x[array_index]
    y_shuffled_train = train_y[array_index]

    # history = model.fit(x_shuffled_train,
    model.fit(x_shuffled_train,
              y_shuffled_train,
              validation_split=0.15,
              epochs=num_epochs,
              batch_size=network['batch_size'],
              callbacks=[reduce_lr, early_stopping, checkpointer, csv_logger],
              verbose=0)

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

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1, patience=15, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(
        filepath='./weights/weights-3.hdf5', verbose=0, save_best_only=True)
    # Vals *.8 (train / test split) / batch size * num epochs for cycle
    # step_size = math.ceil(train_x_scaled.shape[0] * 0.8 / 1024) * 4
    # clr = CyclicLR(base_lr=0.001, max_lr=0.03, step_size=step_size)

    dimensions = train_x.shape[1]

    network = {
        'activation': 'PReLU',
        'optimizer': 'AdamW',
        'batch_size': 1024,
        'dropout': 0.05,
        'model_type': 'mae_mape',
        'kernel_initializer': 'normal',
        'hidden_layers': [5],
    }

    num_samples = train_x.shape[0]
    dimensions = train_x.shape[1]
    num_epochs = 500
    
    network['weight_decay'] = 0.005 * (network['batch_size'] / num_samples / num_epochs )**0.5

    model = compile_keras_model(network, dimensions)

    # See if we should load previous weights
    if use_previous_training_weights and Path('./weights/weights-3.hdf5').exists():
        model.load_weights('./weights/weights-3.hdf5')

    print('\rNetwork')

    for network_property in network:
        print(network_property, ':', network[network_property])

    # history = model.fit(train_x_scaled, train_y,
    model.fit(train_x_scaled, train_y,
              batch_size=network['batch_size'],
              epochs=num_epochs,
              verbose=0,
              validation_split=0.2,
              callbacks=[csv_logger, reduce_lr, early_stopping, checkpointer])

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
        this_symbol_results = eval_results({symbol: {
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
        for key in this_symbol_results[symbol]:
            symbol_dict[key] = [this_symbol_results[symbol][key]]

        # create data frame from results
        df_symbol_result = pd.DataFrame.from_dict(symbol_dict)

        # Add data frame into all results
        df_results = pd.concat([df_results, df_symbol_result])

    # When all symbols are done, write the results as a csv
    df_results.to_csv('./results/' + run_str + '.csv')


def execute_train_test_predictions(df_all_train_x, train_x_model_names, train_x_gics_industry_groups,
                                   df_all_train_actuals, df_all_test_x, test_x_model_names,
                                   test_x_gics_industry_groups, df_all_test_actuals, xgb_models,
                                   xgb_industry_models, keras_models):
    print('Executing and exporting predictions data...')
    # export results
    df_all_test_actuals.to_pickle(
        'data/test_actuals.pkl.gz', compression='gzip')
    df_all_train_actuals.to_pickle(
        'data/train_actuals.pkl.gz', compression='gzip')

    train_y_predictions = execute_model_predictions(df_all_train_x, train_x_model_names,
                                                    train_x_gics_industry_groups, xgb_models,
                                                    xgb_industry_models, keras_models)
    test_y_predictions = execute_model_predictions(df_all_test_x, test_x_model_names,
                                                   test_x_gics_industry_groups, xgb_models,
                                                   xgb_industry_models, keras_models)

    train_predictions = pd.DataFrame.from_dict({
        'xgboost_log': train_y_predictions['xgboost_log'],
        'xgboost_industry_log': train_y_predictions['xgboost_industry_log'],
        'keras_mape': train_y_predictions['keras_mape'],
        'keras_log': train_y_predictions['keras_log'],
        'xgboost_keras_log': train_y_predictions['xgboost_keras_log'],
        'xgboost_keras_log_log': train_y_predictions['xgboost_keras_log_log'],
    })
    train_predictions.to_pickle(
        'data/train_predictions.pkl.gz', compression='gzip')

    test_predictions = pd.DataFrame.from_dict({
        'xgboost_log': test_y_predictions['xgboost_log'],
        'xgboost_industry_log': test_y_predictions['xgboost_industry_log'],
        'keras_mape': test_y_predictions['keras_mape'],
        'keras_log': test_y_predictions['keras_log'],
        'xgboost_keras_log': test_y_predictions['xgboost_keras_log'],
        'xgboost_keras_log_log': test_y_predictions['xgboost_keras_log_log'],
    })
    test_predictions.to_pickle(
        'data/test_predictions.pkl.gz', compression='gzip')

    return train_predictions, test_predictions


def execute_model_predictions(df_x, x_model_names, x_gics_industry_groups, xgb_models, xgb_industry_models,
                              keras_models):
    print('Executing xgb symbol predictions.  Number of rows:', len(df_x))
    xgb_predictions = execute_xgb_predictions(
        df_x, x_model_names, xgb_models, keras_models)
    gen_predictions = xgb_predictions['log_y_predictions']
    xgboost_keras_gen_predictions = xgb_predictions['keras_mae_predictions']
    xgboost_keras_log_predictions = xgb_predictions['keras_log_mae_predictions']

    print('Executing xgb industry predictions.  Number of rows:', len(df_x))
    xgb_industry_predictions = execute_xgb_predictions(df_x, x_gics_industry_groups, xgb_industry_models,
                                                       keras_models)
    gen_industry_predictions = xgb_industry_predictions['log_y_predictions']

    print('Executing keras predictions.  Number of rows:', len(df_x))
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
        'xgboost_industry_log': flatten_array(gen_industry_predictions),
        'keras_mape': flatten_array(keras_mape_predictions),
        'keras_log': flatten_array(keras_log_predictions),
        'xgboost_keras_log': flatten_array(xgboost_keras_gen_predictions),
        'xgboost_keras_log_log': flatten_array(xgboost_keras_log_predictions),
    })

    return predictions_df


def convert_file_string(original_name):
    output_str = original_name.replace('&', 'And')
    output_str = output_str.replace(',', '')
    output_str = output_str.replace('-', '')
    output_str = output_str.replace('(', '')
    output_str = output_str.replace(')', '')
    output_str = output_str.replace(' ', '')
    return output_str


def fix_categorical(categorical_data):
    unique_vals = categorical_data.unique()
    category_renamer = {}

    for index, element in enumerate(unique_vals):
        category_renamer[element] = convert_file_string(element)

    print('Converting categories')
    print(category_renamer)

    return categorical_data.rename_categories(category_renamer)
