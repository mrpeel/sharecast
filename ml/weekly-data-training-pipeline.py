
import sys
import pandas as pd
import numpy as np
from numpy import array
import math
from timeit import default_timer as timer
from datetime import datetime, timedelta
import numba
from ensemble_processing import load_data, load, save
import ta
from scipy import stats
from processing_constants import ALL_CONTINUOUS_COLUMNS, HIGH_NAN_COLUMNS, WHOLE_MARKET_COLUMNS
from processing_constants import BOLLINGER_PREDICTION_COLUMNS, BOLLINGER_VALUE_COLUMNS
from optimise_dataframe import optimise_df
import glob
from sklearn import preprocessing
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt


# sys.path.append('../')


# # Training pipeline steps
# 
# 0. Run execute_data_preparation.py for daily data set
# 1. Load daily data values
# 2. Create weekly values for data
# 3. Execute technical analysis library
# 4. Encode class vals
# 5. Scale numeric vals
# 6. Convert target val into bins
#   bins = -99.07692719, -13.13461361, -3.00000238, 0.,9.25964718, 1493.
#   bin_names = 'strong_sell', 'sell', 'hold', 'buy', 'strong_buy'
# 7. Convert bin vals to class indexes
# 8. Divide data into train / test split - balanced per symbol
# 9. Train ensemble VotingClassifier with soft voting:
#     - RandomForestClassifier(max_depth=20, n_estimators=505, max_features=1242, min_samples_split=2, min_samples_leaf=1)
#     - AdaBoostClassifier(RandomForestClassifier(max_depth=20, n_estimators=505, max_features=1242, min_samples_split=2, min_samples_leaf=1), n_estimators=100)
#     - HistGradientBoostingClassifier(max_iter=343, max_leaf_nodes=623, learning_rate=0.1, min_samples_leaf=2, l2_regularization=0.)
# 
# 10. Backtest & plot confusion matrix
# 11. Save model
# 

# Load and convert raw data to weekly values, then add TA calcs

def return_week_summary_symbol(df):
    """
        Creates a weekly summary of the daily results for a symbol - assumes a datetime index
    """

    #  setup column defs
    ID_COLUMNS = ['symbol', 'GICSSector', 'GICSIndustryGroup', 'GICSIndustry']
    DATE_COLS = ['exDividendDate']

    median_cols = []
    median_cols.extend(ALL_CONTINUOUS_COLUMNS)
    median_cols.extend(DATE_COLS)
    # Remove val which isn't in df
    median_cols.remove('quoteDate_TIMESTAMP')

    sum_cols = ['totalVolume']

    mode_cols = []

    min_numeric_cols = ['weekLow']

    min_string_cols = []
    min_string_cols.extend(BOLLINGER_PREDICTION_COLUMNS)
    min_string_cols.extend(BOLLINGER_VALUE_COLUMNS)

    max_cols = ['weekHigh']
    first_cols = ['weekOpen']
    last_cols = ['weekClose']

    # remove high nan cols
    df.drop(HIGH_NAN_COLUMNS, axis=1, inplace=True, errors='ignore')

    # remove separate date col
    df.drop(['quoteDate'], axis=1, inplace=True, errors='ignore')

    # cols for special vals
    df['totalVolume'] = df['volume']
    df['weekLow'] = df['daysLow']
    df['weekHigh'] = df['daysHigh']
    df['weekOpen'] = df['previousClose']
    df['weekClose'] = df['adjustedPrice']

    if len(median_cols):
        median_resample = df[median_cols].resample('1W').median()
    else:
        median_resample = pd.DataFrame()

    if len(sum_cols):
        sum_resample = df[sum_cols].resample('1W').sum()
    else:
        sum_resample = pd.DataFrame()

    if len(mode_cols):
        mode_resample = pd.DataFrame()

        for col in ID_COLUMNS:
            temp_df = pd.DataFrame()
            temp_df[col] = df[col].resample('1W').apply(lambda x: (stats.mode(x, axis=None)[0][0]))
            mode_resample = pd.concat([mode_resample, temp_df], axis=1)
            
    else:
        mode_resample = pd.DataFrame()

    if len(min_numeric_cols):
        min_numeric_resample = df[min_numeric_cols].resample('1W').min()
    else:
        min_numeric_resample = pd.DataFrame()

    if len(min_string_cols):
        min_string_resample = df[min_string_cols].resample('1W').min()
    else:
        min_string_resample = pd.DataFrame()

    if len(max_cols):
        max_resample = df[max_cols].resample('1W').max()
    else:
        max_resample = pd.DataFrame()

    if len(first_cols):
        first_resample = df[first_cols].resample('1W').first()
    else:
        first_resample = pd.DataFrame()

    if len(last_cols):
        last_resample = df[last_cols].resample('1W').last()
    else:
        last_resample = pd.DataFrame()

        
    return pd.concat([median_resample, sum_resample, min_numeric_resample, min_string_resample, max_resample, 
                      first_resample, last_resample], axis=1)


def add_ta_values_to_df(df, feature_map):
    # Calculate ta features

    return add_all_ta_features(df, ta_open, ta_high, ta_low, ta_close, ta_volume, 
                               fillna=ta_fillna, colprefix=ta_colprefix)

def retrieve_and_calculate_weekly_symbol_dfs(path, date_str=None):
    """
        Retrieves the individual dataframes saved during pre-processing, calulates weekly values,
          adds Technical Analysis values.  It then divides the data into 70 / 15 / 15 as train / 
          validation / test and returns the three data sets
    """
    WHOLE_MARKET_TA = ['allordpreviousclose', 'asxpreviousclose','640106_A3597525W', 'FIRMMCRT', 'FXRUSD', 
                    'GRCPAIAD', 'GRCPAISAD', 'GRCPBCAD', 'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD', 
                    'H01_GGDPCVGDP', 'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP']


    all_dfs = []
    new_prediction_dfs = []
    whole_market_df = pd.DataFrame()
    
    # Create list of daily files to load
    print('Checking for files from', path)
    # Return files in path
    file_list = glob.glob(path + 'ml-symbol-*' + date_str + '.pkl.gz')
    print('Found', len(file_list),'symbol files')
    

    # Pre-defined files to look for
    for file in file_list:
        daily_symbol_df = pd.read_pickle(file, compression='gzip')
        whole_market_df = whole_market_df.append(daily_symbol_df[WHOLE_MARKET_COLUMNS])
        symbol = daily_symbol_df.iloc[0,:]['symbol']
        GICSSector = daily_symbol_df.iloc[0,:]['GICSSector']
        GICSIndustryGroup = daily_symbol_df.iloc[0,:]['GICSIndustryGroup']
        GICSIndustry = daily_symbol_df.iloc[0,:]['GICSIndustry']

        weekly_symbol_df = return_week_summary_symbol(daily_symbol_df)
        weekly_symbol_df = fix_duplicate_columns(weekly_symbol_df)
        weekly_symbol_df['symbol'] = symbol
        weekly_symbol_df['GICSSector'] = GICSSector
        weekly_symbol_df['GICSIndustryGroup'] = GICSIndustryGroup
        weekly_symbol_df['GICSIndustry'] = GICSIndustry
        
        num_recs = len(weekly_symbol_df)
        
        print(symbol,num_recs,'recs')
        
        if num_recs >= 28:
            complete_weekly_df = add_all_ta_features(weekly_symbol_df, 'weekOpen', 'weekHigh', 'weekLow', 
                                                     'weekClose', 'totalVolume', fillna=True, colprefix='ta_')
            # Create target column - 8 weeks in the future
            complete_weekly_df['target'] = (complete_weekly_df['adjustedPrice'].shift(-8) - complete_weekly_df['adjustedPrice'])  / complete_weekly_df['adjustedPrice'].clip(lower=0.1) * 100

            # Create dataset for values created in the laast 8 weeks - missing 8 week future predictions
            new_weekly_df = complete_weekly_df[complete_weekly_df['target'].isnull()]
            new_weekly_df = optimise_df(new_weekly_df)

            # Drop values without 8 week future prediction
            complete_weekly_df.dropna(subset=['target'], inplace=True)
            complete_weekly_df = optimise_df(complete_weekly_df)

            all_dfs.append(complete_weekly_df)
            new_prediction_dfs.append(new_weekly_df)
        else:
            print('Skipping',symbol,'less than 28 records')

    print('Consolidating whole market data')
    whole_market_df = whole_market_df.drop_duplicates()
    # Ensure there is only one record per day
    whole_market_df = whole_market_df.groupby('quoteDate').first()
    whole_market_df['quoteDate'] = whole_market_df.index

    # Combine into weekly data
    print('Combining into weekly data and adding ta')

    # convert into weekly vals for each col and add ta
    whole_market_weekly_dfs = []
    for col in WHOLE_MARKET_TA:
        print('Calculating weekly data for', col)
        resample = pd.DataFrame()
        resample[col + '_low'] = whole_market_df[col].resample('1W').min()
        resample[col + '_high'] = whole_market_df[col].resample('1W').max()
        resample[col + '_open'] = whole_market_df[col].resample('1W').first()
        resample[col + '_close'] = whole_market_df[col].resample('1W').last()
        resample[col + '_volume'] = 0

        print('Adding ta data for', col)
        resample = add_all_ta_features(resample, col + '_open', col + '_high', col + '_low', 
                                       col + '_close', col + '_volume', fillna=True, colprefix= col + '_ta_')
        
        resample.index.names = ['week_starting']
        whole_market_weekly_dfs.append(resample)

    print('Concatenating whole market data')
    whole_market_weekly_df = pd.concat(whole_market_weekly_dfs, axis=1)
    whole_market_weekly_df.index.names = ['week_starting']
    print('Concatenated whole market shape', whole_market_weekly_df.shape)
        
    print('Concatenating symbol dfs')
    # Create empty data frame
    all_df = pd.concat(all_dfs)
    all_df.index.names = ['week_starting']
    new_prediction_df = pd.concat(new_prediction_dfs)
    new_prediction_df.index.names = ['week_starting']
    
    print('Adding whole market data')
    all_df = all_df.merge(whole_market_weekly_df, how='left', left_on='week_starting', right_on='week_starting')
    new_prediction_df = new_prediction_df.merge(whole_market_weekly_df, how='left', left_on='week_starting', right_on='week_starting')
    
    print('Optimising symbol dfs')
    all_df = optimise_df(all_df)
    new_prediction_df = optimise_df(new_prediction_df)

    return all_df, new_prediction_df


def fix_duplicate_columns(df):
    # Get unique list of columms
    unique_cols = np.unique(df.columns.values)
    
    for col in unique_cols:
        column_numbers = [x for x in range(df.shape[1])]  # list of columns' integer indices
        remove_index = -1
        already_located = False

        for col_num in range(len(df.columns)):
            if df.columns[col_num] == col and already_located:
                remove_index = col_num
                print('Found duplicate for ', col, '- remove index', remove_index)
            elif df.columns[col_num] == col and not already_located:
                already_located = True
                
        # If a duplicate has been found, remove the column from the index list
        if remove_index != -1:
            column_numbers.remove(remove_index) #removing column integer index n
            df = df.iloc[:, column_numbers] #return all columns except the nth column

    return df


# Execute load and weekly data calcs
def load_data_and_calculate_weekly_dfs(export_str, run_str):
    train_df, prediction_df = retrieve_and_calculate_weekly_symbol_dfs('../data/symbols/', export_str)
    
    return train_df, prediction_df



# Remove cols which have high nans or no variability
def detect_cols_to_remove(train_df, prediction_df):

    # Check data types
    numeric_cols = []

    for col in train_df.columns:
        if train_df[col].dtype.name in ['object', 'category']:
            print(col, train_df[col].dtype.name)
        elif train_df[col].dtype.name != 'int8':
            numeric_cols.append(col)

    print(numeric_cols)

    # Remove high-nan and all 0 cols

    print('Number of columns:', train_df.shape[1])

    stats = pd.DataFrame()    
    stats["Mean"] = train_df.mean()
    stats["Std.Dev"] = train_df.std()
    stats["Var"] = train_df.var()
    stats["NaNs"] = train_df.isnull().sum()
    stats["NaN.Percent"] = stats["NaNs"] / train_df.shape[0] * 100

    cols_to_remove = stats[(stats['Mean']==0) & (stats['Std.Dev']==0) & (stats['Var']==0)]
    nan_cols_to_remove = stats[(stats['NaN.Percent'] > 75)]

    print(cols_to_remove.index.values)
    print(nan_cols_to_remove.index.values)

    train_df.drop(cols_to_remove.index.values, axis=1, inplace=True)
    train_df.drop(nan_cols_to_remove.index.values, axis=1, inplace=True)
    prediction_df.drop(cols_to_remove.index.values, axis=1, inplace=True)
    prediction_df.drop(nan_cols_to_remove.index.values, axis=1, inplace=True)
    
    dump(train_df.columns, 'models/data-prep-cols.joblib.z') 

    return train_df, prediction_df


def trim_cols(train_df, prediction_df):

    # Check data types
    numeric_cols = []

    for col in train_df.columns:
        if train_df[col].dtype.name in ['object', 'category']:
            print(col, train_df[col].dtype.name)
        elif train_df[col].dtype.name != 'int8':
            numeric_cols.append(col)

    print(numeric_cols)

    # Remove high-nan and all 0 cols

    print('Number of columns:', prediction_df.shape[1])

    cols_to_keep = load('models/data-prep-cols.joblib.z') 

    cols_to_remove = []

    # Check each col to see whether it should be kept - add other to remove list
    for col in train_df.columns:
        if col not in cols_to_keep:
            cols_to_remove.append(col)

    # Remove cols on remove list
    train_df.drop(cols_to_remove.index.values, axis=1, inplace=True)
    prediction_df.drop(cols_to_remove.index.values, axis=1, inplace=True)

    print(prediction_df.shape)

    return train_df, prediction_df


# Train and fit class encoders
def train_class_encoders(all_df, new_prediction_df):
    # Train and fit encoders

    # need to keep symbol for dividing up data later - so make copy of symbol for tranforming into a numeric
    all_df['symbolc'] = all_df['symbol']
    new_prediction_df['symbolc'] = new_prediction_df['symbol']

    # Fit on maximal data available
    symbol_encoder = LabelEncoder()
    symbol_encoder.fit(pd.concat([new_prediction_df['symbol'], all_df['symbol']], axis=0).values)

    GICSSector_encoder = LabelEncoder()
    GICSSector_encoder.fit(pd.concat([new_prediction_df['GICSSector'], all_df['GICSSector']], axis=0).values)

    GICSIndustryGroup_encoder = LabelEncoder()
    GICSIndustryGroup_encoder.fit(pd.concat([new_prediction_df['GICSIndustryGroup'], all_df['GICSIndustryGroup']], axis=0).values)

    GICSIndustry_encoder = LabelEncoder()
    GICSIndustry_encoder.fit(pd.concat([new_prediction_df['GICSIndustry'], all_df['GICSIndustry']], axis=0).values)

    # Transform training data
    all_df['symbolc'] = symbol_encoder.transform(all_df['symbolc'].values)
    all_df['GICSSector'] = GICSSector_encoder.transform(all_df['GICSSector'].values)
    all_df['GICSIndustryGroup'] = GICSIndustryGroup_encoder.transform(all_df['GICSIndustryGroup'].values)
    all_df['GICSIndustry'] = GICSIndustry_encoder.transform(all_df['GICSIndustry'].values)

    # Apply encoders to prediction data set
    new_prediction_df['symbolc'] = symbol_encoder.transform(new_prediction_df['symbolc'].values)
    new_prediction_df['GICSSector'] = GICSSector_encoder.transform(new_prediction_df['GICSSector'].values)
    new_prediction_df['GICSIndustryGroup'] = GICSIndustryGroup_encoder.transform(new_prediction_df['GICSIndustryGroup'].values)
    new_prediction_df['GICSIndustry'] = GICSIndustry_encoder.transform(new_prediction_df['GICSIndustry'].values)

    bollinger_prediction_encoders = []
    bollinger_value_encoders = []

    for col in BOLLINGER_PREDICTION_COLUMNS:
        bollinger_prediction_encoder = LabelEncoder()
        # Fit maximial set of values
        bollinger_prediction_encoder.fit(pd.concat([new_prediction_df[col], all_df[col]], axis=0).values)
        # Transform training data
        all_df[col] = bollinger_prediction_encoder.transform(all_df[col].values)
        # Transform prediction data
        new_prediction_df[col] = bollinger_prediction_encoder.transform(new_prediction_df[col].values)
        
        bollinger_prediction_encoders.append(bollinger_prediction_encoder)

    for col in BOLLINGER_VALUE_COLUMNS:
        bollinger_value_encoder = LabelEncoder()
        # Fit maximial set of values
        bollinger_value_encoder.fit(pd.concat([new_prediction_df[col], all_df[col]], axis=0).values)
        # Transform training data
        all_df[col] = bollinger_value_encoder.transform(all_df[col].values)
        # Transform prediction data
        new_prediction_df[col] = bollinger_value_encoder.transform(new_prediction_df[col].values)

        bollinger_value_encoders.append(bollinger_value_encoder)

    # Save encoders
    dump(symbol_encoder, 'models/symbol_encoder.joblib.z') 
    dump(GICSSector_encoder, 'models/GICSSector_encoder.joblib.z') 
    dump(GICSIndustryGroup_encoder, 'models/GICSIndustryGroup_encoder.joblib.z') 
    dump(GICSIndustry_encoder, 'models/GICSIndustry_encoder.joblib.z') 
    dump(bollinger_prediction_encoders, 'models/bollinger_prediction_encoder.joblib.z') 
    dump(bollinger_value_encoders, 'models/bollinger_value_encoder.joblib.z') 

    return all_df, new_prediction_df


def load_encode_classes(df):
    symbol_encoder = load('models/symbol_encoder.joblib.z') 
    GICSSector_encoder = load('models/GICSSector_encoder.joblib.z') 
    GICSIndustryGroup_encoder = load('models/GICSIndustryGroup_encoder.joblib.z') 
    GICSIndustry_encoder = load('models/GICSIndustry_encoder.joblib.z') 
    bollinger_prediction_encoders = load('models/bollinger_prediction_encoder.joblib.z') 
    bollinger_value_encoders = load('models/bollinger_value_encoder.joblib.z') 

    df['symbolc'] = df['symbol']    

    # Transform data
    df['symbolc'] = symbol_encoder.transform(df['symbolc'].values)
    df['GICSSector'] = GICSSector_encoder.transform(df['GICSSector'].values)
    df['GICSIndustryGroup'] = GICSIndustryGroup_encoder.transform(df['GICSIndustryGroup'].values)
    df['GICSIndustry'] = GICSIndustry_encoder.transform(df['GICSIndustry'].values)

    prediciton_encoders = bollinger_prediction_encoders.copy()
    value_encoders = bollinger_value_encoders.copy()

    for col in BOLLINGER_PREDICTION_COLUMNS:
        # Transform training data
        bollinger_prediction_encoder = prediciton_encoders.pop(0)
        df[col] = bollinger_prediction_encoder.transform(df[col].values)

    for col in BOLLINGER_VALUE_COLUMNS:
        # Transform training data
        bollinger_value_encoder = value_encoders.pop(0)
        df[col] = bollinger_value_encoder.transform(df[col].values)

    return df


# Train and apply standard scaler
def train_scaler(all_df, new_prediction_df, numeric_cols, target_col):
    # Scale all numeric columns excluding target col
    scale_cols = []

    for col in all_df.columns:
        if col in numeric_cols and col != target_col:
            scale_cols.append(col)
            
    print('Scaling', len(scale_cols), 'of', all_df.shape[1], 'cols')

    scaler = preprocessing.RobustScaler()

    # Fill any infinities, -infinites, NaNs which will cause scaler to fail
    all_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    all_df[scale_cols] = all_df[scale_cols].fillna(0)
    new_prediction_df[scale_cols] = new_prediction_df[scale_cols].fillna(0) 

    # Fit and transform from training set
    all_df[scale_cols] = scaler.fit_transform(all_df[scale_cols])

    # Apply scaler to prediction data set
    new_prediction_df[scale_cols] = scaler.transform(new_prediction_df[scale_cols])
    new_prediction_df[scale_cols] = new_prediction_df[scale_cols].fillna(0)

    dump(scaler, 'models/scaler.joblib.z') 

    return all_df, new_prediction_df


# Load and apply  scaler
def load_and_execute_scaler(df, numeric_cols, target_col):
    # Scale all numeric columns excluding target col
    scale_cols = []

    scaler = load('models/scaler.joblib.z') 

    for col in df.columns:
        if col in numeric_cols and col != target_col:
            scale_cols.append(col)
            
    print('Scaling', len(scale_cols), 'of', df.shape[1], 'cols')

    # Fill any infinities, -infinites, NaNs which will cause scaler to fail
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df[scale_cols] = df[scale_cols].fillna(0)

    # Transform data and fill any NAs
    df[scale_cols] = scaler.transform(df[scale_cols])
    df[scale_cols] = df[scale_cols].fillna(0)

    return df

def save_processed_data(train_df, prediction_df, run_str):
        # Export data
    train_df.to_pickle('../data/ml-ta-all-data-' + run_str + '.pkl.gz', compression='gzip')
    prediction_df.to_pickle('../data/ml-ta-new-prediction-data-' + run_str + '.pkl.gz', compression='gzip')


def load_processed_data(load_id):
    # Export data
    train_df = pd.read_pickle('../data/ml-ta-all-data-' + load_id + '.pkl.gz', compression='gzip')
    prediction_df = pd.read_pickle('../data/ml-ta-new-prediction-data-' + load_id + '.pkl.gz', compression='gzip')
    
    return train_df, prediction_df


def one_hot_encode_field(df, column_name, categories):
    new_cols = pd.get_dummies(df[column_name])
    new_cols = new_cols.astype('int8', errors='ignore')

    new_cols.T.reindex(categories).T.fillna(0)
    
    name_map = {}
    # rename the categories
    for val in categories:
        name_map[val] = val

    new_cols.rename(name_map, axis=1, inplace=True)
    
    # Remove the original column
    df.drop([column_name], axis=1, inplace=True)
    
    # Return df with new cols 
    return pd.concat([df, new_cols], axis=1)



def prep_class_index(df, column_name, categories):
    class_encoder = LabelEncoder()
    class_vals = df[column_name].get_values()
    class_vals = class_vals.reshape(-1, 1)
    class_encoder.fit(class_vals)
    df['class_index'] = class_encoder.transform(class_vals)
    
    # Return df with new target index col
    return df

def discretise_data(df, bins, bin_names):

    # prep data for training with one hot encoding
    discretised_df = df
    discretised_df['class'] = pd.cut(discretised_df['target'], bins=bins,labels=bin_names)
    discretised_df = prep_class_index(discretised_df, 'class', bin_names)
    discretised_df = one_hot_encode_field(discretised_df, 'class', bin_names)

def convert_to_class_targets(df, run_str, bins, bin_names):
    discretised_df = discretise_data(df, bins, bin_names)
    discretised_df.to_pickle('../data/ml-ta-discretised-training-data-' + run_str + '.pkl.gz', compression='gzip')
    return discretised_df, bin_names


def load_discretised_data(run_str):
    return pd.read_pickle('../data/ml-ta-discretised-training-data-' + run_str + '.pkl.gz', compression='gzip')


# Prepare a dataset for use in training or testing
def split_dataset(df, bin_names, train_proportion=1.):
    list_train_x = list()
    list_train_y_oh = list()
    list_train_y_le = list()
    list_train_symbols = list()

    list_test_x = list()
    list_test_y_oh = list()
    list_test_y_le = list()
    list_test_symbols = list()
    
    symbols = df['symbol'].unique()
    
    cols_to_drop = bin_names.copy()
    cols_to_drop.extend(['symbol', 'target', 'class', 'class_index'])

    for symbol in symbols:
        # Filter to model data for this symbol and re-set the pandas indexes
        model_data = df.loc[df['symbol'] == symbol]
        
        has_y = False
        
        if('target' in model_data.columns):
            has_y = True

        # If train proportion is less than 100%, divide data, otherwise make a copy
        if train_proportion < 1.:
            train_df = model_data.sample(frac=train_proportion)
            test_df = model_data.drop(train_df.index)
        else:
            train_df = model_data
            test_df = model_data
    
        
        # Create target data set - set to empty if no target (prediction set)
        if has_y:
            train_target_df_oh = train_df[bin_names]
            train_target_df_le = train_df['class_index']
            test_target_df_oh = test_df[bin_names]
            test_target_df_le = test_df['class_index']

        # Remove cols not in X array
        train_df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
        test_df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

        # number of features
        n_features = model_data.shape[1]

        # add records to training set 
        for val_num in range(len(train_df)-1):
            list_train_x.append(train_df.iloc[val_num:val_num+1, :].values)
            list_train_symbols.append(symbol)
            
            if has_y:
                list_train_y_oh.append(train_target_df_oh.iloc[val_num:val_num+1, :].values)
                list_train_y_le.append(train_target_df_le.iloc[val_num:val_num+1].values)

        # add records to test set 
        for val_num in range(len(test_df)-1):
            list_test_x.append(test_df.iloc[val_num:val_num+1, :].values)
            list_test_symbols.append(symbol)
            
            if has_y:
                list_test_y_oh.append(test_target_df_oh.iloc[val_num:val_num+1, :].values)
                list_test_y_le.append(test_target_df_le.iloc[val_num:val_num+1].values)

    train_x = array(list_train_x)
    train_symbols = array(list_train_symbols)
    
    if has_y:
        train_y_oh = array(list_train_y_oh)
        train_y_le = array(list_train_y_le)

    test_x = array(list_test_x)
    test_symbols = array(list_test_symbols)
    
    if has_y:
        test_y_oh = array(list_test_y_oh)
        test_y_le = array(list_test_y_le)
    
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[2])
    if has_y:
        train_y_oh = train_y_oh.reshape(train_y_oh.shape[0], train_y_oh.shape[2])
        train_y_le = train_y_le.reshape(train_y_le.shape[0])
    else:
        train_y_oh = None
        train_y_le = None
        
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[2])
    if has_y:
        test_y_oh = test_y_oh.reshape(test_y_oh.shape[0], test_y_oh.shape[2])
        test_y_le = test_y_le.reshape(test_y_le.shape[0])
    else:
        test_y_oh = None
        test_y_le = None
        
    return train_x, train_y_oh, train_y_le, train_symbols, test_x, test_y_oh, test_y_le, test_symbols


def train_voting_classifier(df):

    classifiers = [
        ('Random Forest', 
        RandomForestClassifier(max_depth=20, n_estimators=505, min_samples_split=2, min_samples_leaf=1)
        ),
        ('Adaboost Random Forest', 
        AdaBoostClassifier(RandomForestClassifier(max_depth=20, n_estimators=505, min_samples_split=2, min_samples_leaf=1)
                        , n_estimators=100)
        ),
        ('Histogram Gradient Boosting', 
        HistGradientBoostingClassifier(max_iter=343, max_leaf_nodes=623, learning_rate=0.1, min_samples_leaf=2, l2_regularization=0.)
        )
    ]

    # Use full training set (proportion 1)
    train_x, train_y_oh, train_y_le, train_symbols, test_x, test_y_oh, test_y_le, test_symbols = split_dataset(df, 1.)


    print('Training voting classifier with soft voting')
    voting_clf = VotingClassifier(estimators=classifiers, voting='soft')
    voting_clf = voting_clf.fit(train_x, train_y_le)

    return voting_clf


def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def predict_values(clf, X):
    print('Executing predictions')
    pred_y = clf.predict(X)
    return pred_y

def evaluate_classifier(test_y, pred_y, bin_names):
    print('Scoring classifier')
    score = accuracy_score(test_y, pred_y)

    print('Accuracy:', score)

    # Compute confusion matrix
    cm = confusion_matrix(test_y, pred_y)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, bin_names, title='Confusion matrix for classifier')

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix for voting classifier')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, bin_names, title='Normalized confusion matrix for voting classifier')

    print('-'*20)

def save_model(clf):
    dump(clf, 'models/clf.joblib.z') 


## Output prediction / actual results
def save_predictions(df, X_symbols, pred_y, run_str):
    # Create new dataframe using the index (week date) & symbol & prediction
    output_df = pd.Dataframe({
        'week': df.index,
        'symbol': X_symbols,
        'prediction': pred_y
    })

    file_location = './predictions/' + run_str + '.csv'
    print('Writing predictions to', file_location)
    output_df.to_csv(file_location)

# Execute voting classifier over prediction set
def main(**kwargs):
    """Runs the process of loading, preparing, training and evaluating a soft voting classifier
       Arguments:
        execution_type='training'
        run_str=value for recording this run
        export_str=value for the original file export for locating data files
        target_col=column name for target value
        execute_load_discretised_data=False
        execute_load_processed_data=False
        load_data_id=''
        execute_load_data_and_calculate_weekly_dfs=False
        load_processed_data=False
        bins=Array of values (None)
        bin_names=Array of strings (None)
        train_xgb=True
        train_industry_xgb=True
        train_deep_bagging=True
    """

    run_str = kwargs.get('execution_type', 'training')
    run_str = kwargs.get('run_str', '')
    export_str = kwargs.get('export_str', None)
    target_col = kwargs.get('target_col', None)
    execute_load_discretised_data = kwargs.get('execute_load_discretised_data', False)
    execute_load_processed_data = kwargs.get('load_processed_data', False)
    load_data_id = kwargs.get('load_data_id', '')
    execute_load_data_and_calculate_weekly_dfs = kwargs.get('execute_load_data_and_calculate_weekly_dfs', False)
    bins = kwargs.get('bins', None)
    bin_names = kwargs.get('bin_names', None)
    train_industry_xgb = kwargs.get('train_industry_xgb', True)
    train_bagging = kwargs.get('train_bagging', True)

    # If starting from beginning load & calculate weekly values
    if execute_load_data_and_calculate_weekly_dfs:
        load_data_and_calculate_weekly_dfs(run_str, export_str)
    

    if execution_type=='training':

    elif execute_load_processed_data:
        if train_cols_to_remove:
            detect_cols_to_remove
        else:
            load

        ml-ta-discretised-training-data-201908262001.pkl

    else: # Loading discretised data ready to go    

if __name__ == "__main__":
    main(run_str = datetime.now().strftime('%Y%m%d%H%M'),
        execution_type='training',
        export_str = '20190416',
        target_col = 'target',
        execute_load_data_and_calculate_weekly_dfs=False,
        train_cols_to_remove=False,
        train_label_encoders=False,
        train_scaler=False,
        execute_load_discretised_data=True, # Data which has been fully prepared with target classes for training
        execute_load_processed_data=False,  # Data which has been classified and scaled but no target classes
        load_data_id='201908262001',
        bins = [-1.0e10, -13.13461361, -3.00000238, 0.,9.25964718, 1.0e10],
        bin_names = ['strong_sell', 'sell', 'hold', 'buy', 'strong_buy'],

         )

    # Create run identifier
