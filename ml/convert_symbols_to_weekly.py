from scipy import stats
import numpy as np
import pandas as pd
from processing_constants import ALL_CONTINUOUS_COLUMNS, HIGH_NAN_COLUMNS
from processing_constants import BOLLINGER_PREDICTION_COLUMNS, BOLLINGER_VALUE_COLUMNS
from optimise_dataframe import optimise_df
from processing_constants import WHOLE_MARKET_COLUMNS
import glob
import ta

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


def return_week_summary_symbol(df):
    """
        Creates a weekly summary of the daily results for a symbol - assumes a datetime index
    """

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
            temp_df[col] = df[col].resample('1W').apply(
                lambda x: (stats.mode(x, axis=None)[0][0]))
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


WHOLE_MARKET_TA = ['allordpreviousclose', 'asxpreviousclose', '640106_A3597525W', 'FIRMMCRT', 'FXRUSD',
                   'GRCPAIAD', 'GRCPAISAD', 'GRCPBCAD', 'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD',
                   'H01_GGDPCVGDP', 'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP']


def fix_duplicate_columns(df):
    # Get unique list of columms
    unique_cols = np.unique(df.columns.values)

    for col in unique_cols:
        # list of columns' integer indices
        column_numbers = [x for x in range(df.shape[1])]
        remove_index = -1
        already_located = False

        for col_num in range(len(df.columns)):
            if df.columns[col_num] == col and already_located:
                remove_index = col_num
                print('Found duplicate for ', col,
                      '- remove index', remove_index)
            elif df.columns[col_num] == col and not already_located:
                already_located = True

        # If a duplicate has been found, remove the column from the index list
        if remove_index != -1:
            # removing column integer index n
            column_numbers.remove(remove_index)
            # return all columns except the nth column
            df = df.iloc[:, column_numbers]

    return df


def retrieve_and_calculate_weekly_symbol_dfs(path, date_str=None):
    """
        Retrieves the individual dataframes saved during pre-processing, calulates weekly values,
          adds Technical Analysis values.  It then divides the data into 70 / 15 / 15 as train / 
          validation / test and returns the three data sets
    """
    all_dfs = []
    new_prediction_dfs = []
    whole_market_df = pd.DataFrame()

    # Create list of daily files to load
    print('Checking for files from', path)
    # Return files in path
    file_list = glob.glob(path + 'ml-symbol-*' + date_str + '.pkl.gz')
    print('Found', len(file_list), 'symbol files')

    # Pre-defined files to look for
    for file in file_list:
        daily_symbol_df = pd.read_pickle(file, compression='gzip')
        whole_market_df = whole_market_df.append(
            daily_symbol_df[WHOLE_MARKET_COLUMNS])
        symbol = daily_symbol_df.iloc[0, :]['symbol']
        GICSSector = daily_symbol_df.iloc[0, :]['GICSSector']
        GICSIndustryGroup = daily_symbol_df.iloc[0, :]['GICSIndustryGroup']
        GICSIndustry = daily_symbol_df.iloc[0, :]['GICSIndustry']

        weekly_symbol_df = return_week_summary_symbol(daily_symbol_df)
        weekly_symbol_df = fix_duplicate_columns(weekly_symbol_df)
        weekly_symbol_df['symbol'] = symbol
        weekly_symbol_df['GICSSector'] = GICSSector
        weekly_symbol_df['GICSIndustryGroup'] = GICSIndustryGroup
        weekly_symbol_df['GICSIndustry'] = GICSIndustry

        num_recs = len(weekly_symbol_df)

        print(symbol, num_recs, 'recs')

        if num_recs >= 28:
            complete_weekly_df = ta.add_all_ta_features(weekly_symbol_df, 'weekOpen', 'weekHigh', 'weekLow',
                                                        'weekClose', 'totalVolume', fillna=True, colprefix='ta_')
            # Create target column - 8 weeks in the future
            complete_weekly_df['target'] = (complete_weekly_df['adjustedPrice'].shift(
                -8) - complete_weekly_df['adjustedPrice']) / complete_weekly_df['adjustedPrice'].clip(lower=0.1) * 100

            # Create dataset for values created in the laast 8 weeks - missing 8 week future predictions
            new_weekly_df = complete_weekly_df[complete_weekly_df['target'].isnull(
            )]
            new_weekly_df = optimise_df(new_weekly_df)

            # Drop values without 8 week future prediction
            complete_weekly_df.dropna(subset=['target'], inplace=True)
            complete_weekly_df = optimise_df(complete_weekly_df)

            all_dfs.append(complete_weekly_df)
            new_prediction_dfs.append(new_weekly_df)
        else:
            print('Skipping', symbol, 'less than 28 records')

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
        resample = ta.add_all_ta_features(resample, col + '_open', col + '_high', col + '_low',
                                          col + '_close', col + '_volume', fillna=True, colprefix=col + '_ta_')

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
    all_df = all_df.merge(whole_market_weekly_df, how='left',
                          left_on='week_starting', right_on='week_starting')
    new_prediction_df = new_prediction_df.merge(
        whole_market_weekly_df, how='left', left_on='week_starting', right_on='week_starting')

    print('Optimising symbol dfs')
    all_df = optimise_df(all_df)
    new_prediction_df = optimise_df(new_prediction_df)

    return all_df, new_prediction_df
