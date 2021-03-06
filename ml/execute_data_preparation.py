import glob
import math
import gc
from timeit import default_timer as timer
from datetime import datetime, timedelta
import numba
import pandas as pd
import numpy as np
from processing_constants import LABEL_COLUMN, RETURN_COLUMN, PAST_RESULTS_DATE_REF_COLUMNS
from processing_constants import HIGH_NAN_COLUMNS, SYMBOL_COPY_COLUMNS, WHOLE_MARKET_COLUMNS
from processing_constants import SYMBOL_ZERO_COPY_COLUMNS
from optimise_dataframe import optimise_df
from print_logger import initialise_print_logger, print
from ensemble_processing import convert_date
from week_summary import return_week_summary

# S3 copy command to retrieve data
# aws s3 cp s3://{location} ./ --exclude "*" --include "companyQuotes-{DatePrefix}*" --recursive

OLD_RETURN_COLUMNS = ['Future1WeekDividend',
                      'Future1WeekPrice',
                      'Future1WeekReturn',
                      'Future1WeekRiskAdjustedReturn',
                      'Future2WeekDividend',
                      'Future2WeekPrice',
                      'Future2WeekReturn',
                      'Future2WeekRiskAdjustedReturn',
                      'Future4WeekDividend',
                      'Future4WeekPrice',
                      'Future4WeekReturn',
                      'Future4WeekRiskAdjustedReturn',
                      'Future8WeekDividend',
                      'Future8WeekPrice',
                      'Future8WeekReturn',
                      'Future8WeekRiskAdjustedReturn',
                      'Future12WeekDividend',
                      'Future12WeekPrice',
                      'Future12WeekReturn',
                      'Future12WeekRiskAdjustedReturn',
                      'Future26WeekDividend',
                      'Future26WeekPrice',
                      'Future26WeekReturn',
                      'Future26WeekRiskAdjustedReturn',
                      'Future52WeekDividend',
                      'Future52WeekPrice',
                      'Future52WeekReturn',
                      'Future52WeekRiskAdjustedReturn',
                      '1WeekVolatility',
                      '2WeekVolatility',
                      '4WeekVolatility',
                      '8WeekVolatility',
                      '12WeekVolatility',
                      '26WeekVolatility',
                      '52WeekVolatility',
                      '4WeekBollingerBandLower',
                      '4WeekBollingerBandUpper',
                      '4WeekBollingerPrediction',
                      '4WeekBollingerType',
                      '12WeekBollingerBandLower',
                      '12WeekBollingerBandUpper',
                      '12WeekBollingerPrediction',
                      '12WeekBollingerType'
                      ]


@numba.jit
def calculate_week_stats(num_weeks: int, df: pd.DataFrame, price_col: str):
    """
        Calculates the stats using a rolling window for the number of weeks specified.
    """
    rolling_expression = str(num_weeks * 7) + 'D'
    rolling_window = df[price_col].rolling(rolling_expression)
    prefix = return_prefix(num_weeks)
    df[prefix + '_week_min'] = rolling_window.min()
    df[prefix + '_week_max'] = rolling_window.max()
    df[prefix + '_week_mean'] = rolling_window.mean()
    df[prefix + '_week_median'] = rolling_window.median()
    df[prefix + '_week_std'] = rolling_window.std()
    df[prefix + '_week_bollinger_upper'] = rolling_window.mean() + \
        (2 * rolling_window.std())
    df[prefix + '_week_bollinger_lower'] = rolling_window.mean() - \
        (2 * rolling_window.std())

    # Set bollinger predictions: above -> within = falling, below -> within = rising
    bollinger_types = []
    bollinger_predictions = []
    bollinger_previous_value = ''

    # Create bollinger values dataframe
    bollinger_df = df[[price_col, prefix +
                       '_week_bollinger_upper', prefix + '_week_bollinger_lower']]

    # Loop through, calulcate bollinger value predictors and compare to previous result
    for row in bollinger_df.itertuples():
        current_price = row[1]
        bollinger_upper = row[2]
        bollinger_lower = row[3]

        if current_price > bollinger_upper:
            bollinger_value = 'Above'
        elif current_price < bollinger_lower:
            bollinger_value = 'Below'
        else:
            bollinger_value = 'Within'

        # Append bollinger value to type to list
        bollinger_types.append(bollinger_value)

        if (bollinger_value == 'Within') & (bollinger_previous_value == 'Above'):
            bollinger_predictions.append('Falling')
        elif (bollinger_value == 'Within') & (bollinger_previous_value == 'Below'):
            bollinger_predictions.append('Rising')
        else:
            bollinger_predictions.append('Steady')

        # Reset value for next iteration
        bollinger_previous_value = bollinger_value

    # Add results to dataframe
    df[prefix + '_week_bollinger_type'] = bollinger_types
    df[prefix + '_week_bollinger_prediction'] = bollinger_predictions

    return df


@numba.jit
def calculate_week_price_return(num_weeks: int, df: pd.DataFrame, price_col: str):
    """
        Calculates the return based on the comparison of price to num_weeks ago
    """

    df_col = df[price_col]
    index = df_col.index
    comparison_date = index - pd.DateOffset(weeks=num_weeks)
    prefix = return_prefix(num_weeks)

    # Create offset for number of weeks - this resets the index value
    ref_vals_df = pd.DataFrame()
    ref_vals_df[prefix + '_week_comparison_' +
                price_col] = df_col.asof(comparison_date)
    ref_vals_df[prefix + '_week_comparison_date'] = comparison_date

    # Reset the index value back to original
    ref_vals_df.index = index

    # concatenate the offset values with the original values
    combined_vals = pd.concat([df, ref_vals_df], axis=1)
    combined_vals[prefix + '_week_price_change'] = combined_vals[price_col] - combined_vals[
        prefix + '_week_comparison_' + price_col]
    combined_vals[prefix + '_week_price_return'] = combined_vals[prefix + '_week_price_change'] / combined_vals[
        price_col] * 100

    return combined_vals


@numba.jit
def calculate_period_dividends(df: pd.DataFrame, dividend_date_col: str,
                               dividend_payout_col: str, start_date: datetime,
                               end_date: datetime):
    """
        Calculates the dividends (if any) payed out during with the time period.
    """

    # Limit results to period
    ref_vals_df = df[(df.index >= start_date) & (df.index < end_date)]
    # Limit cols to dividens cols
    ref_vals_df = ref_vals_df[[dividend_date_col, dividend_payout_col]]
    # Drop duplicates
    unique_dividends = ref_vals_df.drop_duplicates()

    # Sum dividends which have an ex-dividend date within the time period
    total_dividends = 0

    for row in unique_dividends.itertuples():
        if (str(row[1]) >= start_date) & (str(row[1]) <= end_date):
            total_dividends = total_dividends + row[2]

    return total_dividends


@numba.jit
def return_dividends(dividend_period_start: datetime, dividend_period_end: datetime,
                     unique_dividends: pd.Series):
    """
        Sums the dividends within the specified time period
    """
    if unique_dividends.size == 0:
        return 0

    period_dividends = unique_dividends.loc[str(
        dividend_period_start): str(dividend_period_end)]
    dividends = period_dividends['exDividendPayout'].sum()
    if math.isnan(dividends):
        return 0
    else:
        return dividends


@numba.jit
def calculate_dividend_returns(num_weeks: int, df: pd.DataFrame,
                               price_col: str, dividend_date_col: str,
                               dividend_payout_col: str,
                               unique_dividends: pd.Series):
    """
        Calculates the dividends payed out in the last num_weeks.
    """
    prefix = return_prefix(num_weeks)
    dividend_column = []

    ref_vals_df = df[[dividend_date_col, dividend_payout_col]]

    # Create reference table with start and end dates for period
    dividend_dates = pd.DataFrame()
    dividend_dates['dividend_period_start'] = ref_vals_df.index - \
        pd.DateOffset(weeks=num_weeks)
    dividend_dates['dividend_period_end'] = ref_vals_df.index

    # Iterate through rows and add vals to dividend list
    # z1 = timer()
    for row in list(zip(dividend_dates['dividend_period_start'], dividend_dates['dividend_period_end'])):
        dividends = return_dividends(row[0], row[1], unique_dividends)
        dividend_column.append(dividends)

    # z2 = timer()
    # print('Calculating all dividends took: ', z2 - z1)

    transformed_df = df
    transformed_df[prefix + '_week_dividend_value'] = dividend_column
    transformed_df[prefix + '_week_dividend_return'] = transformed_df[prefix +
                                                                      '_week_dividend_value'] / transformed_df[price_col] * 100

    return transformed_df


@numba.jit
def generate_range_median(days_from: int, days_to: int, range_series: pd.Series):
    """
        Calculates a median value across a range of values prior to the current value
    """
    # Create empty dictionary
    df_dict = {}
    # Populate dictionary with values in series
    for x in range(days_from, (days_to+1)):
        df_dict['T' + str(x)] = range_series.shift(x)

    # Create data frame for values and return median
    temp_df = pd.DataFrame.from_dict(df_dict)
    return temp_df.median(axis=1, skipna=True)


@numba.jit
def append_recurrent_columns(df: pd.DataFrame):
    """
        Adds a series of columns representing previous values at different time ranges.
    """

    # Add extra columns for previous values, previous day, 2-5 day median, 6 - 10 median, 11-20 day median
    rec_df = df
    recurrent_columns = ['adjustedPrice', 'allordpreviousclose', 'asxpreviousclose', 'FXRUSD', 'FIRMMCRT',
                         'GRCPAIAD', 'GRCPAISAD', 'GRCPBCAD', 'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD',
                         'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP']

    # Add time shifted columns with proportion of change ####
    for rec_col in recurrent_columns:
        # Value yesterday
        rec_df[rec_col + '_T1P'] = (rec_df[rec_col] -
                                    rec_df[rec_col].shift(1)) / rec_df[rec_col].shift(1)

        # Create median for previous days (2-5)
        median_val = generate_range_median(2, 5, rec_df[rec_col])
        rec_df[rec_col + '_T2_5P'] = (rec_df[rec_col] -
                                      median_val) / median_val

        # Create median for previous days (6-10)
        median_val = generate_range_median(6, 10, rec_df[rec_col])
        rec_df[rec_col +
               '_T6_10P'] = (rec_df[rec_col] - median_val) / median_val

        # Create median for previous days (11-20)
        median_val = generate_range_median(11, 20, rec_df[rec_col])
        rec_df[rec_col +
               '_T11_20P'] = (rec_df[rec_col] - median_val) / median_val

    return rec_df


@numba.jit
def add_stats_and_returns(num_weeks: int, df: pd.DataFrame, price_col: str,
                          dividend_date_col: str, dividend_payout_col: str,
                          unique_dividends: pd.Series):
    """
        Executes processes to claculate all stats and return values for the last num_weeks
    """

    # Set prefix for week
    prefix = return_prefix(num_weeks)
    # t1 = timer()

    stats_df = df

    # Add stats cols
    stats_df = calculate_week_stats(num_weeks, stats_df, price_col)
    # t4 = timer()
    # print('Calculate weekly stats took: ', t4 - t1)

    # Calculate price returns
    stats_df = calculate_week_price_return(num_weeks, stats_df, price_col)
    # t5 = timer()
    # print('Calculate weekly price return took: ', t5 - t4)

    # Calculate dividend returns
    stats_df = calculate_dividend_returns(num_weeks, stats_df, price_col, dividend_date_col, dividend_payout_col,
                                          unique_dividends)
    # t6 = timer()
    # print('Calculate dividend return took: ', t6 - t5)

    # Add in total period returns
    stats_df[prefix + '_week_total_return'] = stats_df[prefix + '_week_dividend_return'] + stats_df[
        prefix + '_week_price_return']
    # t7 = timer()
    # print('Calculate total return took: ', t7 - t6)

    return stats_df


def return_prefix(week_num: int):
    """
        Converts a numeric value to a string value to use for a column name
    """
    # Change number to correct word prefix
    if week_num == 1:
        return 'one'
    elif week_num == 2:
        return 'two'
    elif week_num == 4:
        return 'four'
    elif week_num == 8:
        return 'eight'
    elif week_num == 12:
        return 'twelve'
    elif week_num == 26:
        return 'twenty_six'
    elif week_num == 52:
        return 'fifty_two'
    else:
        return str(week_num)


@numba.jit
def transform_symbol_returns(df: pd.DataFrame, symbol: str):
    """
        Executes the calculations for returns acrued for a specific symbol
    """
    # Set up config values for calculation
    return_weeks = [1, 2, 4, 8, 12, 26, 52]
    date_col = 'quoteDate'
    date_ref_col = date_col + '_ref'
    price_col = 'adjustedPrice'
    dividend_date_col = 'exDividendDate'
    dividend_payout_col = 'exDividendPayout'

    print('Processing returns for: ', symbol)
    t_1 = timer()

    # Reduce the data to the supplied symbol
    # transformed_df = df.loc[df['symbol'] == symbol, :]
    transformed_df = df
    # Ensure sorting is correct
    transformed_df.sort_values(by=['quoteDate'], inplace=True)
    # t2 = timer()
    # print('Reducing data to ' + symbol + ' took: ', t2 - t1)

    check_for_null_symbols(transformed_df)

    # Set the date index for the data frame
    transformed_df[date_ref_col] = transformed_df[date_col]
    transformed_df = transformed_df.set_index(date_ref_col)
    # t2a = timer()
    # print('Setting index for ' + symbol + ' took: ', t2a - t2)

    # Create reference table of unique dividends
    unique_dividends = transformed_df[[
        dividend_date_col, dividend_payout_col]].dropna().drop_duplicates()
    unique_dividends.set_index(['exDividendDate'], inplace=True)

    # Add stats and return columns by num of weeks
    for week in return_weeks:
        # print('Processing week: ', week)
        w_1 = timer()
        transformed_df = add_stats_and_returns(week, transformed_df, price_col, dividend_date_col, dividend_payout_col,
                                               unique_dividends)
        w_2 = timer()
        print('Week ' + str(week) + ' took: ', w_2 - w_1)

    # Add recurrent return columns
    transformed_df = append_recurrent_columns(transformed_df)

    transformed_df = optimise_df(transformed_df)

    t_3 = timer()
    print('Processing returns for symbol ' + symbol + ' took:', t_3 - t_1)
    return transformed_df


@numba.jit
def fill_symbol_empty_days(df: pd.DataFrame, symbol: str, reference_whole_market_data: pd.DataFrame):
    """
        Fills in gaps in symbol records.  Missing days are filled with the same price info 
        as previous days, the correct market refernce data for the day and a volume of 0.

        df: Pandas dataframe with all records
        symbol: string with symbol
        reference_whole_market_data: Pandas dataframe with the unique reference data for each day
    """
    print('Fill empty days for symbol: ', symbol)
    t_1 = timer()

    # Reduce the data to the supplied symbol
    symbol_df = df.loc[df['symbol'] == symbol, :]
    symbol_df.sort_values(by=['quoteDate'], inplace=True)
    symbol_df['qdIndex'] = symbol_df['quoteDate']
    symbol_df.set_index('qdIndex', inplace=True)
    symbol_min_date = symbol_df['quoteDate'].min().to_datetime64()
    symbol_max_date = symbol_df['quoteDate'].max().to_datetime64()

    check_for_null_symbols(symbol_df)

    # Keep reference data within symbol min and max
    mask = (reference_whole_market_data.index > symbol_min_date) & (
        reference_whole_market_data.index <= symbol_max_date)
    symbol_reference_data = reference_whole_market_data.loc[mask]
    symbol_reference_data['qdIndex'] = symbol_reference_data['quoteDate']
    symbol_reference_data.set_index('qdIndex', inplace=True)

    # Find missing days from reference set
    missing_days = symbol_reference_data[~symbol_reference_data.index.isin(
        symbol_df.index)]

    # Set the symbol values as the base for the merged frame
    merged_df = pd.DataFrame()
    merged_df = merged_df.append(symbol_df)

    # Find the correct references for each missing day
    missing_days['insertion_reference'] = symbol_df.index.searchsorted(
        missing_days.index) - 1

    # Iterate through the missing days and fill in the symbol based columns
    for row in missing_days.itertuples():
        # Construct a dict starting with the missing date
        new_row = {}
        new_row['symbol'] = symbol

        values_to_skip = ['Index', 'insertion_reference']

        for row_name in row._fields:
            # Check whether this field name should be skipped
            if row_name not in values_to_skip:
                new_row[row_name] = getattr(row, row_name)

        lookup_row = symbol_df.iloc[row.insertion_reference]

        # Loop through possible column names and copy in values, else set null
        for col_name in SYMBOL_COPY_COLUMNS:
            if col_name in lookup_row:
                new_row[col_name] = lookup_row[col_name]
            else:
                new_row[col_name] = np.nan

        # Loop through possible column names and copy in values, else set null
        for col_name in SYMBOL_ZERO_COPY_COLUMNS:
            new_row[col_name] = 0

        # Reset previousclose to be the same as adjustedprice
        new_row['previousClose'] = new_row['adjustedPrice']

        merged_df = merged_df.append(new_row, ignore_index=True)

    # Optimise the transformed data
    merged_df = optimise_df(merged_df)

    t_3 = timer()
    print('Processing missing days for symbol ' + symbol + ' took:', t_3 - t_1)
    return merged_df


def process_dataset(df: pd.DataFrame, symbols_to_skip, path, run_str):
    """
        Prepares a dataset for prediction by working through each symbol and adding
          returns and stats for each symbol and time period

            df: dataframe with complete dataset
            symbols_to_skip: list with any symbols to skip over processing
            path: path to save symbol data sets
            run_str: identifier for data run
    """
    print('Dropping unused columns and setting date/time types and index')
    columns_to_drop = OLD_RETURN_COLUMNS
    columns_to_drop.extend(HIGH_NAN_COLUMNS)
    df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
    df['quoteDate'] = pd.to_datetime(df['quoteDate'], errors='coerce')
    df['exDividendDate'] = pd.to_datetime(
        df['exDividendDate'], errors='coerce')

    print('Making ex-dividend date a relative number')
    df['exDividendRelative'] = df['exDividendDate'] - df['quoteDate']

    # convert string difference value to integer
    df['exDividendRelative'] = df['exDividendRelative'].apply(
        lambda x: np.nan if pd.isnull(x) else x.days)

    # Make sure it is the minimum data type size
    df.loc[:, 'exDividendRelative'] = df['exDividendRelative'].astype(
        'int32', errors='ignore')

    zero_columns = ['change', 'changeInPercent', 'changeFrom52WeekHigh',
                    'changeFrom52WeekLow', 'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow',
                    'allordchange', 'allordpercebtChangeFrom52WeekHigh', 'allordpercentChangeFrom52WeekLow',
                    'asxchange', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow']
    print('Fixing missing values as 0 for:', zero_columns)
    df[zero_columns].fillna(0, inplace=True)

    print('Fill missing adjustedPrice values')
    # df['adjustedPrice'] = df.apply(
    #     lambda row: row['lastTradePriceOnly'] if np.isnan(
    #         row['adjustedPrice']) else row['adjustedPrice'],
    #     axis=1
    # )

    df['adjustedPrice'].fillna(df['lastTradePriceOnly'])

    check_for_null_symbols(df)

    symbols = df['symbol'].drop_duplicates()

    # Output symbol list to csv
    symbol_list = symbols.dropna().to_frame()
    symbol_list.to_csv(path + run_str + '-symbols-list.csv')
    del symbol_list

    # ######## TEMP ###########
    # symbols = symbols.head(100)
    # ########################

    num_symbols = symbols.shape[0]
    symbol_count = 0

    # Determine all possible dates for symbol records and the
    # overall min and max dates for all records
    # all_possible_dates = df['quoteDate'].dropna().drop_duplicates()
    # overall_min_date = all_possible_dates.min()[0]
    # overall_max_date = all_possible_dates.max()[0]

    # Retrieve a unique reference set of whole market data , e.g. all ords / asx
    # for each day with records
    reference_whole_market_data = df[WHOLE_MARKET_COLUMNS].drop_duplicates()
    # Ensure there is only one record per day
    reference_whole_market_data = reference_whole_market_data.groupby(
        'quoteDate').first()
    reference_whole_market_data['quoteDate'] = reference_whole_market_data.index


    s_1 = timer()
    for symbol in symbols:
        if symbol not in symbols_to_skip:
            print(80*'-')
            symbol_df = fill_symbol_empty_days(
                df, symbol, reference_whole_market_data)

            print(80*'-')
            symbol_df = transform_symbol_returns(symbol_df, symbol)

            check_for_null_symbols(symbol_df)

            # Remove extra columns created during calculations
            symbol_df.drop(PAST_RESULTS_DATE_REF_COLUMNS, axis=1, inplace=True, errors='ignore')

            symbol_df = optimise_df(symbol_df)

            # Save pickle data
            symbol_df.to_pickle(path + 'ml-symbol-' + symbol + '-' + run_str + '.pkl.gz', compression='gzip')

            weekly_symbol_df = return_week_summary(symbol_df)

            weekly_symbol_df.to_pickle(path + 'ml-weekly-symbol-' + symbol + '-' + run_str + '.pkl.gz', compression='gzip')

            symbol_count = symbol_count + 1
            # Every tenth symbol see how things are going
            if symbol_count % 100 == 0:
                s_2 = timer()
                elapsed = s_2 - s_1
                print(80*'-')
                print(str(symbol_count) + ' of ' + str(num_symbols) +
                    ' completed.  Elapsed time: ' + str(elapsed))
                #   ' completed.  Elapsed time: ' + str(datetime.timedelta(seconds=elapsed)))
                print(80 * '-')

    print('Retrieving symbol dfs')
    output_df = retrieve_symbol_dfs(path, run_str)


    print('Number of "NA" symbols in concatenated df:',
          output_df[output_df['symbol'] == 'NA'].shape[0])

    print('Number of "NA" symbols in optimised df:',
          output_df[output_df['symbol'] == 'NA'].shape[0])

    
    return output_df

def retrieve_symbol_dfs(path, run_str):
    """
        Retrieves the infividual dataframes saved during pre-processing and returns them 
    """
    symbol_dfs = []

    print('Checking for files from', path)
    # Return files in path
    file_list = glob.glob(path + 'ml-symbol-*' + run_str + '.pkl.gz')

    # load each model set
    for file_name in file_list:
        # retrieve data frame and append to list
        symbol_df = pd.read_pickle(file_name, compression='gzip')
        symbol_df = optimise_df(symbol_df)

        symbol_dfs.append(symbol_df)

    print('Concatenating symbol dfs')
    # Create empty data frame
    output_df = pd.concat(symbol_dfs)

    print('Optimising symbol dfs')
    output_df = optimise_df(output_df)

    return output_df


def append_industry_categories(df: pd.DataFrame, catgories_df: pd.DataFrame):
    """
        Adds the industry categpry (GICS) values to the dataset by matching on symbol
    """

    print('Adding industry categories to data set')
    output_df = df.merge(catgories_df, left_on='symbol',
                         right_on='symbol', how='left')
    
    category_cols = catgories_df.columns.values

    print('Checking for any null columns')
    has_nulls = output_df[category_cols].isnull().any().any()
    if has_nulls:
        print('WARNING - Found nulls vals')
        print(output_df[output_df[category_cols].isnull().any(axis=1)])
        raise('Symbols missing matching category records')
    else:
        print('No nulls found')

    return output_df


def load_data(base_path: str, no_files: int):
    """
        Loads a series of data export files
    """

    print('Loading', no_files, 'files for:', base_path)
    # Set the file increments range
    increments = range(1, (no_files + 1))
    # Create empty data frame
    loading_data = pd.DataFrame()
    # Loop through each file and append to data frame
    for increment in increments:
        path = base_path % increment
        frame = pd.read_csv(path, compression='gzip', parse_dates=[
                            'quoteDate', 'exDividendDate'], infer_datetime_format=True, low_memory=False)
        loading_data = loading_data.append(frame, ignore_index=True)
        del frame
        print('Loaded:', path)

    return loading_data


@numba.jit
def generate_label_column(df, num_weeks, reference_date, date_col):
    """ Generate a label column for each record num_weeks in the future
        then reduce data set to values which occurr <= (reference_date - num_weeks)
     """

    # Retrieve unique symbols
    symbols = df['symbol'].unique()
    symbol_counter = 0

    date_ref_col = date_col + '_ref'

    # Array to hold completed dataframes
    symbol_dfs = []

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
    # converted_ref_date = datetime.strptime(reference_date, '%Y-%m-%d')

    # filter dataframe
    # output_df = output_df.loc[output_df[LABEL_COLUMN +
    #                                     '_date'] <= converted_ref_date]
    # Remove extra column with future date reference
    output_df.drop([LABEL_COLUMN + '_date'], axis=1, inplace=True)

    # Use most efficient storage for memory
    output_df.loc[:, LABEL_COLUMN] = output_df[LABEL_COLUMN].astype(
        'float32', errors='ignore')

    return output_df


@numba.jit
def generate_tminus_values(df):
    """
        Generate return medians for previous 10, 20 & 40 days
    """

    print('Generating median returns for previous 10, 20 & 40 days',)

    # Retrieve unique symbols
    symbols = df['symbol'].drop_duplicates()
    symbol_counter = 0

    # Array to hold completed dataframes
    symbol_dfs = []

    for symbol in symbols:
        gc.collect()
        symbol_counter = symbol_counter + 1
        # print(80 * '-')
        print('Generating tminus values for:', symbol, 'num:', symbol_counter)
        # Retrieve data for symbol and sort by date
        symbol_df = df.loc[df['symbol'] == symbol, :]
        symbol_df.sort_values(by=['quoteDate'], inplace=True)

        # Generate median value for past 1 - 10 days
        median_val = generate_range_median(1, 10, symbol_df[LABEL_COLUMN])
        symbol_df[LABEL_COLUMN + '_T1_10P'] = median_val

        # Generate median value for past 1 - 20 days
        median_val = generate_range_median(1, 20, symbol_df[LABEL_COLUMN])
        symbol_df[LABEL_COLUMN + '_T1_20P'] = median_val

        # Generate median value for past 1 - 40 days
        median_val = generate_range_median(1, 40, symbol_df[LABEL_COLUMN])
        symbol_df[LABEL_COLUMN + '_T1_40P'] = median_val

        # Append dataframe to dataframe array
        symbol_dfs.append(symbol_df)

    # Generate concatenated dataframe
    output_df = pd.concat(symbol_dfs)

    # Use most efficient storage for memory
    output_df.loc[:, LABEL_COLUMN + '_T1_10P'] = output_df[LABEL_COLUMN + '_T1_10P'].astype(
        'float32', errors='ignore')

    output_df.loc[:, LABEL_COLUMN + '_T1_20P'] = output_df[LABEL_COLUMN + '_T1_20P'].astype(
        'float32', errors='ignore')

    output_df.loc[:, LABEL_COLUMN + '_T1_40P'] = output_df[LABEL_COLUMN + '_T1_40P'].astype(
        'float32', errors='ignore')

    return output_df


@numba.jit
def divide_labelled_unlabelled(df, num_weeks, reference_date):
    """
        Generate two datasets.
        Reference period = (reference_date - num_weeks)
        labelled - keep data before reference period with a label
        unlabelled - remove all data which is before the reference period.
    """
    print('Calculating reference date comparison, reference_date:',
          reference_date, ', num_weeks:', num_weeks)

    weeks_delta = timedelta(weeks=num_weeks)
    converted_ref_date = datetime.strptime(reference_date, '%Y-%m-%d')
    comparison_date = converted_ref_date - weeks_delta
    print('Calculated comparison date:', comparison_date)

    labelled_df = df.loc[df.index <= comparison_date]
    print('Retaining', len(labelled_df.index), 'labelled records')

    unlabelled_df = df.loc[df.index > comparison_date]
    print('Retaining', len(unlabelled_df.index), 'unlabelled records')

    return labelled_df, unlabelled_df


def check_for_null_symbols(df):
    if 'symbol' in df.columns:
        print(df['symbol'].dtype)
        print('Number of null symbols:', df['symbol'].isnull().sum())
        print('Number of "NA" symbols:', df[df['symbol'] == 'NA'].shape[0])
    else:
        print('Symbol column not found')

def return_existing_symbol_list(path, run_str):
    """Returns any existing file for the symbol / run_str combination in a set path
       Arguments:
        path
        run_str
    """
    existing_symbol_list = []

    print('Checking for files from', path)
    # Return files in path
    file_list = glob.glob(path + 'ml-symbol-*' + run_str + '.pkl.gz')

    # load each model set
    for file_name in file_list:
        # remove leading path and trailing file extension
        symbol = file_name.replace(
            path, '').replace('ml-symbol-', '').replace('-' + run_str + '.pkl.gz', '')

        # create model property and load model set into it
        existing_symbol_list.append(symbol)

    return existing_symbol_list


def main(**kwargs):
    """Runs the process of loading, pre-processing and generating labels for the raw sharecast data
       Arguments:
        run_str=None
        load_individual_data_files=True
        base_path=None
        add_industry_categories=True
        symbol_industry_path=None
        no_files=None
        generate_labels=True
        generate_tminus_labels=True
        generate_label_weeks=None
        labelled_file_name=None
        unlabelled_file_name=None
        reference_date=None
        unlabelled_data_file=None
        labelled_data_file=None
        skip_existing_symbols=False
        existing_symbols_path='data/'
    """

    run_str = kwargs.get('run_str', None)
    load_individual_data_files = kwargs.get('load_individual_data_files', True)
    base_path = kwargs.get('base_path', None)
    add_industry_categories = kwargs.get('add_industry_categories', True)
    symbol_industry_path = kwargs.get('symbol_industry_path', None)
    no_files = kwargs.get('no_files', None)
    generate_labels = kwargs.get('generate_labels', True)
    generate_tminus_labels = kwargs.get('generate_tminus_labels', True)
    generate_label_weeks = kwargs.get('generate_label_weeks', None)
    reference_date = kwargs.get('reference_date', None)
    skip_existing_symbols = kwargs.get('reference_date', False)
    existing_symbols_path = kwargs.get('existing_symbols_path', 'data/')

    initialise_print_logger('logs/data-preparation-' +
                            datetime.now().strftime('%Y%m%d%H%M') + '.log')

    # Check whether to load individual files or an aggregated file
    if load_individual_data_files:
        print('Loading individual data files')
        l_1 = timer()
        df = load_data(base_path, no_files)
        gc.collect()
        print('Optimising data')
        df = optimise_df(df)
        print('Saving data')
        df.to_pickle(
            'data/ml-' + run_str + '-data.pkl.gz', compression='gzip')
        l_2 = timer()
        print('Loading and saving individual data took:', l_2 - l_1)
    else:
        print('Loading aggregated data files')
        l_1 = timer()
        df = pd.read_pickle(
            'data/ml-' + run_str + '-data.pkl.gz', compression='gzip')
        l_2 = timer()
        print('Loading aggregated data file took:', l_2 - l_1)

    print(df['symbol'].unique())

    symbols_to_skip = []
    # if skip existing symbols, check for existing symbols in data location and add to list
    if skip_existing_symbols:
        symbols_to_skip = return_existing_symbol_list(existing_symbols_path, run_str)

    # Load the industry-symbol values
    if add_industry_categories:
        categories_df = pd.read_csv(symbol_industry_path)
        df = append_industry_categories(df, categories_df)

    check_for_null_symbols(df)

    # Processing data
    print('Processing returns and recurrent columns for each symbol')
    processed_data = process_dataset(df, symbols_to_skip, existing_symbols_path, run_str)

    check_for_null_symbols(processed_data)

    # generate labels if required
    if generate_labels and generate_label_weeks and reference_date:
        processed_data = generate_label_column(
            processed_data, generate_label_weeks, reference_date, 'quoteDate')
        gc.collect()

        check_for_null_symbols(processed_data)

        # Generate t-n for label data
        if generate_tminus_labels:
            processed_data = generate_tminus_values(processed_data)
            check_for_null_symbols(processed_data)

        # split data into labelled and unlabelled
        labelled_df, unlabelled_df = divide_labelled_unlabelled(
            processed_data, generate_label_weeks, reference_date)

        labelled_file_name = 'data/ml-' + run_str + '-labelled.pkl.gz'
        print('Saving labelled data to:', labelled_file_name)
        labelled_df.to_pickle(labelled_file_name, compression='gzip')

        unlabelled_file_name = 'data/ml-' + run_str + '-unlabelled.pkl.gz'
        print('Saving processed data to:', unlabelled_file_name)
        unlabelled_df.to_pickle(unlabelled_file_name, compression='gzip')

    check_for_null_symbols(processed_data)

    gc.collect()
    proc_file_name = 'data/ml-' + run_str + '-processed.pkl.gz'
    print('Saving processed data to:', proc_file_name)
    processed_data.to_pickle(proc_file_name, compression='gzip')

    print(80*'-')
    print('Data processing finished')


if __name__ == "__main__":
    # run_str = datetime.now().strftime('%Y%m%d')
    RUN_STR = '20190416'

    main(run_str=RUN_STR,
         load_individual_data_files=False,
         base_path='data/companyQuotes-20190416-%03d.csv.gz',
         add_industry_categories=True,
         symbol_industry_path='data/symbol-industry-lookup.csv',
         no_files=32,
         generate_labels=True,
         generate_label_weeks=8,
         reference_date='2019-04-13',
         generate_tminus_labels=True,
         skip_existing_symbols = True,
         existing_symbols_path = 'data/symbols/'
         )
