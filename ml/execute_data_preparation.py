import pandas as pd
from timeit import default_timer as timer
import math
import numba
from xgboost_general_symbol_ensemble_sharecast import *
from optimise_dataframe import *

# S3 copy command to retrieve data
# aws s3 cp s3://{location} ./ --exclude "*" --include "companyQuotes-{DatePrefix}*" --recursive""

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
def calculate_week_stats(num_weeks, df, price_col):
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
def calculate_week_price_return(num_weeks, df, price_col):
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
def calculate_period_dividends(df, dividend_date_col, dividend_payout_col, start_date, end_date):
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
def return_dividends(dividend_period_start, dividend_period_end, unique_dividends):
    period_dividends = unique_dividends.loc[dividend_period_start: dividend_period_end]
    dividends = period_dividends['exDividendPayout'].sum()
    if math.isnan(dividends):
        return 0
    else:
        return dividends


@numba.jit
def calculate_dividend_returns(num_weeks, df, price_col, dividend_date_col, dividend_payout_col, unique_dividends):
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
def generate_range_median(days_from, days_to, range_series):
    # Create empty dictionary
    df_dict = {}
    # Populate dictionary with values in series
    for x in range(days_from, (days_to+1)):
        df_dict['T' + str(x)] = range_series.shift(x)

    # Create data frame for values and return median
    temp_df = pd.DataFrame.from_dict(df_dict)
    return temp_df.median(axis=1)


@numba.jit
def append_recurrent_columns(df):
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
def add_stats_and_returns(num_weeks, df, price_col, dividend_date_col, dividend_payout_col, unique_dividends):
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


def return_prefix(week_num):
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
def transform_symbol_returns(df, symbol):
    # Set up config values for calculation
    return_weeks = [1, 2, 4, 8, 12, 26, 52]
    date_col = 'quoteDate'
    date_ref_col = date_col + '_ref'
    price_col = 'adjustedPrice'
    dividend_date_col = 'exDividendDate'
    dividend_payout_col = 'exDividendPayout'

    print('Processing symbol: ', symbol)
    t1 = timer()

    # Reduce the data to the supplied symbol
    transformed_df = df.loc[df['symbol'] == symbol, :]
    transformed_df.sort_values(by=['quoteDate'], inplace=True)
    # t2 = timer()
    # print('Reducing data to ' + symbol + ' took: ', t2 - t1)

    # Set the date index for the data frame
    transformed_df[date_ref_col] = transformed_df[date_col]
    transformed_df = transformed_df.set_index(date_ref_col)
    # t2a = timer()
    # print('Setting index for ' + symbol + ' took: ', t2a - t2)

    # Create reference table of unique dividends
    unique_dividends = transformed_df[[
        dividend_date_col, dividend_payout_col]].drop_duplicates()
    unique_dividends.set_index(['exDividendDate'])

    # Add stats and return columns by num of weeks
    for week in return_weeks:
        # print('Processing week: ', week)
        w1 = timer()
        transformed_df = add_stats_and_returns(week, transformed_df, price_col, dividend_date_col, dividend_payout_col,
                                               unique_dividends)
        w2 = timer()
        print('Week ' + str(week) + ' took: ', w2 - w1)

    # Add recurrent return columns
    transformed_df = append_recurrent_columns(transformed_df)

    transformed_df = optimise_df(transformed_df)

    t3 = timer()
    print('Processing symbol ' + symbol + ' took:', t3 - t1)
    return transformed_df


def process_dataset(df):
    print('Dropping unused columns and setting date/time types and index')
    columns_to_drop = OLD_RETURN_COLUMNS
    columns_to_drop.extend(HIGH_NAN_COLUMNS)
    df.drop(columns_to_drop, axis=1, inplace=True)
    df['quoteDate'] = pd.to_datetime(df['quoteDate'], errors='coerce')
    df['exDividendDate'] = pd.to_datetime(
        df['exDividendDate'], errors='coerce')

    symbols = df['symbol'].unique()

    # ######## TEMP ###########
    # symbols = symbols.head(20)
    # ########################

    num_symbols = symbols.shape[0]
    symbol_count = 0

    # Create list for symbol results
    symbol_dfs = []

    s1 = timer()
    for symbol in symbols:
        print(80*'-')
        symbol_df = transform_symbol_returns(df, symbol)

        # Add data frame into all results
        symbol_dfs.append(symbol_df)

        symbol_count = symbol_count + 1
        # Every tenth symbol see how things are going
        if symbol_count % 100 == 0:
            s2 = timer()
            elapsed = s2 - s1
            print(80*'-')
            print(str(symbol_count) + ' of ' + str(num_symbols) +
                  ' completed.  Elapsed time: ' + str(elapsed))
            #   ' completed.  Elapsed time: ' + str(datetime.timedelta(seconds=elapsed)))
            print(80 * '-')

    # Create empty data frame
    output_df = pd.concat(symbol_dfs)

    output_df = optimise_df(output_df)

    # Remove extra columns created during calculations
    df.drop(PAST_RESULTS_DATE_REF_COLUMNS,
            axis=1, inplace=True, errors='ignore')

    return output_df


def add_industry_categories(df, catgories_df):
    print('Adding industry categories to data set')
    output_df = df.merge(catgories_df, left_on='symbol',
                         right_on='symbol', how='left')

    return output_df


def load_data(base_path, no_files):
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


def main(load_config):
    # Check whether to load individual files or an aggregated file
    if load_config.get('load_individual_data_files') is True:
        print('Loading individual data files')
        l1 = timer()
        df = load_data(load_config.get('base_path'),
                       load_config.get('no_files'))
        gc.collect()
        print('Optimising data')
        df = optimise_df(df)
        print('Saving data')
        df.to_pickle(
            'data/ml-' + load_config.get('run_str') + '-data.pkl.gz', compression='gzip')
        l2 = timer()
        print('Loading and saving individual data took:', l2 - l1)
    else:
        print('Loading aggregated data files')
        l1 = timer()
        df = pd.read_pickle(
            'data/ml-' + load_config.get('run_str') + '-data.pkl.gz', compression='gzip')
        l2 = timer()
        print('Loading aggregated data file took:', l2 - l1)

    # Load the industry-symbol values
    if load_config.get('add_industry_categories') is True:
        categories_df = pd.read_csv(load_config.get('symbol_industry_path'))
        df = add_industry_categories(df, categories_df)

    # Processing data
    print('Processing returns and recurrent columns for each symbol')
    processed_data = process_dataset(df)

    print('Saving processed data')
    processed_data.to_pickle(
        'data/ml-' + load_config.get('run_str') + '-processed.pkl.gz', compression='gzip')

    print(80*'-')
    print('Data processing finished')


if __name__ == "__main__":
    # run_str = datetime.now().strftime('%Y%m%d')
    RUN_STR = '20180512'

    LOAD_CONFIG = {
        'run_str': RUN_STR,
        'load_individual_data_files': True,
        'base_path': 'data/companyQuotes-20180512-%03d.csv.gz',
        'add_industry_categories': True,
        'symbol_industry_path': 'data/symbol-industry-lookup.csv',
        'no_files': 64,
    }

    main(LOAD_CONFIG)
