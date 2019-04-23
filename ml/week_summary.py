from scipy import stats
import pandas as pd
from processing_constants import ALL_CONTINUOUS_COLUMNS, HIGH_NAN_COLUMNS
from processing_constants import BOLLINGER_PREDICTION_COLUMNS, BOLLINGER_VALUE_COLUMNS
ID_COLUMNS = ['symbol', 'GICSSector', 'GICSIndustryGroup', 'GICSIndustry']
DATE_COLS = ['exDividendDate']

median_cols = []
median_cols.extend(ALL_CONTINUOUS_COLUMNS)
median_cols.extend(DATE_COLS)
# Remove val which isn't in df
median_cols.remove('quoteDate_TIMESTAMP')

sum_cols = ['sumVolume']

mode_cols = []
mode_cols.extend(BOLLINGER_PREDICTION_COLUMNS)
mode_cols.extend(BOLLINGER_VALUE_COLUMNS)
mode_cols.extend(ID_COLUMNS)


def return_week_summary(df):
    """
        Creates a weekly summary of the daily results for a symbol - assumes a datetime index
    """

    # remove high nan cols
    df.drop(HIGH_NAN_COLUMNS, axis=1, inplace=True, errors='ignore')

    # remove separate date col
    df.drop(['quoteDate'], axis=1, inplace=True, errors='ignore')

    # col for sum of volume
    df['sumVolume'] = df['volume']

    if len(median_cols):
        median_resample = df[median_cols].resample('1W').median()
    else:
        median_resample = pd.DataFrame()

    if len(sum_cols):
        sum_resample = df[sum_cols].resample('1W').sum()
    else:
        sum_resample = pd.DataFrame()

    if len(mode_cols):
        mode_resample = df[mode_cols].resample(
            '1W').apply(lambda x: stats.mode(x)[0][0])
    else:
        mode_resample = pd.DataFrame()

    return pd.concat([median_resample, sum_resample, mode_resample], axis=1)
