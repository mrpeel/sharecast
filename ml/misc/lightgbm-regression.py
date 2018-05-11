import numpy as np
import pandas as pd
import lightgbm as lgb
from memory_profiler import profile
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import time
import gc





pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

# Define columns
data_columns = ['symbol', 'quoteDate', 'adjustedPrice', 'volume', 'previousClose', 'change', 'changeInPercent',
                '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
                'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'Price200DayAverage',
                'Price52WeekPercChange', '1WeekVolatility', '2WeekVolatility', '4WeekVolatility', '8WeekVolatility',
                '12WeekVolatility', '26WeekVolatility', '52WeekVolatility', '4WeekBollingerPrediction',
                '4WeekBollingerType',
                '12WeekBollingerPrediction', '12WeekBollingerType', 'allordpreviousclose', 'allordchange',
                'allorddayshigh', 'allorddayslow', 'allordpercebtChangeFrom52WeekHigh',
                'allordpercentChangeFrom52WeekLow', 'asxpreviousclose', 'asxchange', 'asxdayshigh',
                'asxdayslow', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow', 'exDividendDate',
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

selected_columns = ['symbol', 'adjustedPrice', 'volume', 'previousClose', 'change',
                    '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
                    'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'Price200DayAverage',
                    'Price52WeekPercChange', '1WeekVolatility', '2WeekVolatility', '4WeekVolatility', '8WeekVolatility',
                    '12WeekVolatility', '26WeekVolatility', '52WeekVolatility', '4WeekBollingerPrediction',
                    '4WeekBollingerType',
                    '12WeekBollingerPrediction', '12WeekBollingerType', 'allordchange',
                    'allorddayshigh', 'allorddayslow', 'allordpercebtChangeFrom52WeekHigh',
                    'allordpercentChangeFrom52WeekLow', 'asxchange', 'asxdayshigh',
                    'asxdayslow', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow', 'AverageVolume',
                    'EBITDMargin', 'EPSGrowthRate10Years', 'EPSGrowthRate5Years', 'FIRMMCRT', 'FXRUSD', 'Float',
                    'GRCPAIAD', 'GRCPBCAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD', 'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP',
                    'IAD', 'LTDebtToEquityQuarter', 'LTDebtToEquityYear', 'MarketCap',
                    'NetIncomeGrowthRate5Years', 'NetProfitMarginPercent',
                    'PriceToBook', 'ReturnOnAssets5Years', 'ReturnOnAssetsTTM', 'ReturnOnAssetsYear',
                    'ReturnOnEquity5Years', 'ReturnOnEquityTTM', 'RevenueGrowthRate10Years',
                    'RevenueGrowthRate5Years', 'TotalDebtToAssetsQuarter', 'TotalDebtToAssetsYear',
                    'TotalDebtToEquityQuarter', 'bookValue', 'earningsPerShare',
                    'ebitda', 'epsEstimateCurrentYear', 'marketCapitalization', 'peRatio', 'pegRatio', 'pricePerBook',
                    'pricePerEpsEstimateCurrentYear', 'pricePerEpsEstimateNextYear', 'pricePerSales']

returns = {
    '1': 'Future1WeekReturn',
    '2': 'Future2WeekReturn',
    '4': 'Future4WeekReturn',
    '8': 'Future8WeekReturn',
    '12': 'Future12WeekReturn',
    '26': 'Future26WeekReturn',
    '52': 'Future52WeekReturn',
    '1ra': 'Future1WeekRiskAdjustedReturn',
    '2ra': 'Future2WeekRiskAdjustedReturn',
    '4ra': 'Future4WeekRiskAdjustedReturn',
    '8ra': 'Future8WeekRiskAdjustedReturn',
    '12ra': 'Future12WeekRiskAdjustedReturn',
    '26ra': 'Future26WeekRiskAdjustedReturn',
    '52ra': 'Future52WeekRiskAdjustedReturn'
}

# Load data
# raw_data = pd.read_csv('data/companyQuotes-20170417-001.csv')
@profile
def load_data(base_path, increments):
    loading_data = pd.DataFrame()
    for increment in increments:
        path = base_path % increment
        frame = pd.read_csv(path, compression='gzip', parse_dates=['quoteDate'], infer_datetime_format=True, low_memory=False)
        loading_data = loading_data.append(frame, ignore_index=True)
        del frame
        print('Loaded:', path)

    return loading_data

def get_shift_value(df):
    # if the minimum value is < 1, shift all the values to make them >= 0
    min_val = min(df.values)
    if min_val < 1:
        return (min_val * -1)
    else:
        return 0

@profile
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


def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals[neg_mask] *= -1
    return_vals = np.log(return_vals + 1)
    return_vals[neg_mask] *= -1
    return return_vals


def mle(actual_y, prediction_y):
    """
    Compute the Root Mean Squared Log Error

    Args:
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
    """

    # return np.sqrt(np.square(np.log(prediction_y + 1) - np.log(actual_y + 1)).mean())
    # return np.median(np.absolute(np.log(prediction_y + 1) - np.log(actual_y + 1)))
    return np.mean(np.absolute(safe_log(prediction_y) - safe_log(actual_y)))


def mle_eval(y, y0):
    y0 = y0.get_label()
    assert len(y) == len(y0)
    # return 'error', np.sqrt(np.mean(np.square(np.log(y + 1) - np.log(y0 + 1))))
    return 'error', np.mean(np.absolute(safe_log(y) - safe_log(y0))), False

share_data = load_data(base_path='data/companyQuotes-20170514-%03d.csv.gz', increments=range(1, 77))
gc.collect()

print(len(share_data))
print('Post load:')
print(share_data.info(max_cols=0, memory_usage=True))

# Set target column
target_column = returns['8']

# Remove rows missing the target column
share_data = share_data.dropna(subset=[target_column], how='all')
print('Post drop NA:')
print(share_data.info(max_cols=0, memory_usage=True))


# Shift values to range of >= 1
shift_val = 0


print(share_data[target_column].head(5))

#shift_val = get_shift_value(share_data[target_column])
#print('Shift value:', shift_val)

#share_data[target_column] = share_data[target_column].add(shift_val)

# Clip to -99 to 1000 range
share_data[target_column] = share_data[target_column].clip(-99, 1000)
print('Post clip:')
print(share_data.info(max_cols=0, memory_usage=True))



# Set log values
print(share_data[target_column].head(5))

print('Min:', min(share_data[target_column].values), ', Max:', max(share_data[target_column].values))


# Filter down data to the X columns being used
all_columns = data_columns[:]
all_columns.insert(0, target_column)

share_data = drop_unused_columns(share_data, all_columns)
print('Post drop unused columns:')
print(share_data.info(max_cols=0, memory_usage=True))



# Convert quote dates data to year and month
share_data['quoteDate'] = pd.to_datetime(share_data['quoteDate'])
share_data['exDividendDate'] = pd.to_datetime(share_data['exDividendDate'])
print('Post re-set date types:')
print(share_data.info(max_cols=0, memory_usage=True))


# Reset divident date as a number
share_data['exDividendRelative'] = \
    share_data['exDividendDate'] - \
    share_data['quoteDate']

# convert string difference value to integer
share_data['exDividendRelative'] = share_data['exDividendRelative'].apply(lambda x: np.nan if pd.isnull(x) else x.days)
print('Post dividend relative:')
print(share_data.info(max_cols=0, memory_usage=True))


share_data['quoteYear'], share_data['quoteMonth'],  = \
    share_data['quoteDate'].dt.year, \
    share_data['quoteDate'].dt.month.astype('int8')


print('Post quote date:')
print(share_data.info(max_cols=0, memory_usage=True))


# Remove quote dates column
share_data.drop(['quoteDate', 'exDividendDate'], axis=1, inplace=True)

print('Post drop dates:')
print(share_data.info(max_cols=0, memory_usage=True))

categorical_columns = ['symbol', 'quoteMonth', '4WeekBollingerPrediction', '4WeekBollingerType',
                       '12WeekBollingerPrediction', '12WeekBollingerType']

# Factorize each categorical column
for cat_col in categorical_columns:
    share_data[cat_col] = pd.factorize(share_data[cat_col])[0]

print('Training for', target_column)

maes = []
r2s = []
errs = []

for r in range(0, 3):
    # Set-up lgb data
    msk = np.random.rand(len(share_data)) < 0.75
    df_train = share_data[msk].copy()
    train_y = df_train[target_column].values
    df_train.drop([target_column], axis=1, inplace=True)

    df_valid = share_data[~msk].copy()
    valid_y = df_valid[target_column].values
    df_valid.drop([target_column], axis=1, inplace=True)

    train_set = lgb.Dataset(df_train, label=train_y, categorical_feature=categorical_columns)
    eval_set = lgb.Dataset(df_valid, reference=train_set, label=valid_y, categorical_feature=categorical_columns)

    # Fit model with training set
    start = time.time()

    params = {
        'objective': 'regression',
        'num_leaves': 8192,
        'max_bin': 25500,
        'boosting_type': 'dart',
        'silent': True
    }

    # feature_name and categorical_feature
    gbm = lgb.train(params,
                    train_set,
                    valid_sets=eval_set,  # eval training data
                    feval=mle_eval,
                    learning_rates=lambda iter: 0.15 * (0.99 ** iter),
                    num_boost_round=2000,
                    early_stopping_rounds=10)

    del df_train
    del train_y
    gc.collect()

    # Output model settings
    fit_time = time.time()
    print('Elapsed time: %d' % (fit_time - start))

    predictions = gbm.predict(df_valid)

    err = mle(valid_y, predictions)
    mae = mean_absolute_error(valid_y, predictions)
    print(err)
    print(mae)
    errs.append(err)
    maes.append(mae)
    r2 = r2_score(valid_y, predictions)
    r2s.append(r2)
    print("Fold mean mle (log of y): %s" % err)
    print("Fold mean absolute error: %s" % mae)
    print("Fold r2: %s" % r2)

    del df_valid
    del valid_y
    gc.collect()


print('-----')
print("Average (3 folds) mle (log of y): %s" % np.mean(errs))
print("Average (3 folds) mae: %s" % np.mean(maes))
print("Average (3 folds) r2: %s" % np.mean(r2s))
