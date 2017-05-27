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

print('Loading pickled data')

share_data = pd.read_pickle('data/ml-data.pkl.gz', compression='gzip')
gc.collect()

# Set target column
target_column = returns['8']

categorical_columns = ['symbol', 'quoteMonth', '4WeekBollingerPrediction', '4WeekBollingerType',
                       '12WeekBollingerPrediction', '12WeekBollingerType']


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
        'max_bin': 1250000,
        #'boosting_type': 'dart',
        #'feature_fraction': 0.9,
        #'bagging_fraction': 0.8,
        #'bagging_freq': 5,
        'silent': True
    }

    # feature_name and categorical_feature
    gbm = lgb.train(params,
                    train_set,
                    valid_sets=eval_set,  # eval training data
                    feval=mle_eval,
                    learning_rates=lambda iter: 0.1 * (0.99 ** iter),
                    num_boost_round=2000,
                    early_stopping_rounds=5)

    del df_train
    del train_y
    gc.collect()

    file_name = 'lgbm-model-%d' % r
    lgb.save_model(file_name, num_iteration=gbm.best_iteration)

    # Output model settings
    fit_time = time.time()
    print('Elapsed time: %d' % (fit_time - start))

    predictions = gbm.predict(df_valid, num_iteration=gbm.best_iteration)

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
