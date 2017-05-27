import numpy as np
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import gc


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
    return 'error', np.mean(np.absolute(safe_log(y) - safe_log(y0))), False

print('Loading pickled data')

share_data = pd.read_pickle('data/ml-data.pkl.gz', compression='gzip')
gc.collect()

# Set target column
target_column = returns['8']


share_data = pd.get_dummies(data=share_data, columns=['quoteMonth',
                                                      '4WeekBollingerPrediction',
                                                      '4WeekBollingerType',
                                                      '12WeekBollingerPrediction',
                                                      '12WeekBollingerType'])

share_data = share_data.fillna(-99999)

X_train, X_test, y_train, y_test = train_test_split(share_data.drop([target_column], axis=1).values,
                                                    share_data[target_column].values,
                                                    train_size=0.75, test_size=0.25)

regressor_config_dict = {
    'xgboost.XGBRegressor': {
        'nthread': [-1],
        'n_estimators': [250],
        'max_depth': [70, 90],
        'base_score': [i/100.0 for i in range(0, 101, 5)],
        'colsample_bylevel': [i/100.0 for i in range(0, 101, 5)],
        'colsample_bytree': [i/100.0 for i in range(0, 101, 5)],
        'gamma': [i for i in range(0, 11, 1)],
        'max_delta_step': [i/100.0 for i in range(0, 101, 5)],
        'min_child_weight': [i for i in range(0, 11, 1)],
        'reg_alpha': [i/100.0 for i in range(0, 101, 5)],
        'reg_lambda': [i/100.0 for i in range(0, 101, 5)],
        'scale_pos_weight': [i/100.0 for i in range(0, 101, 5)],
        'subsample': [i/100.0 for i in range(0, 101, 5)]
    }
}

tpot_model = TPOTRegressor(generations=5, population_size=10, verbosity=2, cv=3,
                      scoring=mle, config_dict=regressor_config_dict)
tpot_model.fit(X_train, y_train)
print(tpot_model.score(X_test, y_test))
tpot_model.export('tpot_xgboost_pipeline.py')