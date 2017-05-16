import xgboost as xgb
import numpy as np
import pandas as pd
# from xgboost import plot_importance
# from matplotlib import pyplot
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import OneHotEncoder
# from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import time
from dateutil.parser import parse




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

increments = range(1, 77)

share_data = pd.DataFrame()
for increment in increments:
    path = 'data/companyQuotes-20170514-%03d.csv.gz' % increment
    frame = pd.read_csv(path, compression='gzip', parse_dates=['quoteDate'], infer_datetime_format=True)
    share_data = share_data.append(frame, ignore_index=True)
    print('Loaded:', path)

print(share_data.head(5))
print(len(share_data))

# Set target column
target_column = returns['8']

# Remove rows missing the target column
share_data = share_data.dropna(subset=[target_column], how='all')

# Shift values to range of >= 1
shift_val = 0


def get_shift_value(data_frame):
    # if the minimum value is < 1, shift all the values to make them >= 1
    min_val = min(data_frame.values)
    if min_val < 1:
        return (min_val * -1) + 1
    else:
        return 0


print(share_data[target_column].head(5))

shift_val = get_shift_value(share_data[target_column])
print(shift_val)

share_data[target_column] = share_data[target_column].add(shift_val)

print(share_data[target_column].head(5))


# Set log values
print(share_data[target_column].head(5))

print('Min:', min(share_data[target_column].values), ', Max:', max(share_data[target_column].values))

share_data[target_column] = np.log(share_data[target_column])

print(share_data[target_column].head(5))

print('Min:', min(share_data[target_column].values), ', Max:', max(share_data[target_column].values))

# Create y_data
y_data = share_data[target_column].values


# Filter down data to the X columns being used
share_data = share_data[data_columns]


print(share_data.dtypes)

print('Min:',min(y_data),', Max:', max(y_data))

# Convert quote dates data to year and month
share_data['quoteDate'] = pd.to_datetime(share_data['quoteDate'])
share_data['quoteYear'], share_data['quoteMonth'],  share_data['quoteDay'] = share_data['quoteDate'].dt.year, share_data['quoteDate'].dt.month, share_data['quoteDate'].dt.day

# Convert dividend dates data to year and month
share_data['exDividendDate'] = pd.to_datetime(share_data['exDividendDate'])
share_data['exDividendYear'], share_data['exDividendMonth'],  share_data['exDividendDay'] = share_data['exDividendDate'].dt.year, share_data['exDividendDate'].dt.month, share_data['exDividendDate'].dt.day


# Remove quote dates column
del share_data['quoteDate']
del share_data['exDividendDate']


# Convert categorical variables to boolean fields
#  4WeekBollingerPrediction
#  4WeekBollingerType
#  12WeekBollingerPrediction
#  12WeekBollingerType

share_data = pd.get_dummies(data=share_data, columns=['symbol', '4WeekBollingerPrediction', '4WeekBollingerType', '12WeekBollingerPrediction', '12WeekBollingerType'])

# Fill nan values with placeholder and check for null values
share_data = share_data.fillna(-99999)
print(pd.isnull(share_data).any())


# Check data types
print(share_data.dtypes)

# Copy over X_data columns
X_data = share_data.values


# Check how many fields in X_data
print(X_data.shape)


# Split into train and test data
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.7, test_size=0.3)

print('Training for', target_column)

# Fit model with training set
start = time.time()
model = xgb.XGBRegressor(nthread=-1, n_estimators=10000)

print(model)

kfold = KFold(n_splits=3, shuffle=True)

errs = []
r2s = []

for train_index, test_index in kfold.split(X_data):
    actuals = y_data[test_index]
    eval_set = [(X_data[test_index], actuals)]
    model.fit(X_data[train_index], y_data[train_index], early_stopping_rounds=30, eval_metric="mae", eval_set=eval_set, verbose=True)
    predictions = model.predict(X_data[test_index])

    # Output model settings
    fit_time = time.time()
    print('Elapsed time: %d' % (fit_time - start))
    err = mean_absolute_error(actuals, predictions)
    errs.append(err)
    r2 = r2_score(actuals, predictions)
    r2s.append(r2)
    print("Fold mean absolute error (log of y): %s" % err)
    print("Fold r2: %s" % r2)

print('-----')
print("Average (3 folds) mean absolute error (log of y): %s" % np.mean(errs))
print("Average (3 folds) r2: %s" % np.mean(r2s))
