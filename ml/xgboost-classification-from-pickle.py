import xgboost as xgb
import numpy as np
import pandas as pd
from memory_profiler import profile
from sklearn.metrics import accuracy_score, confusion_matrix
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

LABEL_COLUMN = 'Future8WeekReturn'


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



print('Loading pickled data')

df = pd.read_pickle('data/ml-sample-data-2.pkl.gz', compression='gzip')
gc.collect()

# Set-up columns to keep
data_columns.append(LABEL_COLUMN)

drop_unused_columns(df, data_columns)


# Convert quote dates data to year and month
df['quoteDate'] = pd.to_datetime(df['quoteDate'])
df['exDividendDate'] = pd.to_datetime(df['exDividendDate'])

# Reset divident date as a number
df['exDividendRelative'] = \
    df['exDividendDate'] - \
    df['quoteDate']

# convert string difference value to integer
df['exDividendRelative'] = df['exDividendRelative'].apply(
    lambda x: np.nan if pd.isnull(x) else x.days)

df['quoteYear'], df['quoteMonth'], = \
    df['quoteDate'].dt.year, \
    df['quoteDate'].dt.month.astype('int8')

# Remove dates columns
df.drop(['quoteDate', 'exDividendDate'], axis=1, inplace=True)

df = df.dropna(subset=[LABEL_COLUMN], how='all')

# Clip to -99 to 1000 range
df[LABEL_COLUMN] = df[LABEL_COLUMN].clip(-99, 1000)

bins = [-100, -50, -25, -10, -5, 0, 2, 5, 10, 20, 50, 100, 1001]
group_names = ['Wipeout', 'Lost 25 to 50 percent', 'Lost 10 to 25 percent', 'Lost 5 to 10 percent',
               'Lost under 5 percent', 'Steady', 'Gained 2 to 5 percent', 'Gained 5 to 10 percent',
               'Gained 10 to 20 percent', 'Gained 20 to 50 percent', 'Gained 50 to 100 percent',
               'More than doubled']

df[LABEL_COLUMN + '_categories'] = pd.cut(df[LABEL_COLUMN], bins, labels=group_names)
df[LABEL_COLUMN + '_cat_class'] = pd.factorize(df[LABEL_COLUMN + '_categories'])[0]

# Drop base label cols
df.drop([LABEL_COLUMN, LABEL_COLUMN + '_categories'], axis=1, inplace=True)


# Fill N/A vals with dummy number
df.fillna(-99999, inplace=True)

# Convert symbol to integer
df['symbol'] = pd.factorize(df['symbol'])[0]

df = pd.get_dummies(data=df, columns=['4WeekBollingerPrediction', '4WeekBollingerType', '12WeekBollingerPrediction',
                                      '12WeekBollingerType'])


print('Training for', LABEL_COLUMN)


# Create mask for splitting data 75 / 25
msk = np.random.rand(len(df)) < 0.75



model = xgb.XGBClassifier(nthread=-1, objective='multi:softmax',
                          n_estimators=1000, max_depth=110, base_score = 0.35, colsample_bylevel = 0.8,
                          colsample_bytree = 0.8, gamma = 0, learning_rate = 0.015, max_delta_step = 0,
                          min_child_weight = 0)

# Set y values to log of y, and drop original label and log of y label for x values
train_y = df[msk][LABEL_COLUMN + '_cat_class'].values
train_x = df[msk].drop([LABEL_COLUMN + '_cat_class'], axis=1).values

test_y = df[~msk][LABEL_COLUMN + '_cat_class'].values
test_x = df[~msk].drop([LABEL_COLUMN + '_cat_class'], axis=1).values

start = time.time()

eval_set = [(test_x, test_y)]
model.fit(train_x, train_y, early_stopping_rounds=10, eval_metric='mlogloss', eval_set=eval_set, verbose=True)

# Output model settings
fit_time = time.time()
print('Fit elapsed time: %d' % (fit_time - start))

gc.collect()

predictions = model.predict(test_x)
predition_time = time.time()
print('Prediction elapsed time: %d' % (predition_time - fit_time))
print(model)

# evaluate predictions
accuracy = accuracy_score(test_y, predictions)
print('Accuracy:', accuracy)

print(confusion_matrix(test_y, predictions))

