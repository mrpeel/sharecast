
# coding: utf-8

# ## Load and prep columns

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('qt5agg')
import math
from auto_ml import Predictor
import math

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# Calculate the value for a data frame column to shift all the values to make them >= 1
#    allows values to have natural log conversion applied
def get_shift_value(data_frame_column):
    min_val = min(data_frame_column.values)
    if min_val < 1:
        return (min_val * -1) + 1
    else:
        return 0


# Define columns
data_columns = ['symbol', 'quoteDate', 'adjustedPrice', 'volume', 'previousClose', 'change', 'changeInPercent',
                '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
                'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'Price200DayAverage',
                'Price52WeekPercChange', '1WeekVolatility', '2WeekVolatility', '4WeekVolatility', '8WeekVolatility',
                '12WeekVolatility', '26WeekVolatility','52WeekVolatility','4WeekBollingerPrediction', '4WeekBollingerType',
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

# Load data
raw_data = pd.read_csv('data/companyQuotes-20170417-001.csv')
raw_data.head(5)


# Set target column
target_column = returns['8']

# Remove rows missing the target column
filtered_data = raw_data.dropna(subset=[target_column], how='all')

# Determine the
shift_val = get_shift_value(filtered_data[target_column])

# Make all values >= 1
filtered_data[target_column] = filtered_data[target_column].add(shift_val)


all_columns = data_columns[:]

all_columns.insert(0, target_column)

print(all_columns)

# Columns to use
filtered_data = filtered_data[all_columns]


print(filtered_data.dtypes)


# ## Run auto-ml

# Split data frame into 75 / 25 train test
msk = np.random.rand(len(filtered_data)) < 0.75
df_train_base = filtered_data[msk]
df_test = filtered_data[~msk]

# Re-split the training data into a deep learning set and a regressor set
msk = np.random.rand(len(df_train_base)) < 0.66
df_train = df_train_base[msk]
dl_train = df_train_base[~msk]


column_descriptions = {
    'Future8WeekReturn': 'output'
    , 'symbol': 'categorical'
    , 'quoteDate': 'date'
    , '4WeekBollingerPrediction': 'categorical'
    , '4WeekBollingerType': 'categorical'
    , '12WeekBollingerPrediction': 'categorical'
    , '12WeekBollingerType': 'categorical'
    , 'exDividendDate': 'date'
}


ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_train, feature_learning=True, take_log_of_y=True, fl_data=dl_train,
                    model_names=['XGBRegressor'])

ml_predictor.score(df_test, df_test.Future8WeekReturn, verbose=3)
