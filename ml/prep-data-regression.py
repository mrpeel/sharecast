import numpy as np
import pandas as pd
from memory_profiler import profile
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



print('Pickling data')

share_data.to_pickle('data/ml-data.pkl.gz', compression='gzip')
