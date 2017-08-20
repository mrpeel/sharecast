import numpy as np
import pandas as pd
from memory_profiler import profile
import gc
from xgboost_general_symbol_ensemble_sharecast import *





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
def load_data(base_path, increments):
    loading_data = pd.DataFrame()
    for increment in increments:
        path = base_path % increment
        frame = pd.read_csv(path, compression='gzip', parse_dates=['quoteDate'], infer_datetime_format=True, low_memory=False)
        loading_data = loading_data.append(frame, ignore_index=True)
        del frame
        print('Loaded:', path)

    return loading_data



share_data = load_data(base_path='data/companyQuotes-20170725-%03d.csv.gz', increments=range(1, 54))
gc.collect()

save_data = setup_data_columns(share_data)
gc.collect()


print('Saving data')

# msk = np.random.rand(len(share_data)) < 0.10
# sample2 = share_data[msk].copy()
#
# sample2.to_pickle('data/ml-july-data.pkl.gz', compression='gzip')

save_data.to_pickle('data/ml-july-data.pkl.gz', compression='gzip')
# share_data.to_hdf('data/ml-july-data.hdf5', 'sharecast', mode='w', complevel=9, complib='bzip2')
