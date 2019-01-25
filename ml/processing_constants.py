COLUMNS = ['symbol', '4WeekBollingerPrediction', '4WeekBollingerType', '12WeekBollingerPrediction',
           '12WeekBollingerType', 'adjustedPrice', 'quoteMonth', 'volume', 'previousClose', 'change',
           'changeInPercent', '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
           'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'Price200DayAverage',
           'Price52WeekPercChange', '1WeekVolatility', '2WeekVolatility', '4WeekVolatility', '8WeekVolatility',
           '12WeekVolatility', '26WeekVolatility', '52WeekVolatility', 'allordpreviousclose', 'allordchange',
           'allorddayshigh', 'allorddayslow', 'allordpercebtChangeFrom52WeekHigh',
           'allordpercentChangeFrom52WeekLow', 'asxpreviousclose', 'asxchange', 'asxdayshigh',
           'asxdayslow', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow', 'exDividendRelative',
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

LABEL_COLUMN = 'future_eight_week_return'
RETURN_COLUMN = 'eight_week_total_return'

CATEGORICAL_COLUMNS = ['symbol_encoded', 'quoteDate_YEAR',
                       'quoteDate_MONTH', 'quoteDate_DAY', 'quoteDate_DAYOFWEEK']

CONTINUOUS_COLUMNS = ['lastTradePriceOnly', 'adjustedPrice', 'quoteDate_TIMESTAMP', 'volume', 'previousClose',
                      'change', 'changeInPercent',
                      '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
                      'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'allordpreviousclose',
                      'allordchange', 'allorddayshigh', 'allorddayslow', 'allordpercebtChangeFrom52WeekHigh',
                      'allordpercentChangeFrom52WeekLow', 'asxpreviousclose', 'asxchange', 'asxdayshigh',
                      'asxdayslow', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow',
                      'exDividendRelative', 'exDividendPayout', '640106_A3597525W', 'AINTCOV', 'Beta',
                      'BookValuePerShareYear', 'CashPerShareYear', 'DPSRecentYear', 'EPS', 'FIRMMCRT', 'FXRUSD',
                      'Float', 'GRCPAIAD', 'GRCPAISAD', 'GRCPBCAD', 'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD',
                      'H01_GGDPCVGDP', 'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP', 'MarketCap', 'OperatingMargin', 'PE',
                      'QuoteLast', 'ReturnOnEquityYear', 'TotalDebtToEquityYear', 'daysHigh', 'daysLow']


PAST_RESULTS_CONTINUOUS_COLUMNS = ['one_week_min', 'one_week_max', 'one_week_mean', 'one_week_median', 'one_week_std',
                                   'one_week_bollinger_upper', 'one_week_bollinger_lower',
                                   'one_week_comparison_adjustedPrice', 'one_week_price_change',
                                   'one_week_price_return',
                                   'one_week_dividend_value', 'one_week_dividend_return',
                                   'one_week_total_return', 'two_week_min', 'two_week_max',
                                   'two_week_mean', 'two_week_median', 'two_week_std',
                                   'two_week_bollinger_upper', 'two_week_bollinger_lower',
                                   'two_week_comparison_adjustedPrice',
                                   'two_week_price_change', 'two_week_price_return',
                                   'two_week_dividend_value', 'two_week_dividend_return',
                                   'two_week_total_return', 'four_week_min', 'four_week_max',
                                   'four_week_mean', 'four_week_median', 'four_week_std',
                                   'four_week_bollinger_upper', 'four_week_bollinger_lower',
                                   'four_week_comparison_adjustedPrice',
                                   'four_week_price_change', 'four_week_price_return',
                                   'four_week_dividend_value', 'four_week_dividend_return',
                                   'four_week_total_return', 'eight_week_min', 'eight_week_max',
                                   'eight_week_mean', 'eight_week_median', 'eight_week_std',
                                   'eight_week_bollinger_upper', 'eight_week_bollinger_lower',
                                   'eight_week_comparison_adjustedPrice',
                                   'eight_week_price_change',
                                   'eight_week_price_return', 'eight_week_dividend_value',
                                   'eight_week_dividend_return', 'eight_week_total_return',
                                   'twelve_week_min', 'twelve_week_max', 'twelve_week_mean',
                                   'twelve_week_median', 'twelve_week_std',
                                   'twelve_week_bollinger_upper', 'twelve_week_bollinger_lower',
                                   'twelve_week_comparison_adjustedPrice',
                                   'twelve_week_price_change',
                                   'twelve_week_price_return', 'twelve_week_dividend_value',
                                   'twelve_week_dividend_return', 'twelve_week_total_return',
                                   'twenty_six_week_min', 'twenty_six_week_max',
                                   'twenty_six_week_mean', 'twenty_six_week_median',
                                   'twenty_six_week_std', 'twenty_six_week_bollinger_upper',
                                   'twenty_six_week_bollinger_lower',
                                   'twenty_six_week_comparison_adjustedPrice',
                                   'twenty_six_week_price_change',
                                   'twenty_six_week_price_return', 'twenty_six_week_dividend_value',
                                   'twenty_six_week_dividend_return', 'twenty_six_week_total_return',
                                   'fifty_two_week_min', 'fifty_two_week_max', 'fifty_two_week_mean',
                                   'fifty_two_week_median', 'fifty_two_week_std',
                                   'fifty_two_week_bollinger_upper', 'fifty_two_week_bollinger_lower',
                                   'fifty_two_week_comparison_adjustedPrice',
                                   'fifty_two_week_price_change',
                                   'fifty_two_week_price_return', 'fifty_two_week_dividend_value',
                                   'fifty_two_week_dividend_return', 'fifty_two_week_total_return']

PAST_RESULTS_CATEGORICAL_COLUMNS = ['one_week_bollinger_type', 'one_week_bollinger_prediction',
                                    'two_week_bollinger_type', 'two_week_bollinger_prediction',
                                    'four_week_bollinger_type', 'four_week_bollinger_prediction',
                                    'eight_week_bollinger_type', 'eight_week_bollinger_prediction',
                                    'twelve_week_bollinger_type', 'twelve_week_bollinger_prediction',
                                    'twenty_six_week_bollinger_type', 'twenty_six_week_bollinger_prediction',
                                    'fifty_two_week_bollinger_type', 'fifty_two_week_bollinger_prediction']

RECURRENT_COLUMNS = ['asxpreviousclose_T11_20P', 'asxpreviousclose_T1P', 'asxpreviousclose_T2_5P',
                     'asxpreviousclose_T6_10P', 'asxpreviousclose_T11_20P', 'asxpreviousclose_T1P',
                     'asxpreviousclose_T2_5P', 'asxpreviousclose_T6_10P', 'allordpreviousclose_T11_20P',
                     'allordpreviousclose_T1P', 'allordpreviousclose_T2_5P', 'allordpreviousclose_T6_10P',
                     'adjustedPrice_T11_20P', 'adjustedPrice_T1P', 'adjustedPrice_T2_5P', 'adjustedPrice_T6_10P',
                     'FIRMMCRT_T11_20P', 'FIRMMCRT_T1P', 'FIRMMCRT_T2_5P', 'FIRMMCRT_T6_10P', 'FXRUSD_T11_20P',
                     'FXRUSD_T1P', 'FXRUSD_T2_5P', 'FXRUSD_T6_10P', 'GRCPAIAD_T11_20P', 'GRCPAIAD_T1P',
                     'GRCPAIAD_T2_5P', 'GRCPAIAD_T6_10P', 'GRCPAISAD_T1P', 'GRCPAISAD_T2_5P', 'GRCPAISAD_T6_10P',
                     'GRCPAISAD_T11_20P', 'GRCPBCAD_T1P', 'GRCPBCAD_T2_5P', 'GRCPBCAD_T6_10P', 'GRCPBCAD_T11_20P',
                     'GRCPBCSAD_T1P', 'GRCPBCSAD_T2_5P', 'GRCPBCSAD_T6_10P', 'GRCPBCSAD_T11_20P',
                     'GRCPBMAD_T1P', 'GRCPBMAD_T2_5P', 'GRCPBMAD_T6_10P', 'GRCPBMAD_T11_20P', 'GRCPNRAD_T1P',
                     'GRCPNRAD_T2_5P', 'GRCPNRAD_T6_10P', 'GRCPNRAD_T11_20P', 'GRCPRCAD_T1P', 'GRCPRCAD_T2_5P',
                     'GRCPRCAD_T6_10P', 'GRCPRCAD_T11_20P', 'H01_GGDPCVGDPFY_T1P', 'H01_GGDPCVGDPFY_T2_5P',
                     'H01_GGDPCVGDPFY_T6_10P', 'H01_GGDPCVGDPFY_T11_20P', 'H05_GLFSEPTPOP_T1P', 'H05_GLFSEPTPOP_T2_5P',
                     'H05_GLFSEPTPOP_T6_10P', 'H05_GLFSEPTPOP_T11_20P', 'future_eight_week_return_T1_10P',
                     'future_eight_week_return_T1_20P', 'future_eight_week_return_T1_40P']

HIGH_NAN_COLUMNS = ['Price200DayAverage', 'Price52WeekPercChange', 'AverageVolume', 'EBITDMargin',
                    'EPSGrowthRate10Years', 'EPSGrowthRate5Years', 'IAD', 'LTDebtToEquityQuarter',
                    'LTDebtToEquityYear', 'NetIncomeGrowthRate5Years', 'NetProfitMarginPercent',
                    'PriceToBook', 'ReturnOnAssets5Years', 'ReturnOnAssetsTTM', 'ReturnOnAssetsYear',
                    'ReturnOnEquity5Years', 'ReturnOnEquityTTM', 'RevenueGrowthRate10Years',
                    'RevenueGrowthRate5Years', 'TotalDebtToAssetsQuarter', 'TotalDebtToAssetsYear',
                    'TotalDebtToEquityQuarter', 'bookValue', 'earningsPerShare', 'ebitda',
                    'epsEstimateCurrentYear', 'marketCapitalization', 'peRatio', 'pegRatio', 'pricePerBook',
                    'pricePerEpsEstimateCurrentYear', 'pricePerEpsEstimateNextYear', 'pricePerSales']

PAST_RESULTS_DATE_REF_COLUMNS = ['one_week_comparison_date', 'two_week_comparison_date', 'four_week_comparison_date',
                                 'eight_week_comparison_date', 'twelve_week_comparison_date',
                                 'twenty_six_week_comparison_date', 'fifty_two_week_comparison_date']

WHOLE_MARKET_COLUMNS = ['quoteDate', 'allordpreviousclose', 'allordchange', 'allorddayshigh', 'allorddayslow',
                        'allordpercebtChangeFrom52WeekHigh', 'allordpercentChangeFrom52WeekLow', 'asxpreviousclose',
                        'asxchange', 'asxdayshigh', 'asxdayslow', 'asxpercebtChangeFrom52WeekHigh',
                        'asxpercentChangeFrom52WeekLow', '640106_A3597525W', 'FIRMMCRT', 'FXRUSD', 'GRCPAIAD',
                        'GRCPAISAD', 'GRCPBCAD', 'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD', 'H01_GGDPCVGDP',
                        'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP']

SYMBOL_COPY_COLUMNS = ['adjustedPrice', 'previousClose', 'change', 'changeInPercent', '52WeekHigh', '52WeekLow',
                       'changeFrom52WeekHigh', 'changeFrom52WeekLow', 'percebtChangeFrom52WeekHigh',
                       'percentChangeFrom52WeekLow', 'Price200DayAverage', 'Price52WeekPercChange', 'AverageVolume',
                       'BookValuePerShareYear', 'CashPerShareYear', 'DPSRecentYear', 'EBITDMargin', 'EPS',
                       'EPSGrowthRate10Years', 'EPSGrowthRate5Years', 'IAD', 'LTDebtToEquityQuarter',
                       'LTDebtToEquityYear', 'MarketCap', 'NetIncomeGrowthRate5Years', 'NetProfitMarginPercent',
                       'OperatingMargin', 'PE', 'PriceToBook', 'ReturnOnAssets5Years', 'ReturnOnAssetsTTM',
                       'ReturnOnAssetsYear', 'ReturnOnEquity5Years', 'ReturnOnEquityTTM', 'ReturnOnEquityYear',
                       'RevenueGrowthRate10Years', 'RevenueGrowthRate5Years', 'TotalDebtToAssetsQuarter',
                       'TotalDebtToAssetsYear', 'TotalDebtToEquityQuarter', 'TotalDebtToEquityYear', 'bookValue',
                       'earningsPerShare', 'ebitda', 'epsEstimateCurrentYear', 'marketCapitalization', 'peRatio',
                       'pegRatio', 'pricePerBook', 'pricePerEpsEstimateCurrentYear', 'pricePerEpsEstimateNextYear',
                       'pricePerSales']

ALL_CONTINUOUS_COLUMNS = []
ALL_CONTINUOUS_COLUMNS.extend(CONTINUOUS_COLUMNS)
ALL_CONTINUOUS_COLUMNS.extend(PAST_RESULTS_CONTINUOUS_COLUMNS)
ALL_CONTINUOUS_COLUMNS.extend(RECURRENT_COLUMNS)

ALL_CATEGORICAL_COLUMNS = []
ALL_CATEGORICAL_COLUMNS.extend(CATEGORICAL_COLUMNS)
ALL_CATEGORICAL_COLUMNS.extend(PAST_RESULTS_CATEGORICAL_COLUMNS)

COLUMNS_TO_REMOVE = []
COLUMNS_TO_REMOVE.extend(HIGH_NAN_COLUMNS)
COLUMNS_TO_REMOVE.extend(PAST_RESULTS_DATE_REF_COLUMNS)

COLUMN_TYPES = {
    'symbol': 'category',
    'one_week_bollinger_type': 'category',
    'one_week_bollinger_prediction': 'category',
    'two_week_bollinger_type': 'category',
    'two_week_bollinger_prediction': 'category',
    'four_week_bollinger_type': 'category',
    'four_week_bollinger_prediction': 'category',
    'eight_week_bollinger_type': 'category',
    'eight_week_bollinger_prediction': 'category',
    'twelve_week_bollinger_type': 'category',
    'twelve_week_bollinger_prediction': 'category',
    'twenty_six_week_bollinger_type': 'category',
    'twenty_six_week_bollinger_prediction': 'category',
    'fifty_two_week_bollinger_type': 'category',
    'fifty_two_week_bollinger_prediction': 'category',
}


XGB_SET_PATH = './models/xgb-sets/'
INDUSTRY_XGB_SET_PATH = './models/xgb-industry-models/'
