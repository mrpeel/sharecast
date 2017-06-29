
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pickle
import gzip
import sys
from sklearn.externals import joblib
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



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



# returns = {
#     '1': 'Future1WeekReturn',
#     '2': 'Future2WeekReturn',
#     '4': 'Future4WeekReturn',
#     '8': 'Future8WeekReturn',
#     '12': 'Future12WeekReturn',
#     '26': 'Future26WeekReturn',
#     '52': 'Future52WeekReturn',
#     '1ra': 'Future1WeekRiskAdjustedReturn',
#     '2ra': 'Future2WeekRiskAdjustedReturn',
#     '4ra': 'Future4WeekRiskAdjustedReturn',
#     '8ra': 'Future8WeekRiskAdjustedReturn',
#     '12ra': 'Future12WeekRiskAdjustedReturn',
#     '26ra': 'Future26WeekRiskAdjustedReturn',
#     '52ra': 'Future52WeekRiskAdjustedReturn'
# }


LABEL_COLUMN = "Future8WeekReturn"
CATEGORICAL_COLUMNS = ['symbol', '4WeekBollingerPrediction', '4WeekBollingerType',
                       '12WeekBollingerPrediction', '12WeekBollingerType']
CONTINUOUS_COLUMNS = ['adjustedPrice', 'quoteMonth', 'quoteYear', 'volume', 'previousClose', 'change',
                'changeInPercent','52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
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


def save(object, filename):
        """Saves a compressed object to disk
        """
        with gzip.open(filename, 'wb') as f:
            f.write(pickle.dumps(object, pickle.HIGHEST_PROTOCOL))

        # max_bytes = 2 ** 31 - 1
        #
        # ## write
        # bytes_out = pickle.dumps(object)
        # with open(filename, 'wb') as f_out:
        #     for idx in range(0, sys.getsizeof(object), max_bytes):
        #         f_out.write(bytes_out[idx:idx + max_bytes])


def load(filename):
        """Loads a compressed object from disk
        """
        # file = gzip.GzipFile(filename, 'rb')
        # buffer = ""
        # while True:
        #         data = file.read()
        #         if data == "":
        #                 break
        #         buffer += data
        # object = pickle.loads(buffer)
        # file.close()
        object = pickle.dump(object, open(filename, "rb"))
        return object

def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log(np.absolute(return_vals) + 1)
    return_vals[neg_mask] *= -1
    return return_vals

def safe_exp(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.exp(np.absolute(return_vals)) - 1
    return_vals[neg_mask] *= -1
    return return_vals

def y_scaler(input_array):
    transformed_array = safe_log(input_array)
    scaler = MaxAbsScaler()
    #transformed_array = scaler.fit_transform(transformed_array)
    return transformed_array, scaler

def y_inverse_scaler(prediction_array, scaler):
    transformed_array = prediction_array #scaler.inverse_transform(prediction_array)
    transformed_array = safe_exp(transformed_array)
    return transformed_array


def mle(actual_y, prediction_y):
    """
    Compute the Root Mean  Log Error

    Args:
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
    """

    return np.mean(np.absolute(safe_log(prediction_y) - safe_log(actual_y)))

def mle_eval(actual_y, eval_y):
    """
    Used during xgboost training

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', np.mean(np.absolute(safe_log(actual_y) - safe_log(prediction_y)))

def safe_mape(actual_y, prediction_y):
    """
    Calculate mean absolute percentage error

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    denominators = actual_y.copy()
    set_ones = (denominators >= 0) & (denominators <= 1)
    set_neg_ones = (denominators >= -1) & (denominators < 0)
    denominators[set_ones] = 1
    denominators[set_neg_ones] = -1

    return np.mean(np.absolute((prediction_y - actual_y) / denominators * 100))



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


def load_data():
    """Load pickled data and run combined prep """
    # Return dataframe and mask to split data
    df = pd.read_pickle('data/ml-sample-data.pkl.gz', compression='gzip')
    gc.collect()

    # Remove columns not referenced in either algorithm
    columns_to_keep = [LABEL_COLUMN, 'quoteDate', 'exDividendDate']
    columns_to_keep.extend(CONTINUOUS_COLUMNS)
    columns_to_keep.extend(CATEGORICAL_COLUMNS)
    df = drop_unused_columns(df, columns_to_keep)

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

    # Add scaled value for y - using log of y
    df[LABEL_COLUMN + '_scaled'] = safe_log(df[LABEL_COLUMN].values)

    # Fill N/A vals with dummy number
    df.fillna(-99999, inplace=True)

    return df

def prep_data():
    df = load_data()

    scaler = StandardScaler()
    df[CONTINUOUS_COLUMNS] = scaler.fit_transform(df[CONTINUOUS_COLUMNS].as_matrix())
    return df

def divide_data(share_data):
    # Use pandas dummy columns for categorical columns
    share_data = pd.get_dummies(data=share_data, columns=['4WeekBollingerPrediction',
                                                          '4WeekBollingerType',
                                                          '12WeekBollingerPrediction',
                                                          '12WeekBollingerType'])

    symbol_models = {}
    symbols = share_data['symbol'].unique().tolist()
    symbol_map = {}
    symbol_num = 0

    print('No of symbols:', len(symbols))

    df_all_train_x = pd.DataFrame()
    df_all_train_y = pd.DataFrame()
    df_all_test_x = pd.DataFrame()
    df_all_test_y = pd.DataFrame()
    df_all_test_actuals = pd.DataFrame()

    symbols_train_x = {}
    symbols_train_y = {}
    symbols_test_x = {}
    symbols_test_y = {}
    symbols_test_actuals = {}

    # prep data for fitting into both model types
    for symbol in symbols:
        gc.collect()

        print('Symbol:', symbol, 'num:', symbol_num)

        # Update string to integer
        share_data.loc[share_data.symbol == symbol, 'symbol'] = symbol_num

        # Take copy of model data and re-set the pandas indexes
        model_data = share_data.loc[share_data['symbol'] == symbol_num]


        msk = np.random.rand(len(model_data)) < 0.75

        # Prep dataframes and reset index for appending
        df_train = model_data[msk]
        df_test = model_data[~msk]
        df_train.reset_index()
        df_test.reset_index()

        # MAke sure a monimum number of rows are present in sample for symbol
        if (len(df_train) > 150 & len(df_test) > 50):
            symbols_train_y[symbol] = df_train[LABEL_COLUMN + '_scaled'].values
            symbols_train_x[symbol] = df_train.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1).values

            symbols_test_actuals[symbol] = df_test[LABEL_COLUMN].values
            symbols_test_y[symbol] = df_test[LABEL_COLUMN + '_scaled'].values
            symbols_test_x[symbol] = df_test.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1).values

            df_all_train_y = pd.concat([df_all_train_y, df_train[LABEL_COLUMN + '_scaled']])
            df_all_train_x = df_all_train_x.append(df_train.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

            df_all_test_actuals = pd.concat([df_all_test_actuals, df_test[LABEL_COLUMN]])
            df_all_test_y = pd.concat([df_all_test_y, df_test[LABEL_COLUMN + '_scaled']])
            df_all_test_x = df_all_test_x.append(df_test.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

            # Set up map of symbol name to number
            symbol_map[symbol] = symbol_num

        symbol_num += 1

    print(symbol_map)

    # Clean-up the initial data variable
    return symbol_map, symbols_train_y, symbols_train_x, symbols_test_actuals, symbols_test_y, symbols_test_x, \
           df_all_train_y, df_all_train_x, df_all_test_actuals, df_all_test_y, df_all_test_x


def train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x):
    #Train general model
    # Create model
    model = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=110, base_score=0.35, colsample_bylevel=0.8,
                             colsample_bytree=0.8, gamma=0, learning_rate=0.01, max_delta_step=0,
                             min_child_weight=0)

    all_train_y = df_all_train_y.as_matrix()
    all_train_x = df_all_train_x.as_matrix()
    all_test_actuals = df_all_test_actuals.as_matrix()
    all_test_y = df_all_test_y.as_matrix()
    all_test_x = df_all_test_x.as_matrix()

    eval_set = [(all_test_x, all_test_y)]
    model.fit(all_train_x, all_train_y, early_stopping_rounds=50, eval_metric='mae', eval_set=eval_set,
              verbose=False)

    gc.collect()

    predictions = model.predict(all_test_x)
    inverse_scaled_predictions = safe_exp(predictions)

    err = mean_absolute_error(all_test_y, predictions)
    mae = mean_absolute_error(all_test_actuals, inverse_scaled_predictions)
    mape = safe_mape(all_test_actuals, inverse_scaled_predictions)
    r2 = r2_score(all_test_actuals, inverse_scaled_predictions)

    print('General model xgboost results')
    print("Mean log of error: %s" % err)
    print("Mean absolute error: %s" % mae)
    print("Mean absolute percentage error: %s" % mape)
    print("r2: %s" % r2)

    return model

def train_symbol_models(symbol_map, symbols_train_y, symbols_train_x, symbols_test_actuals, symbols_test_y,
                        symbols_test_x, gen_model):

    # Create and execute models
    symbol_models = {}
    all_results = pd.DataFrame()
    results_output = pd.DataFrame()

    ## Run the predictions across using each symbol model and the genral model
    for symbol in symbol_map:
        # Create model
        symbol_model = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=110, base_score=0.35,
                                        colsample_bylevel=0.8, colsample_bytree = 0.8, gamma = 0, learning_rate = 0.01,
                                        max_delta_step = 0, min_child_weight = 0)

        train_y = symbols_train_y[symbol]
        train_x = symbols_train_x[symbol]
        test_actuals = symbols_test_actuals[symbol]
        test_y = symbols_test_y[symbol]
        test_x = symbols_test_x[symbol]

        eval_set = [(test_x, test_y)]
        symbol_model.fit(train_x, train_y, early_stopping_rounds=50, eval_metric='mae', eval_set=eval_set,
                         verbose=False)

        gc.collect()

        # Add model to models dictionary
        symbol_models[symbol] = symbol_model

        # Run predictions and prepare results
        predictions = symbol_model.predict(test_x)
        inverse_scaled_predictions = safe_exp(predictions)

        gen_predictions = gen_model.predict(test_x)
        gen_inverse_scaled_predictions = safe_exp(gen_predictions)

        #Evaluet results
        symbol_err = mean_absolute_error(test_y, predictions)
        symbol_mae = mean_absolute_error(test_actuals, inverse_scaled_predictions)
        symbol_mape = safe_mape(test_actuals, inverse_scaled_predictions)
        symbol_r2 = r2_score(test_actuals, inverse_scaled_predictions)

        gen_err = mean_absolute_error(test_y, gen_predictions)
        gen_mae = mean_absolute_error(test_actuals, gen_inverse_scaled_predictions)
        gen_mape = safe_mape(test_actuals, gen_inverse_scaled_predictions)
        gen_r2 = r2_score(test_actuals, gen_inverse_scaled_predictions)


        fifty_err = mean_absolute_error(test_y, ((gen_predictions + predictions) / 2))
        fifty_mae = mean_absolute_error(test_actuals, ((gen_inverse_scaled_predictions + inverse_scaled_predictions) / 2))
        fifty_mape = safe_mape(test_actuals, ((gen_inverse_scaled_predictions + inverse_scaled_predictions) / 2))
        fifty_r2 = r2_score(test_actuals, ((gen_inverse_scaled_predictions + inverse_scaled_predictions) / 2))


        sixty_err = mean_absolute_error(test_y, ((gen_predictions + (predictions*2)) / 3))
        sixty_mae = mean_absolute_error(test_actuals, ((gen_inverse_scaled_predictions + (inverse_scaled_predictions*2)) / 3))
        sixty_mape = safe_mape(test_actuals, ((gen_inverse_scaled_predictions + (inverse_scaled_predictions*2)) / 3))
        sixty_r2 = r2_score(test_actuals, ((gen_inverse_scaled_predictions + (inverse_scaled_predictions*2)) / 3))


        all_results = all_results.append(pd.DataFrame.from_dict({'actuals': test_actuals,
                                                                'gen_predictions': gen_inverse_scaled_predictions,
                                                                'symbol_predictions': inverse_scaled_predictions
                                                                }))

        results_output = results_output.append(pd.DataFrame.from_dict({'symbol': [symbol],
                                                                       'general_mle': [gen_err],
                                                                       'symbol_mle': [symbol_err],
                                                                       'fifty_fifty_mle': [fifty_err],
                                                                       'sixty_thirty_mle': [sixty_err],
                                                                       'general_mae': [gen_mae],
                                                                       'symbol_mae': [symbol_mae],
                                                                       'fifty_fifty_mae': [fifty_mae],
                                                                       'sixty_thirty_mae': [sixty_mae],
                                                                       'general_mape': [gen_mape],
                                                                       'symbol_mape': [symbol_mape],
                                                                       'fifty_fifty_mape': [fifty_mape],
                                                                       'sixty_thirty_mape': [sixty_mape],
                                                                       'general_r2': [gen_r2],
                                                                       'symbol_r2': [symbol_r2],
                                                                       'fifty_fifty_r2': [fifty_r2],
                                                                       'sixty_thirty_r2': [sixty_r2]
                                                                       }))


                                                                       # Print results
        print('Results for', symbol)
        print('-------------------')
        print('Mean log of error')
        print('     gen:', gen_err)
        print('  symbol:', symbol_err)
        print('   50/50:', fifty_err)
        print('   66/32:', sixty_err)
        print('Mean absolute error')
        print('     gen:', gen_mae)
        print('  symbol:', symbol_mae)
        print('   50/50:', fifty_mae)
        print('   66/32:', sixty_mae)
        print("Mean absolute percentage error")
        print('     gen:', gen_mape)
        print('  symbol:', symbol_mape)
        print('   50/50:', fifty_mape)
        print('   66/32:', sixty_mape)
        print('r2')
        print('     gen:', gen_r2)
        print('  symbol:', symbol_r2)
        print('   50/50:', fifty_r2)
        print('   66/32:', sixty_r2)


    return symbol_models, all_results, results_output


if __name__ == "__main__":
    # Prepare run_str
    run_str = datetime.datetime.now().strftime('%Y%m%d%H%M')

    # Retrieve and run combined prep on data
    share_data  = prep_data()
    gc.collect()

    # Divide data into symbol and general data for training an testing
    symbol_map, symbols_train_y, symbols_train_x, symbols_test_actuals, symbols_test_y, symbols_test_x, df_all_train_y, \
    df_all_train_x, df_all_test_actuals, df_all_test_y, df_all_test_x = divide_data(share_data)

    del share_data
    gc.collect()

    # Train the general model
    gen_model = train_general_model(df_all_train_x, df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x)

    # Remove combined data, only required for general model
    del df_all_train_y, df_all_test_actuals, df_all_test_y, df_all_test_x
    gc.collect()

    symbol_models, all_results, results_output = train_symbol_models(symbol_map, symbols_train_y, symbols_train_x,
                                                                     symbols_test_actuals, symbols_test_y,
                                                                     symbols_test_x, gen_model)

    gc.collect()

    # Save results as csv
    results_output.to_csv('xgboost-models/results-' + run_str + '.csv', columns = ['symbol',
                                                                                   'general_mle',
                                                                                    'symbol_mle',
                                                                                    'fifty_fifty_mle',
                                                                                   'sixty_thirty_mle',
                                                                                   'general_mae',
                                                                                   'symbol_mae',
                                                                                   'fifty_fifty_mae',
                                                                                   'sixty_thirty_mae',
                                                                                   'general_mape',
                                                                                   'symbol_mape',
                                                                                   'fifty_fifty_mape',
                                                                                   'sixty_thirty_mape',
                                                                                   'general_r2',
                                                                                   'symbol_r2',
                                                                                   'fifty_fifty_r2',
                                                                                   'sixty_thirty_r2'])


    # Save models to file
    save(symbol_map, 'xgboost-models/symbol-map.dat.gz')
    save(gen_model, 'xgboost-models/general.dat.gz')
    save(symbol_models, 'xgboost-models/symbols.dat.gz')
    # joblib.dump(symbol_map, 'xgboost-models/symbol-map.dat.gz', compress=('gzip', 3))
    # joblib.dump(symbol_map, 'xgboost-models/general.xgb.gz', compress=('gzip', 3))
    # joblib.dump(symbol_map, 'xgboost-models/symbols.xgb.gz', compress=('gzip', 3))


    # Generate final and combined results
    gen_err = mean_absolute_error(all_results['actuals'].values, all_results['gen_predictions'].values)
    symbol_err = mean_absolute_error(all_results['actuals'].values, all_results['symbol_predictions'].values)

    gen_mape = safe_mape(all_results['actuals'].values, all_results['gen_predictions'].values)
    symbol_mape = safe_mape(all_results['actuals'].values, all_results['symbol_predictions'].values)

    gen_r2 = r2_score(all_results['actuals'].values, all_results['gen_predictions'].values)
    symbol_r2 = r2_score(all_results['actuals'].values, all_results['symbol_predictions'].values)

    # Print results
    print('Overall results')
    print('-------------------')
    print('Mean absolute error')
    print('     gen:', gen_err)
    print('  symbol:', symbol_err)

    print('Mean absolute percentage error')
    print('     gen:', gen_mape)
    print('  symbol:', symbol_mape)

    print('r2')
    print('     gen:', gen_r2)
    print('  symbol:', symbol_r2)

    overall_results_output = pd.DataFrame()


    result_ranges = [-50, -25, -10, -5, 0, 2, 5, 10, 20, 50, 100, 1001]
    lower_range = -100

    for upper_range in result_ranges:
        range_results = all_results.loc[(all_results['actuals'] >= lower_range) &
                                        (all_results['actuals'] < upper_range)]
        # Generate final and combined results
        range_actuals = range_results['actuals'].values
        range_gen_predictions = range_results['gen_predictions'].values
        range_symbol_predictions = range_results['symbol_predictions'].values

        gen_mae = mean_absolute_error(range_actuals, range_gen_predictions)
        symbol_mae = mean_absolute_error(range_actuals, range_symbol_predictions)
        gen_mape = safe_mape(range_actuals, range_gen_predictions)
        symbol_mape = safe_mape(range_actuals, range_symbol_predictions)
        gen_mlpa = safe_mlpa(range_actuals, range_gen_predictions)
        symbol_mlpa = safe_mlpa(range_actuals, range_symbol_predictions)


        # Print results
        print('Results:', lower_range, 'to', upper_range)
        print('-------------------')
        print('Mean absolute error')
        print('     gen:', gen_mae)
        print('  symbol:', symbol_mae)

        print('Mean absolute percentage error')
        print('     gen:', gen_mape)
        print('  symbol:', symbol_mape)



        overall_results_output = overall_results_output.append(pd.DataFrame.from_dict(
            {'lower_range': [lower_range],
             'upper_range': [upper_range],
             'gen_mae': [gen_mae],
             'symbol_mae': [symbol_mae],
             'gen_mape': [gen_mape],
             'symbol_mape': [symbol_mape]
             }))


        lower_range = upper_range

    # Output range results as csv
    overall_results_output.to_csv('xgboost-models/range-results-' + run_str+ '.csv', columns=['lower_range',
                                                                                              'upper_range',
                                                                                              'gen_mae',
                                                                                              'symbol_mae',
                                                                                              'gen_mape',
                                                                                              'symbol_mape'])