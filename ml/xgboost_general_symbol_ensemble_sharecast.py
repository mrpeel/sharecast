
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, shutil

import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error




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
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
    """
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', np.mean(np.absolute(safe_log(actual_y) - safe_log(prediction_y)))

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


def train_xgb(share_data):
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

    all_train_x = pd.DataFrame()
    all_train_y = pd.DataFrame()
    all_test_x = pd.DataFrame()
    all_test_y = pd.DataFrame()
    all_test_actuals = pd.DataFrame()

    symbols_train_x = {}
    symbols_train_y = {}
    symbols_test_x = {}
    symbols_test_y = {}
    symbols_test_actuals = {}

    # prep data for fitting into both model types
    for symbol in symbols:
        # Set up map of symbol name to number
        symbol_map[symbol] = symbol_num

        # Update string to integer
        share_data.loc[share_data.symbol == symbol, 'symbol'] = symbol_num

        # Take copy of model data
        model_data = share_data.loc[share_data['symbol'] == symbol_num]

        msk = np.random.rand(len(model_data)) < 0.75

        symbols_test_x = {}
        symbols_test_y = {}
        symbols_test_actuals = {}

        symbols_train_y[symbol] = model_data[msk][LABEL_COLUMN + '_scaled']
        symbols_train_x[symbol] = model_data[msk].drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1)

        symbols_test_actuals[symbol] = model_data[~msk][LABEL_COLUMN]
        symbols_test_y[symbol] = model_data[~msk][LABEL_COLUMN + '_scaled']
        symbols_test_x[symbol] = model_data[~msk].drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1)

        all_train_y = all_train_y.append(model_data[msk][LABEL_COLUMN + '_scaled'])
        all_train_x = all_train_x.append(model_data[msk].drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

        all_test_actuals = all_test_actuals.append(model_data[~msk][LABEL_COLUMN])
        all_test_y = all_test_y.append(model_data[~msk][LABEL_COLUMN + '_scaled'])
        all_test_x = all_test_x.append(model_data[~msk].drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

    print(symbol_map)

    #Train general model
    # Create model
    model = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=110, base_score=0.35, colsample_bylevel=0.8,
                             colsample_bytree=0.8, gamma=0, learning_rate=0.01, max_delta_step=0,
                             min_child_weight=0)

    eval_set = [(all_test_x, all_test_y)]
    model.fit(all_train_x, all_train_y, early_stopping_rounds=25, eval_metric='mae', eval_set=eval_set, verbose=True)

    gc.collect()

    predictions = model.predict(all_test_x)
    inverse_scaled_predictions = safe_exp(predictions)

    err = mle(all_test_actuals, inverse_scaled_predictions)
    mae = mean_absolute_error(all_test_actuals, inverse_scaled_predictions)

    print('General model xgboost results')
    print("Mean log of error: %s" % err)
    print("Mean absolute error: %s" % mae)

    # Prepare results holders
    results = pd.DataFrame()
    #results['actuals'] = actuals
    #results['general_predictions'] = general_predictions
    #results['symbol_predictions'] = symbol_predictions

    # Create and execute models
    ## Run the predictions across using each symbol model and the genral model
    for symbol, symbol_num in symbol_map:
        # Create model
        symbol_model = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=110, base_score=0.35,
                                        colsample_bylevel=0.8, colsample_bytree = 0.8, gamma = 0, learning_rate = 0.01,
                                        max_delta_step = 0, min_child_weight = 0)

        eval_set = [(test_x, test_y)]
        symbol_model.fit(train_x, train_y, early_stopping_rounds=25, eval_metric='mae', eval_set=eval_set, verbose=True)

        gc.collect()

        predictions = symbol_model.predict(test_x)
        inverse_scaled_predictions = safe_exp(predictions)

        err = mle(actuals, inverse_scaled_predictions)
        mae = mean_absolute_error(actuals, inverse_scaled_predictions)

        print('xgboost results for', symbol)
        print("Mean log of error: %s" % err)
        print("Mean absolute error: %s" % mae)

        symbol_models[symbol] = symbol_model

        symbol_num += 1









if __name__ == "__main__":
    # Retrieve and run combined prep on data
    share_data  = prep_data()
    gc.collect()

    # Train model
    results = train_xgb(share_data)


    # results['xgb_predictions'] = xgb_predictions
    #
    # # Generate final and combined results
    # tf_err = mle(actuals, tf_predictions)
    # tf_mae = mean_absolute_error(actuals, tf_predictions)
    #
    #
    # xgb_err = mle(actuals, xgb_predictions)
    # xgb_mae = mean_absolute_error(actuals, xgb_predictions)
    #
    # combined_err = mle(actuals, ((tf_predictions + xgb_predictions) / 2))
    # combined_mae = mean_absolute_error(actuals, ((tf_predictions + xgb_predictions) / 2))
    #
    # # Print results
    # print('Overall results')
    # print('-------------------')
    # print('Mean log of error')
    # print('  tf:', tf_err, ' xgb:', xgb_err, 'combined: ', combined_err)
    #
    # print('Mean absolute error')
    # print('  tf:', tf_mae, ' xgb:', xgb_mae, 'combined: ', combined_mae)
    #
    #
    # result_ranges = [-50, -25, -10, -5, 0, 2, 5, 10, 20, 50, 100, 1001]
    # lower_range = -100
    #
    # for upper_range in result_ranges:
    #     range_results = results.loc[(results['actuals'] >= lower_range) &
    #                                 (results['actuals'] < upper_range)]
    #     # Generate final and combined results
    #     range_actuals = range_results['actuals'].values
    #     range_tf_predictions = range_results['tf_predictions'].values
    #     range_xgb_predictions = range_results['xgb_predictions'].values
    #
    #     tf_err = mle(range_actuals, range_tf_predictions)
    #     tf_mae = mean_absolute_error(range_actuals, range_tf_predictions)
    #
    #
    #     xgb_err = mle(range_actuals, range_xgb_predictions)
    #     xgb_mae = mean_absolute_error(range_actuals, range_xgb_predictions)
    #
    #     combined_err = mle(range_actuals, ((range_tf_predictions + range_xgb_predictions) / 2))
    #     combined_mae = mean_absolute_error(range_actuals, ((range_tf_predictions + range_xgb_predictions) / 2))
    #
    #     # Print results
    #     print('Results:', lower_range, 'to', upper_range)
    #     print('-------------------')
    #     print('Mean log of error')
    #     print('  tf:', tf_err, ' xgb:', xgb_err, 'combined: ', combined_err)
    #
    #     print('Mean absolute error')
    #     print('  tf:', tf_mae, ' xgb:', xgb_mae, 'combined: ', combined_mae)
    #
    #
    #     lower_range = upper_range
