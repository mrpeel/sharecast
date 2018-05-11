# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import os, shutil

import pandas as pd
import numpy as np
import tensorflow as tf
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


def clear_model_dir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

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

def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.
  symbol = tf.contrib.layers.sparse_column_with_hash_bucket(
      "symbol", hash_bucket_size=5000)
  _4WeekBollingerPrediction = tf.contrib.layers.sparse_column_with_hash_bucket(
      "4WeekBollingerPrediction", hash_bucket_size=10)
  _4WeekBollingerType = tf.contrib.layers.sparse_column_with_hash_bucket(
      "4WeekBollingerType", hash_bucket_size=10)
  _12WeekBollingerPrediction = tf.contrib.layers.sparse_column_with_hash_bucket(
      "12WeekBollingerPrediction", hash_bucket_size=10)
  _12WeekBollingerType = tf.contrib.layers.sparse_column_with_hash_bucket(
      "12WeekBollingerType", hash_bucket_size=10)


  # Continuous base columns.
  quoteMonth = tf.contrib.layers.real_valued_column("quoteMonth")
  quoteYear = tf.contrib.layers.real_valued_column("quoteYear")
  adjustedPrice = tf.contrib.layers.real_valued_column("adjustedPrice")
  volume = tf.contrib.layers.real_valued_column("volume")
  previousClose = tf.contrib.layers.real_valued_column("previousClose")
  change = tf.contrib.layers.real_valued_column("change")
  changeInPercent = tf.contrib.layers.real_valued_column("changeInPercent")
  _52WeekHigh = tf.contrib.layers.real_valued_column("52WeekHigh")
  _52WeekLow = tf.contrib.layers.real_valued_column("52WeekLow")
  changeFrom52WeekHigh = tf.contrib.layers.real_valued_column("changeFrom52WeekHigh")
  changeFrom52WeekLow = tf.contrib.layers.real_valued_column("changeFrom52WeekLow")
  percebtChangeFrom52WeekHigh = tf.contrib.layers.real_valued_column("percebtChangeFrom52WeekHigh")
  percentChangeFrom52WeekLow = tf.contrib.layers.real_valued_column("percentChangeFrom52WeekLow")
  Price200DayAverage = tf.contrib.layers.real_valued_column("Price200DayAverage")
  Price52WeekPercChange = tf.contrib.layers.real_valued_column("Price52WeekPercChange")
  _1WeekVolatility = tf.contrib.layers.real_valued_column("1WeekVolatility")
  _2WeekVolatility = tf.contrib.layers.real_valued_column("2WeekVolatility")
  _4WeekVolatility = tf.contrib.layers.real_valued_column("4WeekVolatility")
  _8WeekVolatility = tf.contrib.layers.real_valued_column("8WeekVolatility")
  _12WeekVolatility = tf.contrib.layers.real_valued_column("12WeekVolatility")
  _26WeekVolatility = tf.contrib.layers.real_valued_column("26WeekVolatility")
  _52WeekVolatility = tf.contrib.layers.real_valued_column("52WeekVolatility")
  allordpreviousclose = tf.contrib.layers.real_valued_column("allordpreviousclose")
  allordchange = tf.contrib.layers.real_valued_column("allordchange")
  allorddayshigh = tf.contrib.layers.real_valued_column("allorddayshigh")
  allorddayslow = tf.contrib.layers.real_valued_column("allorddayslow")
  allordpercebtChangeFrom52WeekHigh = tf.contrib.layers.real_valued_column("allordpercebtChangeFrom52WeekHigh")
  allordpercentChangeFrom52WeekLow = tf.contrib.layers.real_valued_column("allordpercentChangeFrom52WeekLow")
  asxpreviousclose = tf.contrib.layers.real_valued_column("asxpreviousclose")
  asxchange = tf.contrib.layers.real_valued_column("asxchange")
  asxdayshigh = tf.contrib.layers.real_valued_column("asxdayshigh")
  asxdayslow = tf.contrib.layers.real_valued_column("asxdayslow")
  asxpercebtChangeFrom52WeekHigh = tf.contrib.layers.real_valued_column("asxpercebtChangeFrom52WeekHigh")
  asxpercentChangeFrom52WeekLow = tf.contrib.layers.real_valued_column("asxpercentChangeFrom52WeekLow")
  exDividendRelative = tf.contrib.layers.real_valued_column("exDividendRelative")
  exDividendPayout = tf.contrib.layers.real_valued_column("exDividendPayout")
  _640106_A3597525W = tf.contrib.layers.real_valued_column("640106_A3597525W")
  AINTCOV = tf.contrib.layers.real_valued_column("AINTCOV")
  AverageVolume = tf.contrib.layers.real_valued_column("AverageVolume")
  BookValuePerShareYear = tf.contrib.layers.real_valued_column("BookValuePerShareYear")
  CashPerShareYear = tf.contrib.layers.real_valued_column("CashPerShareYear")
  DPSRecentYear = tf.contrib.layers.real_valued_column("DPSRecentYear")
  EBITDMargin = tf.contrib.layers.real_valued_column("EBITDMargin")
  EPS = tf.contrib.layers.real_valued_column("EPS")
  EPSGrowthRate10Years = tf.contrib.layers.real_valued_column("EPSGrowthRate10Years")
  EPSGrowthRate5Years = tf.contrib.layers.real_valued_column("EPSGrowthRate5Years")
  FIRMMCRT = tf.contrib.layers.real_valued_column("FIRMMCRT")
  FXRUSD = tf.contrib.layers.real_valued_column("FXRUSD")
  Float = tf.contrib.layers.real_valued_column("Float")
  GRCPAIAD = tf.contrib.layers.real_valued_column("GRCPAIAD")
  GRCPAISAD = tf.contrib.layers.real_valued_column("GRCPAISAD")
  GRCPBCAD = tf.contrib.layers.real_valued_column("GRCPBCAD")
  GRCPBCSAD = tf.contrib.layers.real_valued_column("GRCPBCSAD")
  GRCPBMAD = tf.contrib.layers.real_valued_column("GRCPBMAD")
  GRCPNRAD = tf.contrib.layers.real_valued_column("GRCPNRAD")
  GRCPRCAD = tf.contrib.layers.real_valued_column("GRCPRCAD")
  H01_GGDPCVGDP = tf.contrib.layers.real_valued_column("H01_GGDPCVGDP")
  H01_GGDPCVGDPFY = tf.contrib.layers.real_valued_column("H01_GGDPCVGDPFY")
  H05_GLFSEPTPOP = tf.contrib.layers.real_valued_column("H05_GLFSEPTPOP")
  IAD = tf.contrib.layers.real_valued_column("IAD")
  LTDebtToEquityQuarter = tf.contrib.layers.real_valued_column("LTDebtToEquityQuarter")
  LTDebtToEquityYear = tf.contrib.layers.real_valued_column("LTDebtToEquityYear")
  MarketCap = tf.contrib.layers.real_valued_column("MarketCap")
  NetIncomeGrowthRate5Years = tf.contrib.layers.real_valued_column("NetIncomeGrowthRate5Years")
  NetProfitMarginPercent = tf.contrib.layers.real_valued_column("NetProfitMarginPercent")
  OperatingMargin = tf.contrib.layers.real_valued_column("OperatingMargin")
  PE = tf.contrib.layers.real_valued_column("PE")
  PriceToBook = tf.contrib.layers.real_valued_column("PriceToBook")
  ReturnOnAssets5Years = tf.contrib.layers.real_valued_column("ReturnOnAssets5Years")
  ReturnOnAssetsTTM = tf.contrib.layers.real_valued_column("ReturnOnAssetsTTM")
  ReturnOnAssetsYear = tf.contrib.layers.real_valued_column("ReturnOnAssetsYear")
  ReturnOnEquity5Years = tf.contrib.layers.real_valued_column("ReturnOnEquity5Years")
  ReturnOnEquityTTM = tf.contrib.layers.real_valued_column("ReturnOnEquityTTM")
  ReturnOnEquityYear = tf.contrib.layers.real_valued_column("ReturnOnEquityYear")
  RevenueGrowthRate10Years = tf.contrib.layers.real_valued_column("RevenueGrowthRate10Years")
  RevenueGrowthRate5Years = tf.contrib.layers.real_valued_column("RevenueGrowthRate5Years")
  TotalDebtToAssetsQuarter = tf.contrib.layers.real_valued_column("TotalDebtToAssetsQuarter")
  TotalDebtToAssetsYear = tf.contrib.layers.real_valued_column("TotalDebtToAssetsYear")
  TotalDebtToEquityQuarter = tf.contrib.layers.real_valued_column("TotalDebtToEquityQuarter")
  TotalDebtToEquityYear = tf.contrib.layers.real_valued_column("TotalDebtToEquityYear")
  bookValue = tf.contrib.layers.real_valued_column("bookValue")
  earningsPerShare = tf.contrib.layers.real_valued_column("earningsPerShare")
  ebitda = tf.contrib.layers.real_valued_column("ebitda")
  epsEstimateCurrentYear = tf.contrib.layers.real_valued_column("epsEstimateCurrentYear")
  marketCapitalization = tf.contrib.layers.real_valued_column("marketCapitalization")
  peRatio = tf.contrib.layers.real_valued_column("peRatio")
  pegRatio = tf.contrib.layers.real_valued_column("pegRatio")
  pricePerBook = tf.contrib.layers.real_valued_column("pricePerBook")
  pricePerEpsEstimateCurrentYear = tf.contrib.layers.real_valued_column("pricePerEpsEstimateCurrentYear")
  pricePerEpsEstimateNextYear = tf.contrib.layers.real_valued_column("pricePerEpsEstimateNextYear")
  pricePerSales = tf.contrib.layers.real_valued_column("pricePerSales")

  # Wide columns and deep columns.
  wide_columns = [symbol, _4WeekBollingerPrediction , _4WeekBollingerType, _12WeekBollingerPrediction,
                  _12WeekBollingerType,
                  tf.contrib.layers.crossed_column([symbol, _4WeekBollingerPrediction],
                                                   hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([symbol, _12WeekBollingerPrediction],
                                                   hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([symbol, _4WeekBollingerPrediction, _12WeekBollingerPrediction],
                                                   hash_bucket_size=int(1e8)),
                  tf.contrib.layers.crossed_column([symbol, _4WeekBollingerType],
                                                   hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([symbol, _12WeekBollingerType],
                                                   hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([symbol, _4WeekBollingerType, _12WeekBollingerType],
                                                   hash_bucket_size=int(1e8))]

  deep_columns = [
      tf.contrib.layers.embedding_column(symbol, dimension=8),
      tf.contrib.layers.embedding_column(_4WeekBollingerPrediction, dimension=8),
      tf.contrib.layers.embedding_column(_4WeekBollingerType, dimension=8),
      tf.contrib.layers.embedding_column(_12WeekBollingerPrediction, dimension=8),
      tf.contrib.layers.embedding_column(_12WeekBollingerType, dimension=8),
      adjustedPrice,
      quoteMonth,
      quoteYear,
      volume,
      previousClose,
      change,
      changeInPercent,
      _52WeekHigh,
      _52WeekLow,
      changeFrom52WeekHigh,
      changeFrom52WeekLow,
      percebtChangeFrom52WeekHigh,
      percentChangeFrom52WeekLow,
      Price200DayAverage,
      Price52WeekPercChange,
      _1WeekVolatility,
      _2WeekVolatility,
      _4WeekVolatility,
      _8WeekVolatility,
      _12WeekVolatility,
      _26WeekVolatility,
      _52WeekVolatility,
      allordpreviousclose,
      allordchange,
      allorddayshigh,
      allorddayslow,
      allordpercebtChangeFrom52WeekHigh,
      allordpercentChangeFrom52WeekLow,
      asxpreviousclose ,
      asxchange,
      asxdayshigh,
      asxdayslow,
      asxpercebtChangeFrom52WeekHigh,
      asxpercentChangeFrom52WeekLow,
      exDividendRelative,
      exDividendPayout,
      _640106_A3597525W,
      AINTCOV,
      AverageVolume,
      BookValuePerShareYear,
      CashPerShareYear,
      DPSRecentYear,
      EBITDMargin,
      EPS,
      EPSGrowthRate10Years,
      EPSGrowthRate5Years,
      FIRMMCRT,
      FXRUSD,
      Float,
      GRCPAIAD,
      GRCPAISAD,
      GRCPBCAD,
      GRCPBCSAD,
      GRCPBMAD,
      GRCPNRAD,
      GRCPRCAD,
      H01_GGDPCVGDP,
      H01_GGDPCVGDPFY,
      H05_GLFSEPTPOP,
      IAD,
      LTDebtToEquityQuarter,
      LTDebtToEquityYear,
      MarketCap,
      NetIncomeGrowthRate5Years,
      NetProfitMarginPercent,
      OperatingMargin,
      PE,
      PriceToBook,
      ReturnOnAssets5Years,
      ReturnOnAssetsTTM,
      ReturnOnAssetsYear,
      ReturnOnEquity5Years,
      ReturnOnEquityTTM,
      ReturnOnEquityYear,
      RevenueGrowthRate10Years,
      RevenueGrowthRate5Years,
      TotalDebtToAssetsQuarter,
      TotalDebtToAssetsYear,
      TotalDebtToEquityQuarter,
      TotalDebtToEquityYear,
      bookValue,
      earningsPerShare,
      ebitda,
      epsEstimateCurrentYear,
      marketCapitalization,
      peRatio,
      pegRatio,
      pricePerBook,
      pricePerEpsEstimateCurrentYear,
      pricePerEpsEstimateNextYear,
      pricePerSales]


  m = tf.contrib.learn.DNNLinearCombinedRegressor(model_dir=model_dir,
                                                  linear_feature_columns=wide_columns,
                                                  linear_optimizer=tf.train.AdamOptimizer(
                                                      learning_rate=0.01,
                                                      beta1=0.9,
                                                      beta2=0.999,
                                                  ),
                                                  dnn_feature_columns=deep_columns,
                                                  dnn_hidden_units=[175, 90, 175, 90],
                                                  dnn_optimizer=tf.train.AdamOptimizer(
                                                      learning_rate=0.0125,
                                                      beta1=0.9,
                                                      beta2=0.999,
                                                  ),
                                                  dnn_activation_fn=tf.nn.relu6,
                                                  dnn_dropout=0.05,
                                                  fix_global_step_increment_bug=True,
                                                  config=tf.contrib.learn.RunConfig(save_checkpoints_secs=90))
  return m


def load_and_mask_data():
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

    # Create mask for splitting data 75 / 25
    msk = np.random.rand(len(df)) < 0.75

    return df, msk

def prep_tf_data():
    df, msk = load_and_mask_data()

    scaler = StandardScaler()
    df[CONTINUOUS_COLUMNS] = scaler.fit_transform(df[CONTINUOUS_COLUMNS].as_matrix())
    return df, msk

def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  #label = tf.constant(df[LABEL_COLUMN].values, shape=[df[LABEL_COLUMN].size, 1])
  label = tf.constant(df[LABEL_COLUMN + '_scaled'].values.astype(np.float32), shape=[df[LABEL_COLUMN + '_scaled'].size, 1])
  # Returns the feature columns and the label.
  return feature_cols, label

def train_wide_and_deep(share_data, msk, max_train_steps):
  """Train and evaluate the model."""

  # Use mask to split data
  df_train = share_data[msk].copy()
  df_test = share_data[~msk].copy()


  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  tf.logging.set_verbosity(tf.logging.INFO)

  model_dir = 'data/model'
  print("model directory = %s" % model_dir)

  # Clear model directpry
  clear_model_dir(model_dir)

  validation_metrics = {
      "mean_abs_error": tf.contrib.metrics.streaming_mean_absolute_error,
      "pearson": tf.contrib.metrics.streaming_pearson_correlation
  }


  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      input_fn=lambda: input_fn(df_test),
      eval_steps=1,
      every_n_steps=100,
      metrics=validation_metrics,
      early_stopping_metric="mean_abs_error",
      early_stopping_metric_minimize=True,
      early_stopping_rounds=1000)

  m = build_estimator(model_dir)
  print(m.get_params(deep=True))
  m.fit(input_fn=lambda: input_fn(df_train), steps=max_train_steps, monitors=[validation_monitor])
  # evaluate using tensorflow evaluation
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
        print("%s: %s" % (key, results[key]))

  # evaluate using predictions
  predictions = m.predict(input_fn=lambda: input_fn(df_test))
  np_predictions = np.fromiter(predictions, np.float)

  inverse_scaled_predictions = safe_exp(np_predictions)

  err = mle(df_test[LABEL_COLUMN], inverse_scaled_predictions)
  mae = mean_absolute_error(df_test[LABEL_COLUMN], inverse_scaled_predictions)

  print('tf results')
  print("Mean log of error: %s" % err)
  print("Mean absolute error: %s" % mae)

  # return predicted values
  return  inverse_scaled_predictions


def train_xgb(share_data, msk):
    # Use pandas dummy columns for categorical columns
    share_data = pd.get_dummies(data=share_data, columns=['4WeekBollingerPrediction',
                                                          '4WeekBollingerType',
                                                          '12WeekBollingerPrediction',
                                                          '12WeekBollingerType'])

    # Convert symbol to integer
    share_data['symbol'] = pd.factorize(share_data['symbol'])[0]

    #Create model
    model = xgb.XGBRegressor(nthread=-1, n_estimators=5000, max_depth=110, base_score=0.35, colsample_bylevel=0.8,
                             colsample_bytree=0.8, gamma=0, learning_rate=0.01, max_delta_step=0,
                             min_child_weight=0)



    # Set y values to log of y, and drop original label and log of y label for x values
    train_y = share_data[msk][LABEL_COLUMN + '_scaled'].values
    df_train_x = share_data[msk].drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1)
    train_x = df_train_x.values

    actuals = share_data[~msk][LABEL_COLUMN].values
    test_y = share_data[~msk][LABEL_COLUMN + '_scaled'].values
    df_test_x = share_data[~msk].drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1)
    test_x = df_test_x.values

    # Check columns being used
    print('train_x: ', list(df_train_x.columns.values))
    print('test_x: ', list(df_test_x.columns.values))


    eval_set = [(test_x, test_y)]
    model.fit(train_x, train_y, early_stopping_rounds=25, eval_metric=mle_eval, eval_set=eval_set, verbose=True)

    gc.collect()

    predictions = model.predict(test_x)


    inverse_scaled_predictions = safe_exp(predictions)

    err = mle(actuals, inverse_scaled_predictions)
    mae = mean_absolute_error(actuals, inverse_scaled_predictions)

    print('xgboost results')
    print("Mean log of error: %s" % err)
    print("Mean absolute error: %s" % mae)

    return inverse_scaled_predictions


if __name__ == "__main__":
    # Retrieve and run combined prep on data
    share_data, msk = prep_tf_data()
    gc.collect()

    # Set actual values
    actuals = share_data[~msk][LABEL_COLUMN]

    # Train deep learning model
    tf_predictions = train_wide_and_deep(share_data, msk, 50000)
    xgb_predictions = train_xgb(share_data, msk)

    results = pd.DataFrame()
    results['actuals'] = actuals
    results['tf_predictions'] = tf_predictions
    results['xgb_predictions'] = xgb_predictions

    # Generate final and combined results
    tf_err = mle(actuals, tf_predictions)
    tf_mae = mean_absolute_error(actuals, tf_predictions)


    xgb_err = mle(actuals, xgb_predictions)
    xgb_mae = mean_absolute_error(actuals, xgb_predictions)

    combined_err = mle(actuals, ((tf_predictions + xgb_predictions) / 2))
    combined_mae = mean_absolute_error(actuals, ((tf_predictions + xgb_predictions) / 2))

    # Print results
    print('Overall results')
    print('-------------------')
    print('Mean log of error')
    print('  tf:', tf_err, ' xgb:', xgb_err, 'combined: ', combined_err)

    print('Mean absolute error')
    print('  tf:', tf_mae, ' xgb:', xgb_mae, 'combined: ', combined_mae)


    result_ranges = [-50, -25, -10, -5, 0, 2, 5, 10, 20, 50, 100, 1001]
    lower_range = -100

    for upper_range in result_ranges:
        range_results = results.loc[(results['actuals'] >= lower_range) &
                                    (results['actuals'] < upper_range)]
        # Generate final and combined results
        range_actuals = range_results['actuals'].values
        range_tf_predictions = range_results['tf_predictions'].values
        range_xgb_predictions = range_results['xgb_predictions'].values

        tf_err = mle(range_actuals, range_tf_predictions)
        tf_mae = mean_absolute_error(range_actuals, range_tf_predictions)


        xgb_err = mle(range_actuals, range_xgb_predictions)
        xgb_mae = mean_absolute_error(range_actuals, range_xgb_predictions)

        combined_err = mle(range_actuals, ((range_tf_predictions + range_xgb_predictions) / 2))
        combined_mae = mean_absolute_error(range_actuals, ((range_tf_predictions + range_xgb_predictions) / 2))

        # Print results
        print('Results:', lower_range, 'to', upper_range)
        print('-------------------')
        print('Mean log of error')
        print('  tf:', tf_err, ' xgb:', xgb_err, 'combined: ', combined_err)

        print('Mean absolute error')
        print('  tf:', tf_mae, ' xgb:', xgb_mae, 'combined: ', combined_mae)


        lower_range = upper_range
