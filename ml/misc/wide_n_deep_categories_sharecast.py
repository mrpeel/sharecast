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

import argparse
import sys
import tempfile
import os, shutil

import pandas as pd
import numpy as np
import tensorflow as tf
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
CONTINUOUS_COLUMNS = ['adjustedPrice', 'quoteMonth', 'volume', 'previousClose', 'change', 'changeInPercent',
                '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow',
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


  m = tf.contrib.learn.DNNLinearCombinedClassifier(model_dir=model_dir,
                                                   n_classes=12,
                                                   linear_feature_columns=wide_columns,
                                                   linear_optimizer=tf.train.AdamOptimizer(
                                                      learning_rate=0.01,
                                                      beta1=0.9,
                                                      beta2=0.999,
                                                   ),
                                                   dnn_feature_columns=deep_columns,
                                                   dnn_hidden_units=[10], #, 90, 175, 90],
                                                   dnn_optimizer=tf.train.AdamOptimizer(
                                                      learning_rate=0.0125,
                                                      beta1=0.9,
                                                      beta2=0.999,
                                                   ),
                                                   fix_global_step_increment_bug=True)#,
                                                   #config=tf.contrib.learn.RunConfig(save_checkpoints_secs=90))
  return m



def load_and_prepdata():
    df = pd.read_pickle('data/ml-sample-data.pkl.gz', compression='gzip')
    gc.collect()

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

    # Fill N/A vals with dummy number
    df.fillna(-99999, inplace=True)

    # Clip to -99 to 1000 range
    df[LABEL_COLUMN] = df[LABEL_COLUMN].clip(-99, 1000)

    bins = [-100, -50, -25, -10, -5, 0, 2, 5, 10, 20, 50, 100, 1001]
    group_names = ['Wipeout', 'Lost 25 to 50 percent', 'Lost 10 to 25 percent', 'Lost 5 to 10 percent',
                   'Lost under 5 percent', 'Steady', 'Gained 2 to 5 percent','Gained 5 to 10 percent',
                   'Gained 10 to 20 percent','Gained 20 to 50 percent','Gained 50 to 100 percent',
                   'More than doubled']

    df[LABEL_COLUMN + '_categories'] = pd.cut(df[LABEL_COLUMN], bins, labels=group_names)

    df[LABEL_COLUMN + '_cat_class'] = pd.factorize(df[LABEL_COLUMN + '_categories'])[0]

    scaler = StandardScaler()
    df[CONTINUOUS_COLUMNS] = scaler.fit_transform(df[CONTINUOUS_COLUMNS].as_matrix())
    return df

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
  label = tf.constant(df[LABEL_COLUMN + '_cat_class'].values)


  # Returns the feature columns and the label.
  return feature_cols, label
def train_and_eval(train_steps, continue_training=False):
  """Train and evaluate the model."""
  share_data = load_and_prepdata()
  gc.collect()

  msk = np.random.rand(len(share_data)) < 0.75
  df_train = share_data[msk].copy()
  df_test = share_data[~msk].copy()


  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  tf.logging.set_verbosity(tf.logging.INFO)

  model_dir = 'data/model'
  print("model directory = %s" % model_dir)

  # Clear model directpry
  if not continue_training:
    clear_model_dir(model_dir)

  validation_metrics = {
      "accuracy":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_accuracy,
              prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
      "precision":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_precision,
              prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
      "recall":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_recall,
              prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
  }


  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      input_fn=lambda: input_fn(df_test),
      eval_steps=1,
      every_n_steps=25,
      metrics=validation_metrics,
      early_stopping_metric="accuracy",
      early_stopping_metric_minimize=True,
      early_stopping_rounds=50)

  m = build_estimator(model_dir)
  print(m.get_params(deep=True))
  # run process multiple times
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)#, monitors=[validation_monitor])
  # evaluate using tensorflow evaluation
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
        print("%s: %s" % (key, results[key]))


if __name__ == "__main__":
  train_and_eval(20000, continue_training=False)


