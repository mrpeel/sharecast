"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import xgboost as xgb

from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.models import load_model

from eval_results import *

import lightgbm as lgb

import logging

def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    return 100. * K.mean(diff, axis=-1)

def safe_log(input_array):
    return_vals = input_array.reshape(input_array.shape[0], ).copy()
    neg_mask = return_vals < 0
    return_vals = np.log(np.absolute(return_vals) + 1)
    return_vals[neg_mask] *= -1.
    return return_vals

def safe_exp(input_array):
    return_vals = input_array.reshape(input_array.shape[0], ).copy()
    neg_mask = return_vals < 0
    return_vals = np.exp(np.clip(np.absolute(return_vals), -7, 7)) - 1
    return_vals[neg_mask] *= -1.
    return return_vals


def safe_mape(actual_y, prediction_y):
    actual, prediction = reshape_vals(actual_y, prediction_y)
    diff = np.absolute((actual - prediction) / np.clip(np.absolute(actual), 1., None))
    return 100. * np.mean(diff)

def reshape_vals(actual_y, prediction_y):
    actual_y = actual_y.reshape(actual_y.shape[0], )
    prediction_y = prediction_y.reshape(prediction_y.shape[0], )
    return actual_y, prediction_y

def mape_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'mape', safe_mape(actual_y, prediction_y), False

def maepe_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'maepe', safe_maepe(actual_y, prediction_y), False

def safe_maepe(actual_y, prediction_y):
    actual, prediction = reshape_vals(actual_y, prediction_y)
    mape = safe_mape(actual, prediction)
    mae = mean_absolute_error(actual, prediction)

    return (mape * mae)


def round_down(num, divisor):
    return num - (num%divisor)

def compile_model(network):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    max_depth = network['max_depth']
    base_score = network['base_score']
    colsample_bylevel = network['colsample_bylevel']
    colsample_bytree = network['colsample_bytree']
    gamma = network['gamma']
    learning_rate = network['learning_rate']
    min_child_weight = network['min_child_weight']
    tree_method  = network['tree_method']

    model  = xgb.XGBRegressor(nthread=-1, n_estimators=5000,
                              # booster=booster,
                              max_depth=max_depth,
                              base_score=base_score,
                              colsample_bylevel=colsample_bylevel,
                              colsample_bytree=colsample_bytree,
                              gamma=gamma,
                              learning_rate=learning_rate,
                              min_child_weight=min_child_weight,
                              tree_method=tree_method)

    return model

def train_and_score_xgb(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """

    df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_train_y = pd.read_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
    df_all_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    df_all_test_y = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
    df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.as_matrix()
    test_actuals = df_all_test_actuals.as_matrix()
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)
    test_x = df_all_test_x.as_matrix()

    # Use keras model to generate x vals
    mae_intermediate_model = load_model('models/mae_intermediate_model.h5')

    mae_vals_train = mae_intermediate_model.predict(train_x)
    mae_vals_test = mae_intermediate_model.predict(test_x)

    # train = xgb.DMatrix(mae_vals_train, label=train_log_y)
    # test = xgb.DMatrix(mae_vals_test)

    model = compile_model(network)

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))


    eval_set = [(mae_vals_test, test_log_y)]
    model.fit(mae_vals_train, train_log_y, early_stopping_rounds=5, eval_metric='mae', eval_set=eval_set)
              # , verbose=False)

    # eval_set = [(test, test_log_y)]
    # xgb.train(network, train, num_boost_round=5000, evals=eval_set, early_stopping_rounds=5)


    predictions = model.predict(mae_vals_test)
    # predictions = xgb.predict(test)
    score = mean_absolute_error(test_log_y, predictions)

    print('\rResults')

    best_round = model.best_iteration
    # best_round = xgb.best_iteration

    if np.isnan(score):
        score = 9999

    print('best round:', best_round)
    print('loss:', score)
    print('-' * 20)

    logging.info('best round: %d' % best_round)
    logging.info('loss: %.4f' % score)
    logging.info('-' * 20)

    return score

def train_and_score_bagging(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """

    train_predictions = pd.read_pickle('data/train_predictions.pkl.gz', compression='gzip')
    test_predictions = pd.read_pickle('data/test_predictions.pkl.gz', compression='gzip')

    train_actuals = pd.read_pickle('data/train_actuals.pkl.gz', compression='gzip')
    test_actuals = pd.read_pickle('data/test_actuals.pkl.gz', compression='gzip')


    train_x = np.array(train_predictions.values)
    train_y = train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_x = np.array(test_predictions.values)
    test_y = test_actuals[0].values
    test_log_y = safe_log(test_y)

    model = compile_model(network)

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))

    test = xgb.DMatrix(test_x)
    train = xgb.DMatrix(train_x, label=train_log_y)



    eval_set = [(test_x, test_log_y)]
    model.fit(train_x, train_log_y, early_stopping_rounds=20, eval_metric='mae', eval_set=eval_set,
                verbose=False)

    # eval_set = [(test, test_log_y)]
    # xgb.train(network, train, num_boost_round=5000, evals=eval_set, early_stopping_rounds=5)

    predictions = model.predict(test_x)
    # predictions = xgb.predict(test_x)
    inverse_predictions = safe_exp(predictions)
    score = mean_absolute_error(test_y, inverse_predictions)
    mape = safe_mape(test_y, inverse_predictions)

    print('\rResults')

    best_round = xgb.best_iteration

    if np.isnan(score):
        score = 9999

    print('best round:', best_round)
    print('loss:', score)
    print('mape:', mape)
    print('-' * 20)

    logging.info('best round: %d' % best_round)
    logging.info('loss: %.4f' % score)
    logging.info('mape: %.4f' % mape)
    logging.info('-' * 20)

    eval_results({'xgb_predictions': {
                        'actual_y': test_y,
                        'y_predict': inverse_predictions
                }
    })

    range_results({
        'xgb_predictions': inverse_predictions,
    }, test_y)

def train_and_score_lgbm(network):

    df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_train_y = pd.read_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
    df_all_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    df_all_test_y = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
    df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_actuals = df_all_test_actuals[0].values
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)

    # Use keras model to generate x vals
    #mae_intermediate_model = load_model('models/mae_intermediate_model.h5')
    # mae_vals_train = mae_intermediate_model.predict(train_x)
    # mae_vals_test = mae_intermediate_model.predict(test_x)
    #
    train_set = lgb.Dataset(df_all_train_x, label=train_y)
    eval_set = lgb.Dataset(df_all_test_x, reference=train_set, label=test_y)


    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))

    params = network


    # feature_name and categorical_feature
    gbm = lgb.train(params,
                    train_set,
                    valid_sets=eval_set,  # eval training data
                    feval=maepe_eval,
                    # Set learning rate to reduce every 10 iterations
                    learning_rates=lambda iter: 0.125 * (0.999 ** round_down(iter, 10)),
                    num_boost_round=500,
                    early_stopping_rounds=5)


    iteration_number = 500

    if gbm.best_iteration:
        iteration_number = gbm.best_iteration

    predictions = gbm.predict(df_all_test_x, num_iteration=iteration_number)
    eval_predictions = safe_exp(predictions)
    # eval_predictions = safe_exp(safe_exp(predictions))
    # eval_predictions = predictions

    mae = mean_absolute_error(test_actuals, eval_predictions)
    mape = safe_mape(test_actuals, eval_predictions)


    print('\rResults')

    print('best iteration:', iteration_number)
    print('mae:', mae)
    print('mape:', mape)
    print('-' * 20)

    logging.info('best round: %d' % iteration_number)
    logging.info('mae: %.4f' % mae)
    logging.info('mape: %.4f' % mape)
    logging.info('-' * 20)

    eval_results({'lightgbm_predictions': {
                        'actual_y': test_actuals,
                        'y_predict': eval_predictions
                }
    })

    range_results({
        'lightgbm_predictions': eval_predictions,
    }, test_actuals)

def main(type):
    if type =="xgb":
        network = {
            'max_depth': 110,
            'base_score': 1.0,
            'colsample_bylevel': 0.95,
            'colsample_bytree': 0.7,
            'gamma': 0,
            'min_child_weight': 5,
            'learning_rate': 0.1,
            'tree_method': "hist",
            'eval_metric': "mae",
        }

        train_and_score_xgb(network)

    elif type =="lgbm":
        # network = {
        #     'num_leaves': 65536,
        #     'max_bin': 5000000,
        #     'boosting_type': "gbdt",
        #     'feature_fraction': 0.7,
        #     'min_split_gain': 0,
        #     'boost_from_average': True,
        #     'verbosity': -1,
        #     'histogram_pool_size': 8192,
        #     'metric': ['mae', 'huber'],
        #     'metric_freq': 10,
        # }

        network = {
            'objective': "huber",
            'num_leaves': 65536,
            'max_bin': 5000000,
            'boosting_type': "gbdt",
            'feature_fraction': 0.7,
            'min_split_gain': 0,
            'boost_from_average': True,
            'verbosity': -1,
            'histogram_pool_size': 8192,
            'metric': ['mae', 'huber'],
            'metric_freq': 10,
        }

        network = {
            'objective': "huber",
            'num_leaves': 16384,
            'boosting_type': "gbdt",
            'feature_fraction': 0.5,
            'bagging_fraction': 1.0,
            'bagging_freq': 2,
            'min_split_gain': 0,
            'boost_from_average': False,
            'verbosity': -1,
            'histogram_pool_size': 8192,
            'metric': ['mae', 'huber'],
            'metric_freq': 10,
        }

        # network = {
        #     'objective': "fair",
        #     'num_leaves': 65536,
        #     'max_bin': 5000000,
        #     'boosting_type': "gbdt",
        #     'feature_fraction': 0.7,
        #     'min_split_gain': 0,
        #     'boost_from_average': True,
        #     'verbosity': -1,
        #     'histogram_pool_size': 8192,
        #     'metric': ['mae', 'fair'],
        #     'metric_freq': 10,
        # }

        # network = {
        #     'num_leaves': 32768,
        #     'boosting_type': "gbdt",
        #     'feature_fraction': 0.7,
        #     'min_split_gain': 0,
        #     'boost_from_average': True,
        #     'verbosity': -1,
        #     'histogram_pool_size': 8192,
        #     'metric': ['mae', 'huber'],
        #     'metric_freq': 10,
        # }

        network = {
            'objective': "huber",
            'num_leaves': 65536,
            'boosting_type': "gbdt",
            'feature_fraction': 0.5,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'min_split_gain': 0,
            'boost_from_average': False,
            'verbosity': -1,
            'histogram_pool_size': 8192,
            'metric_freq': 10,
        }

        train_and_score_lgbm(network)


if __name__ == '__main__':
    main("lgbm")