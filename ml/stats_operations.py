import numpy as np
import numba
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error
from keras import backend as K
from print_logger import print


def round_down(num, divisor):
    return num - (num % divisor)


# @profile
@numba.jit
def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log1p(np.absolute(return_vals))
    return_vals[neg_mask] *= -1.
    return return_vals

# @profile


@numba.jit
def safe_exp(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.expm1(np.clip(np.absolute(return_vals), -7, 7))
    return_vals[neg_mask] *= -1.
    return return_vals


@numba.jit
def y_scaler(input_array):
    transformed_array = safe_log(input_array)
    scaler = MaxAbsScaler()
    # transformed_array = scaler.fit_transform(transformed_array)
    return transformed_array, scaler


@numba.jit
def y_inverse_scaler(prediction_array):
    # scaler.inverse_transform(prediction_array)
    transformed_array = prediction_array
    transformed_array = safe_exp(transformed_array)
    return transformed_array


# @profile
@numba.jit
def mle(actual_y, prediction_y):
    """
    Compute the Root Mean  Log Error

    Args:
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
    """

    return np.mean(np.absolute(safe_log(prediction_y) - safe_log(actual_y)))


# @profile
@numba.jit
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


# @profile
@numba.jit
def mae_eval(y, y0):
    y0 = y0.get_label()
    assert len(y) == len(y0)
    # return 'error', np.sqrt(np.mean(np.square(np.log(y + 1) - np.log(y0 + 1))))
    return 'error', np.mean(np.absolute(y - y0)), False


# @profile
@numba.jit
def safe_mape(actual_y, prediction_y):
    """
    Calculate mean absolute percentage error

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    # Ensure data shape is correct
    actual_y = actual_y.reshape(actual_y.shape[0], )
    prediction_y = prediction_y.reshape(prediction_y.shape[0], )
    # Calculate MAPE
    diff = np.absolute((actual_y - prediction_y) /
                       np.clip(np.absolute(actual_y), 1., None))
    return 100. * np.mean(diff)


# @profile
@numba.jit
def mape_eval(actual_y, eval_y):
    """
    Used during xgboost training

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', safe_mape(actual_y, prediction_y)


# @profile
@numba.jit
def mape_log_y(actual_y, prediction_y):
    inverse_actual = actual_y.copy()
    inverse_actual = y_inverse_scaler(inverse_actual)

    inverse_prediction = prediction_y.copy()
    inverse_prediction = y_inverse_scaler(inverse_prediction)

    return safe_mape(inverse_actual, inverse_prediction)


# @profile
@numba.jit
def mape_log_y_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', mape_log_y(actual_y, prediction_y)


@numba.jit
def mae_mape(actual_y, prediction_y):
    mape = safe_mape(actual_y, prediction_y)
    mae = mean_absolute_error(actual_y, prediction_y)
    return mape * mae


def flatten_array(np_array):
    if np_array.ndim > 1:
        new_array = np.concatenate(np_array)
    else:
        new_array = np_array

    return new_array


# @profile
@numba.jit
def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    return 100. * K.mean(diff, axis=-1)


def k_mae_mape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    mape = 100. * K.mean(diff, axis=-1)
    mae = K.mean(K.abs(y_true - y_pred), axis=-1)
    return mape * mae


def k_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    return 100. * K.mean(diff, axis=-1)
