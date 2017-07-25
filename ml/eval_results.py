import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def safe_mape(actual_y, prediction_y):
    """
    Calculate mean absolute percentage error

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    denominators = actual_y.copy()
    set_pos = (denominators >= 0) & (denominators <= 1)
    set_neg = (denominators >= -1) & (denominators < 0)
    denominators[set_pos] = 1
    denominators[set_neg] = -1

    return np.mean(np.absolute((prediction_y - actual_y) / denominators * 100))

def eval_results(result_name, log_y, actual_y, log_y_predict, y_predict):
    """ Calculate errors, print to console and return values """
    err = mean_absolute_error(log_y, log_y_predict)
    mae = mean_absolute_error(actual_y, y_predict)
    mape = safe_mape(actual_y, y_predict)
    r2 = r2_score(actual_y, y_predict)

    print(result_name)
    print("Mean log of error: %s" % err)
    print("Mean absolute error: %s" % mae)
    print("Mean absolute percentage error: %s" % mape)
    print("r2: %s" % r2)

    return {
        'err': err,
        'mae': mae,
        'mape': mape,
        'r2': r2,
    }