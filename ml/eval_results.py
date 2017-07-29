from __future__ import print_function
import pandas as pd
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

def range_results(predictions, actuals):
    """
    
    :param predictions: a dictionary with values {
        prediction_name:  y_predict
     }
    :return: results dictionary as {
        range_name: {
            prediction_name: {
                'err': err,
                'mae': mae,
                'mape': mape,
                'r2': r2,
            }
        }
     }
    """
    overall_results_output = pd.DataFrame()

    result_ranges = [-50, -25, -10, -5, 0, 2, 5, 10, 20, 50, 100, 1001]
    lower_range = -100

    for upper_range in result_ranges:
        range_mask = (actuals >= lower_range) & (actuals <= upper_range)

        # Generate final and combined results
        range_actuals = actuals[range_mask]
        range_results = {}

        for prediction_name in predictions:
            all_predictions = predictions[prediction_name]
            # Reshape array - being sent in a dictionary loses the shape
            all_predictions = all_predictions.reshape(all_predictions.shape[0], 1)
            range_predictions = all_predictions[range_mask]

            range_results[prediction_name] = {
                'mae': mean_absolute_error(range_actuals, range_predictions),
                'mape': safe_mape(range_actuals, range_predictions)
            }

        # Print results
        print('Results:', lower_range, 'to', upper_range)

        pd_dict = {
            'lower_range': [lower_range],
            'upper_range': [upper_range]
        }

        for result in range_results:
            print('  ' + result)
            print('    Mean absolute error: ', range_results[result]['mae'])
            print('    Mean absolute percentage error: ', range_results[result]['mape'])

            pd_dict[result + '_mae'] = [range_results[result]['mae']]
            pd_dict[result + '_mape'] = [range_results[result]['mape']]


        overall_results_output = overall_results_output.append(pd.DataFrame.from_dict(pd_dict))

        lower_range = upper_range

    return overall_results_output

def eval_results(predictions):
    """
    :param predictions: a dictionary with values {
        prediction_name: {
            log_y:,
            actual_y:,
            log_y_predict:,
            y_predict
        }
     }
    :return: results dictionary as {
        prediction_name: {
            'err': err,
            'mae': mae,
            'mape': mape,
            'r2': r2,
        }
     }
    """

    results = {}

    for prediction_name in predictions:
        err = mean_absolute_error(predictions[prediction_name]['log_y'], predictions[prediction_name]['log_y_predict'])
        mae = mean_absolute_error(predictions[prediction_name]['actual_y'], predictions[prediction_name]['y_predict'])
        mape = safe_mape(predictions[prediction_name]['actual_y'], predictions[prediction_name]['y_predict'])
        r2 = r2_score(predictions[prediction_name]['actual_y'], predictions[prediction_name]['y_predict'])

        print(prediction_name)
        print('Mean log of error: %s' % err)
        print('Mean absolute error: %s' % mae)
        print('Mean absolute percentage error: %s' % mape)
        print('r2: %s' % r2)

        results[prediction_name] = {
            'err': err,
            'mae': mae,
            'mape': mape,
            'r2': r2,
        }

    return results