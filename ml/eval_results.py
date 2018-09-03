import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, median_absolute_error
from print_logger import print

# @profile


def safe_mape(actual_y, prediction_y):
    """
    Calculate mean absolute percentage error

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    # Reshape arrays to ensure clip and absolute won't chew through memory
    prediction_y = prediction_y.reshape(prediction_y.shape[0], 1)
    actual_y = actual_y.reshape(actual_y.shape[0], 1)

    diff = np.absolute((actual_y - prediction_y) /
                       np.clip(np.absolute(actual_y), 1., None))
    return 100. * np.mean(diff)

# @profile


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
                'rsquared': rsquared,
            }
        }
     }
    """
    overall_results_output = pd.DataFrame()

    result_ranges = [-50, -25, -10, -5, -0.25,
                     0.25, 1, 2, 5, 10, 20, 50, 100, 1001]
    lower_range = -100

    for upper_range in result_ranges:
        range_mask = (actuals >= lower_range) & (actuals <= upper_range)

        # Generate final and combined results
        range_actuals = actuals[range_mask]
        # Re-shape the actuals array
        range_actuals = range_actuals.reshape(range_actuals.shape[0], 1)
        prediction_range_results = {}

        for prediction_name in predictions:
            all_predictions = predictions[prediction_name]
            # Reshape array - being sent in a dictionary loses the shape
            all_predictions = all_predictions.reshape(
                all_predictions.shape[0], 1)
            range_predictions = all_predictions[range_mask]

            if range_predictions.any():
                prediction_range_results[prediction_name] = {
                    'mae': mean_absolute_error(range_actuals, range_predictions),
                    'mape': safe_mape(range_actuals, range_predictions),
                    'medae': median_absolute_error(range_actuals, range_predictions),
                    'num_vals': len(range_predictions),
                }
            else:
                prediction_range_results[prediction_name] = {
                    'mae': 'N/A',
                    'mape': 'N/A,',
                    'medae': 'N/A,',
                    'num_vals': 0,
                }

        # Print results
        print('Results: %s to %s, number of instances %s' %
              (lower_range, upper_range, len(range_predictions)))

        pd_dict = {
            'lower_range': [lower_range],
            'upper_range': [upper_range]
        }

        for result in prediction_range_results:
            print('  ' + result)
            print('    Mean absolute error: ',
                  prediction_range_results[result]['mae'])
            print('    Mean absolute percentage error: ',
                  prediction_range_results[result]['mape'])
            print('    Median absolute error: ',
                  prediction_range_results[result]['medae'])

            pd_dict[result + '_mae'] = [prediction_range_results[result]['mae']]
            pd_dict[result + '_mape'] = [prediction_range_results[result]['mape']]
            pd_dict[result + '_medae'] = [prediction_range_results[result]['medae']]

        overall_results_output = overall_results_output.append(
            pd.DataFrame.from_dict(pd_dict))

        lower_range = upper_range

    return overall_results_output

# @profile


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
            'rsquared': rsquared,
        }
     }
    """

    results = {}

    for prediction_name in predictions:
        print(prediction_name)

        prediction_vals = predictions[prediction_name]
        actual_y = prediction_vals['actual_y']
        y_predict = prediction_vals['y_predict']

        num_vals = len(actual_y)
        print('Number of instances: %s' % num_vals)

        # Ensure shape is correct
        actual_y = actual_y.reshape(actual_y.shape[0], 1)
        y_predict = y_predict.reshape(y_predict.shape[0], 1)

        if prediction_vals.__contains__('log_y'):
            log_y = prediction_vals['log_y']
            log_y_predict = prediction_vals['log_y_predict']
            # Ensure shape is correct
            log_y = log_y.reshape(log_y.shape[0], 1)
            log_y_predict = log_y_predict.reshape(log_y_predict.shape[0], 1)

            err = mean_absolute_error(log_y, log_y_predict)
            print('Mean log of error: %s' % err)

        mae = mean_absolute_error(actual_y, y_predict)
        print('Mean absolute error: %s' % mae)
        mape = safe_mape(actual_y, y_predict)
        print('Mean absolute percentage error: %s' % mape)
        rsquared = r2_score(actual_y, y_predict)
        print('rsquared: %s' % rsquared)
        explain_variance = explained_variance_score(actual_y, y_predict)
        print('Explained variance: %s' % explain_variance)
        medae = median_absolute_error(actual_y, y_predict)
        print('Median absolute error: %s' % medae)

        if prediction_vals.__contains__('log_y'):
            results[prediction_name] = {
                'err': err,
                'mae': mae,
                'mape': mape,
                'rsquared': rsquared,
                'explain_variance': explain_variance,
                'medae': medae,
                'num_vals': num_vals,
            }
        else:
            results[prediction_name] = {
                'mae': mae,
                'mape': mape,
                'rsquared': rsquared,
                'explain_variance': explain_variance,
                'medae': medae,
                'num_vals': num_vals,
            }

    return results
