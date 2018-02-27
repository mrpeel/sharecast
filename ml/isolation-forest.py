from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def convert_anomaly_values(np_array):
    neg_mask = np_array < 0
    converted_array = np_array
    converted_array[neg_mask] = 'O'
    converted_array[~neg_mask] = 'I'

    unique, counts = np.unique(converted_array, return_counts=True)
    count_vals = dict(zip(unique, counts))

    return converted_array, count_vals

def train_anomaly_detector(X_train_data, X_test_data, Y_test_data, visualise_result=False):
    # fit the model
    if_model = IsolationForest(n_estimators=200, n_jobs=-1, bootstrap=True, verbose=1)
    if_model.fit(X_train_data)

    training_predictions = if_model.predict(X_train_data)
    outliers_mask = training_predictions < 0
    print('Anomaly stats for training data')
    print('Inlier values: ', len(X_train_data[~outliers_mask]))
    print('Outlier values: ', len(X_train_data[outliers_mask]))

    # Execute predictions on test data
    test_predictions = if_model.predict(X_test_data)
    outliers_mask = test_predictions < 0
    print('Anomaly stats for test data')
    print('Inlier values: ', len(X_test_data[~outliers_mask]))
    print('Inlier mean absolute y value:', np.mean(np.abs(Y_test_data[~outliers_mask])))
    print('Outlier values: ', len(X_test_data[outliers_mask]))
    print('Outlier mean absolute y value:', np.mean(np.abs(Y_test_data[outliers_mask])))

    if visualise_result:
        visualise_outliers(Y_test_data[~outliers_mask], Y_test_data[outliers_mask])

    # Return the model
    return if_model


def visualise_outliers(inlier_y, outlier_y):
    plt.title("IsolationForest outlier detection")
    plt.interactive(False)

    inliers = plt.scatter(inlier_y, inlier_y, c='white', s=20, edgecolor='k')
    outliers = plt.scatter(outlier_y, outlier_y, c='red', s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((-100, 100))
    plt.ylim((-100, 100))
    plt.legend([inliers, outliers],
               ["Inlier y values",
                "Outlier y values"],
               loc="upper left")
    plt.show(block=True)


if __name__ == "__main__":
    print('Loading data files')

    df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    train_anomaly_detector(df_all_train_x.values, df_all_test_x.values, df_all_test_actuals.values, visualise_result=False)