from xgboost_general_symbol_ensemble_sharecast import *

# @profile


def prepare_data_for_model(share_data):
    # symbol_models = {}
    symbols = share_data['symbol'].unique()
    # For testing only take the first 10 elements
    # symbols = symbols[:10]
    # symbol_map = {}
    # symbol_num = 0

    # symbol_models = []

    print('No of symbols:', len(symbols))

    # Array to hold completed dataframes
    x_dfs = []
    y_dfs = []
    actuals_dfs = []

    # prep data for fitting into both model types
    for symbol in symbols:
        gc.collect()

        # Take copy of model data and re-set the pandas indexes
        # model_data = df_train_transform.loc[df_train_transform['symbol'] == symbol_num]
        model_data = share_data.loc[share_data['symbol'] == symbol]

        print('Symbol:', symbol, 'number of records:', len(model_data))

        model_data.loc[:, 'model'] = model_data.loc[:, 'symbol']

        y_dfs.append(model_data[LABEL_COLUMN + '_scaled'])
        actuals_dfs.append(model_data[LABEL_COLUMN])
        x_dfs.append(model_data.drop(
            [LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1))

    # Create concatenated dataframes with all data
    print('Creating concatenated dataframes')

    df_all_x = pd.concat(x_dfs)
    del x_dfs
    gc.collect()

    df_all_y = pd.concat(y_dfs)
    del y_dfs
    gc.collect()

    df_all_actuals = pd.concat(actuals_dfs)
    del actuals_dfs
    gc.collect()

    return df_all_x, df_all_y, df_all_actuals


def main(run_config):
    # Prepare run_str
    run_str = datetime.now().strftime('%Y%m%d%H%M')

    # Setup logging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        level=logging.DEBUG,
        filename='logs/execution-' + run_str + '.log'
    )

    print('Starting sharecast prediction:', run_str)

    # Load and divide data
    if 'generate_labels' in run_config and run_config['generate_labels'] is True:
        share_data = load_data(run_config['data_file'], run_config['generate_label_weeks'],
                               run_config['reference_date'], run_config['label_file_name'])
    else:
        share_data = load_data(run_config['data_file'])

    gc.collect()

    # Divide data into symbols and general data for training an testing
    df_all_x, df_all_y, df_all_actuals = prepare_data_for_model(share_data)

    del share_data
    gc.collect()

    # Retain model names for train and test
    print('Retaining model name data')
    x_model_names = df_all_x['model'].values

    # Drop model names
    df_all_x = df_all_x.drop(['model'], axis=1)

    print('Loading pre-processing models')
    # Load pre-processing models
    se = load('models/se.pkl.gz')
    imputer = load('models/imputer.pkl.gz')
    scaler = load('models/scaler.pkl.gz')

    print('Executing pre-processing')
    # Execute pre-processing
    df_all_x = execute_preprocessor(df_all_x, se, imputer, scaler)

    print('Loading keras models')
    # Load keras models
    keras_models = {
        'mape_model': load_model('models/keras-mape-model.h5', custom_objects={
            'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
            'k_mae_mape': k_mae_mape,
        }),
        'mae_model': load_model('models/keras-mae-model.h5', custom_objects={
            'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
            'k_mae_mape': k_mae_mape,
        }),
        'mae_intermediate_model': load_model('models/keras-mae-intermediate-model.h5',
                                             custom_objects={
                                                 'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
                                                 'k_mae_mape': k_mae_mape,
                                             }),
    }

    print('Loading xgboost model list')
    xgb_models = load_xgb_models()
    predictions = execute_model_predictions(
        df_all_x, x_model_names, xgb_models, keras_models)

    print('Loading bagging models')
    bagging_model = load_model('models/keras-bagging-model.h5', custom_objects={
        'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
        'k_mae_mape': k_mae_mape,
    })
    bagging_scaler = load('models/deep-bagging-scaler.pkl.gz')
    deep_bagged_predictions = execute_deep_bagging(
        bagging_model, bagging_scaler, predictions)
    predictions['deep_bagged_predictions'] = deep_bagged_predictions

    if 'eval_results' in run_config and run_config['eval_results'] is True:
        assess_results(predictions, x_model_names, df_all_actuals, run_str)

    print('Prediction completed')


if __name__ == "__main__":
    run_config = {
        'data_file': './data/ml-20180512-labelled.pkl.gz',
        'generate_labels': False,
        'generate_label_weeks': 8,
        'reference_date': '2018-05-12',
        'label_file_name': '',
        'eval_results': True,
    }

    main(run_config)
