from xgboost_general_symbol_ensemble_sharecast import *

# @profile
def prepare_data_for_model(share_data):
    symbol_models = {}
    symbols = share_data['symbol'].unique()
    # For testing only take the first 10 elements
    # symbols = symbols[:10]
    symbol_map = {}
    symbol_num = 0

    symbol_models = []

    print('No of symbols:', len(symbols))

    df_all_x = pd.DataFrame()
    df_all_y = pd.DataFrame()
    df_all_actuals = pd.DataFrame()



    # prep data for fitting into both model types
    for symbol in symbols:
        gc.collect()


        # Take copy of model data and re-set the pandas indexes
        # model_data = df_train_transform.loc[df_train_transform['symbol'] == symbol_num]
        model_data = share_data.loc[share_data['symbol'] == symbol]

        print('Symbol:', symbol, 'number of records:', len(model_data))

        model_data = append_recurrent_columns(model_data)

        model_data.loc[:, 'model'] = model_data.loc[:, 'symbol']

        df_all_y = pd.concat([df_all_y, model_data[LABEL_COLUMN + '_scaled']])
        df_all_actuals = pd.concat([df_all_actuals, model_data[LABEL_COLUMN]])
        df_all_x = pd.concat([df_all_x, model_data.drop([LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1)])

    return df_all_x, df_all_y, df_all_actuals

def main(run_config):
    # Prepare run_str
    run_str = datetime.datetime.now().strftime('%Y%m%d%H%M')

    print('Starting sharecast prediction:', run_str)

    # Load and divide data
    share_data = load_data(run_config['data_file'])
    gc.collect()

    # Divide data into symbol sand general data for training an testing
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
    imputer = load('models/imputer.pkl.gz')
    scaler = load('models/scaler.pkl.gz')
    ce = load('models/ce.pkl.gz')

    print('Executing pre-processing')
    # Execute pre-processing
    df_all_x = execute_preprocessor(df_all_x, imputer, scaler, ce)


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
        'mae_intermediate_model': load_model('models/keras-mae-intermediate-model.h5', custom_objects={
            'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
            'k_mae_mape': k_mae_mape,
        }),
    }

    print('Loading xgboost model list')
    xgb_models = load_xgb_models()
    predictions = execute_model_predictions(df_all_x, x_model_names, xgb_models, keras_models)

    print('Loading bagging models')
    bagging_model = load_model('models/keras-bagging-model.h5', custom_objects={
            'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
            'k_mae_mape': k_mae_mape,
    })
    bagging_scaler = load('models/deep-bagging-scaler.pkl.gz')
    deep_bagged_predictions = execute_deep_bagging(bagging_model, bagging_scaler, predictions)
    predictions['deep_bagged_predictions'] = deep_bagged_predictions

    if 'eval_results' in run_config and run_config['eval_results'] == True:
        assess_results(predictions, x_model_names, df_all_actuals, run_str)


if __name__ == "__main__":
    run_config = {
        'data_file': './data/ml-2018-03-data.pkl.gz',
        'eval_results': True,
    }

    main(run_config)