import gc
from datetime import datetime
import pandas as pd
from keras.models import load_model
from ensemble_processing import load_data, load, save
from ensemble_processing import fix_categorical
from ensemble_processing import divide_data, train_preprocessor, execute_preprocessor
from ensemble_processing import load_xgb_models
from ensemble_processing import train_keras_nn, train_xgb_models
from ensemble_processing import execute_train_test_predictions
from ensemble_processing import train_deep_bagging, execute_deep_bagging
from ensemble_processing import assess_results
from stats_operations import k_mean_absolute_percentage_error, k_mae_mape
from print_logger import initialise_print_logger, print


def main(run_config):
    # Prepare run_str
    run_str = datetime.now().strftime('%Y%m%d%H%M')

    initialise_print_logger('logs/training-' + run_str + '.log')

    print('Starting sharecast run:', run_str)

    # Check whether we can skip all preprocessing steps
    needs_preprocessing = False

    if run_config.get('use_previous_training_weights') is True:
        use_previous_training_weights = True
    else:
        use_previous_training_weights = False

    if run_config.get('load_data') is True:
        needs_preprocessing = True

    if run_config.get('train_pre_process') is True:
        needs_preprocessing = True

    # Retrieve and divide data
    if needs_preprocessing:
        if run_config.get('load_data') is True:
            # Load and divide data
            if run_config.get('generate_labels') is True:
                share_data = load_data(run_config['unlabelled_data_file'],
                                       drop_unlabelled=True,
                                       drop_labelled=False,
                                       generate_labels=True,
                                       label_weeks=run_config['generate_label_weeks'],
                                       reference_date=run_config['reference_date'],
                                       labelled_file_name=run_config['labelled_data_file']
                                       )
            else:
                share_data = load_data(run_config['labelled_data_file'])
            gc.collect()

            # Divide data into symbol sand general data for training an testing
            symbol_map, df_all_train_y, df_all_train_actuals, df_all_train_x, df_all_test_actuals, \
                df_all_test_y, df_all_test_x = divide_data(share_data)

            del symbol_map
            del share_data
            gc.collect()

            # Save data after dividing
            df_all_train_x.to_pickle(
                'data/pp_train_x_df.pkl.gz', compression='gzip')
            df_all_train_y.to_pickle(
                'data/df_all_train_y.pkl.gz', compression='gzip')
            df_all_train_actuals.to_pickle(
                'data/df_all_train_actuals.pkl.gz', compression='gzip')
            df_all_test_x.to_pickle(
                'data/pp_test_x_df.pkl.gz', compression='gzip')
            df_all_test_y.to_pickle(
                'data/df_all_test_y.pkl.gz', compression='gzip')
            df_all_test_actuals.to_pickle(
                'data/df_all_test_actuals.pkl.gz', compression='gzip')

        else:
            # Data already divided
            print('Loading divided data')
            df_all_train_x = pd.read_pickle(
                'data/pp_train_x_df.pkl.gz', compression='gzip')
            df_all_train_y = pd.read_pickle(
                'data/df_all_train_y.pkl.gz', compression='gzip')
            df_all_train_actuals = pd.read_pickle(
                'data/df_all_train_actuals.pkl.gz', compression='gzip')
            df_all_test_x = pd.read_pickle(
                'data/pp_test_x_df.pkl.gz', compression='gzip')
            df_all_test_y = pd.read_pickle(
                'data/df_all_test_y.pkl.gz', compression='gzip')
            df_all_test_actuals = pd.read_pickle(
                'data/df_all_test_actuals.pkl.gz', compression='gzip')

        # Retain model names for train and test
        print('Retaining model name data')
        train_model_names = df_all_train_x['model'].values
        train_gics_sectors = df_all_train_x['GICSSector'].values
        train_gics_industry_groups = df_all_train_x['GICSIndustryGroup'].values
        train_gics_industries = df_all_train_x['GICSIndustry'].values

        test_model_names = df_all_test_x['model'].values
        test_gics_sectors = df_all_test_x['GICSSector'].values
        test_gics_industry_groups = df_all_test_x['GICSIndustryGroup'].values
        test_gics_industry_groups = df_all_test_x['GICSIndustry'].values

        # Fix the names used in the GICS data - remove '&' ',' and ' '
        train_gics_sectors = fix_categorical(train_gics_sectors)
        train_gics_industry_groups = fix_categorical(
            train_gics_industry_groups)
        train_gics_industries = fix_categorical(train_gics_industries)
        test_gics_sectors = fix_categorical(test_gics_sectors)
        test_gics_industry_groups = fix_categorical(test_gics_industry_groups)
        test_gics_industry_groups = fix_categorical(test_gics_industry_groups)

        save(train_model_names, 'data/train_x_model_names.pkl.gz')
        save(train_gics_sectors, 'data/train_x_GICSSector.pkl.gz')
        save(train_gics_industry_groups, 'data/train_x_GICSIndustryGroup.pkl.gz')
        save(train_gics_industries, 'data/train_x_GICSIndustry.pkl.gz')

        save(test_model_names, 'data/test_x_model_names.pkl.gz')
        save(test_gics_sectors, 'data/test_x_GICSSector.pkl.gz')
        save(test_gics_industry_groups, 'data/test_x_GICSIndustryGroup.pkl.gz')
        save(test_gics_industry_groups, 'data/test_x_GICSIndustry.pkl.gz')

        # Drop model names and GICS values
        df_all_train_x = df_all_train_x.drop(
            ['model', 'GICSSector', 'GICSIndustryGroup', 'GICSIndustry'], axis=1)
        df_all_test_x = df_all_test_x.drop(
            ['model', 'GICSSector', 'GICSIndustryGroup', 'GICSIndustry'], axis=1)

        df_all_train_x.info()
        df_all_test_x.info()

        if run_config.get('train_pre_process') is True:
            # Execute pre-processing trainer
            df_all_train_x, symbol_encoder, imputer, scaler = train_preprocessor(
                df_all_train_x)
            df_all_test_x = execute_preprocessor(
                df_all_test_x, symbol_encoder, imputer, scaler)

            # Write processed data to files
            df_all_train_x.to_pickle(
                'data/df_all_train_x.pkl.gz', compression='gzip')
            df_all_test_x.to_pickle(
                'data/df_all_test_x.pkl.gz', compression='gzip')

        if run_config.get('load_and_execute_pre_process') is True:
            print('Loading pre-processing models')
            # Load pre-processing models
            symbol_encoder = load('models/se.pkl.gz')
            imputer = load('models/imputer.pkl.gz')
            scaler = load('models/scaler.pkl.gz')

            print('Executing pre-processing')
            # Execute pre-processing
            df_all_train_x = execute_preprocessor(
                df_all_train_x, symbol_encoder, imputer, scaler)
            df_all_test_x = execute_preprocessor(
                df_all_test_x, symbol_encoder, imputer, scaler)

            # Write processed data to files
            df_all_train_x.to_pickle(
                'data/df_all_train_x.pkl.gz', compression='gzip')
            df_all_test_x.to_pickle(
                'data/df_all_test_x.pkl.gz', compression='gzip')

    else:
        print('Load model name data')
        train_model_names = load('data/train_x_model_names.pkl.gz')
        test_model_names = load('data/test_x_model_names.pkl.gz')
        train_gics_sectors = load('data/train_x_GICSSector.pkl.gz')
        train_gics_industry_groups = load(
            'data/train_x_GICSIndustryGroup.pkl.gz')
        train_gics_industries = load('data/train_x_GICSIndustry.pkl.gz')
        test_gics_sectors = load('data/test_x_GICSSector.pkl.gz')
        test_gics_industry_groups = load(
            'data/test_x_GICSIndustryGroup.pkl.gz')
        test_gics_industry_groups = load('data/test_x_GICSIndustry.pkl.gz')

    if run_config.get('load_processed_data') is True:
        print('Loading pre-processed data')
        df_all_train_x = pd.read_pickle(
            'data/df_all_train_x.pkl.gz', compression='gzip')
        df_all_train_y = pd.read_pickle(
            'data/df_all_train_y.pkl.gz', compression='gzip')
        df_all_train_actuals = pd.read_pickle(
            'data/df_all_train_actuals.pkl.gz', compression='gzip')
        df_all_test_x = pd.read_pickle(
            'data/df_all_test_x.pkl.gz', compression='gzip')
        df_all_test_y = pd.read_pickle(
            'data/df_all_test_y.pkl.gz', compression='gzip')
        df_all_test_actuals = pd.read_pickle(
            'data/df_all_test_actuals.pkl.gz', compression='gzip')

    if run_config.get('train_keras') is True:
        # Train keras models
        keras_models = train_keras_nn(df_all_train_x, df_all_train_y, df_all_train_actuals, df_all_test_actuals,
                                      df_all_test_y, df_all_test_x, use_previous_training_weights)
    else:
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

    if run_config.get('train_xgb') is True:
        train_xgb_models(df_all_train_x, df_all_train_y, train_model_names, test_model_names,
                         df_all_test_actuals, df_all_test_y, df_all_test_x, keras_models, 'symbol')

    print('Loading xgboost symbol model list')
    xgb_models = load_xgb_models('symbol')

    if run_config.get('train_industry_xgb') is True:
        train_xgb_models(df_all_train_x, df_all_train_y, train_gics_industry_groups,
                         test_gics_industry_groups, df_all_test_actuals, df_all_test_y,
                         df_all_test_x, keras_models, 'industry')

    print('Loading xgboost industry model list')
    xgb_industry_models = load_xgb_models('industry')

    # Export data prior to bagging
    train_predictions, test_predictions = execute_train_test_predictions(df_all_train_x,
                                                                         train_model_names,
                                                                         train_gics_industry_groups,
                                                                         df_all_train_actuals,
                                                                         df_all_test_x,
                                                                         test_model_names,
                                                                         test_gics_industry_groups,
                                                                         df_all_test_actuals,
                                                                         xgb_models,
                                                                         xgb_industry_models,
                                                                         keras_models)

    if run_config.get('train_deep_bagging') is True:
        bagging_model, bagging_scaler, deep_bagged_predictions = train_deep_bagging(train_predictions,
                                                                                    df_all_train_actuals,
                                                                                    test_predictions,
                                                                                    df_all_test_actuals,
                                                                                    use_previous_training_weights)
    else:
        print('Loading bagging models')
        bagging_model = load_model('models/keras-bagging-model.h5')
        bagging_scaler = load('models/deep-bagging-scaler.pkl.gz')
        deep_bagged_predictions = execute_deep_bagging(
            bagging_model, bagging_scaler, test_predictions)

    # Add deep bagged predictions to set
    test_predictions['deep_bagged_predictions'] = deep_bagged_predictions

    assess_results(test_predictions, test_model_names,
                   df_all_test_actuals, run_str)

    print('Execution completed')


if __name__ == "__main__":
    RUN_CONFIG = {
        'load_data': False,
        'generate_labels': False,
        'generate_label_weeks': 8,
        'reference_date': '2018-05-12',
        'unlabelled_data_file': './data/ml-20180512-processed.pkl.gz',
        'labelled_data_file': './data/ml-20180512-labelled.pkl.gz',
        'train_pre_process': False,
        'load_and_execute_pre_process': False,
        'load_processed_data': True,
        'train_keras': False,
        'use_previous_training_weights': False,
        'train_xgb': False,
        'train_industry_xgb': False,
        'train_deep_bagging': True,
    }

    main(RUN_CONFIG)
