import gc
from datetime import datetime
import pandas as pd
from keras.models import load_model
from print_logger import initialise_print_logger, print
from processing_constants import LABEL_COLUMN
from ensemble_processing import load_data, load
from ensemble_processing import fix_categorical
from ensemble_processing import execute_preprocessor
from ensemble_processing import load_xgb_models
from ensemble_processing import execute_model_predictions
from ensemble_processing import execute_deep_bagging
from ensemble_processing import assess_results
from stats_operations import k_mean_absolute_percentage_error, k_mae_mape


def prepare_data_for_model(share_data: pd.DataFrame, include_y=True):
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

        x_dfs.append(model_data.drop(
            [LABEL_COLUMN, LABEL_COLUMN + '_scaled'], axis=1, errors='ignore'))

        if include_y is True:
            # Only include label data if it is part of the data set
            y_dfs.append(model_data[LABEL_COLUMN + '_scaled'])
            actuals_dfs.append(model_data[LABEL_COLUMN])

    # Create concatenated dataframes with all data
    print('Creating concatenated dataframes')

    df_all_x = pd.concat(x_dfs)
    del x_dfs
    gc.collect()

    symbol_date_df = pd.DataFrame()
    symbol_date_df['symbol'] = df_all_x['symbol']
    symbol_date_df['GICSSector'] = df_all_x['GICSSector']
    symbol_date_df['GICSIndustryGroup'] = df_all_x['GICSIndustryGroup']
    symbol_date_df['GICSIndustry'] = df_all_x['GICSIndustry']
    symbol_date_df['prediction_date'] = df_all_x.index

    if include_y:
        df_all_y = pd.concat(y_dfs)
        del y_dfs
        gc.collect()

        df_all_actuals = pd.concat(actuals_dfs)
        del actuals_dfs
        gc.collect()

    if include_y:
        return df_all_x, df_all_y, df_all_actuals, symbol_date_df

    return df_all_x, symbol_date_df


def output_predictions(predictions, df_symbol_date, file_name):
    """
        Generate output file for predictions as csv
    """

    # Retrieve baaged prediction
    pred_df = predictions['deep_bagged_predictions']

    # Create  dataframe by resetting the index to allow columns to be concatenated
    output_df = pd.concat([df_symbol_date.reset_index(
        drop=True), pred_df.reset_index(drop=True)], axis=1)

    # Save output to file
    pred_file_location = './predictions/' + file_name + '.csv'
    print('Writing predictions to', pred_file_location)
    output_df.to_csv(pred_file_location)


def main(**kwargs):
    """Execute predictions for a dataset using the trained models
       Arguments:
        data_file=None
        predict_unlabelled=False
        output_preds=True
        eval_results=True
    """
    data_file = kwargs.get('data_file', None)
    predict_unlabelled = kwargs.get('predict_unlabelled', False)
    output_preds = kwargs.get('output_preds', True)
    eval_results = kwargs.get('eval_results', True)

    # Prepare run_str
    run_str = datetime.now().strftime('%Y%m%d%H%M')

    initialise_print_logger('logs/execution-' + run_str + '.log')

    print('Starting sharecast prediction:', run_str)

    # Load and divide data
    share_data = load_data(data_file)
    gc.collect()

    # Divide data into symbols and general data for training an testing
    if predict_unlabelled:
        # Only return x values
        df_all_x, df_symbol_date = prepare_data_for_model(share_data, False)
    else:
        # Return x and y values
        df_all_x, df_all_y, df_all_actuals, df_symbol_date = prepare_data_for_model(
            share_data, True)

        del df_all_y

    del share_data
    gc.collect()

    # Retain model names for train and test
    print('Retaining model name data')
    model_names = df_all_x['model'].values
    gics_sectors = df_all_x['GICSSector'].values
    gics_industry_groups = df_all_x['GICSIndustryGroup'].values
    gics_industries = df_all_x['GICSIndustry'].values

    # Fix the names used in the GICS data - remove '&' ',' and ' '
    gics_sectors = fix_categorical(gics_sectors)
    gics_industry_groups = fix_categorical(gics_industry_groups)
    gics_industries = fix_categorical(gics_industries)

    # Drop model names and GICS values
    df_all_x = df_all_x.drop(
        ['model', 'GICSSector', 'GICSIndustryGroup', 'GICSIndustry'], axis=1)

    print('Loading pre-processing models')
    # Load pre-processing models
    symbol_encoder = load('models/se.pkl.gz')
    imputer = load('models/imputer.pkl.gz')
    scaler = load('models/scaler.pkl.gz')

    print('Executing pre-processing')
    # Execute pre-processing
    df_all_x = execute_preprocessor(df_all_x, symbol_encoder, imputer, scaler)

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

    print('Loading xgboost industry model list')
    xgb_industry_models = load_xgb_models('industry')

    predictions = execute_model_predictions(
        df_all_x, model_names, gics_industry_groups, xgb_models, xgb_industry_models, keras_models)

    print('Loading bagging models')
    bagging_model = load_model('models/keras-bagging-model.h5', custom_objects={
        'k_mean_absolute_percentage_error': k_mean_absolute_percentage_error,
        'k_mae_mape': k_mae_mape,
    })
    bagging_scaler = load('models/deep-bagging-scaler.pkl.gz')
    deep_bagged_predictions = execute_deep_bagging(
        bagging_model, bagging_scaler, predictions)
    predictions['deep_bagged_predictions'] = deep_bagged_predictions

    if eval_results:
        assess_results(predictions, model_names, df_all_actuals, run_str)

    if output_preds:
        output_predictions(predictions, df_symbol_date, run_str)

    print('Prediction completed')


if __name__ == "__main__":

    main(data_file='./data/ml-20180714-processed.pkl.gz',
         generate_labels=True,
         predict_unlabelled=False,
         output_preds=True,
         eval_results=True
         )
