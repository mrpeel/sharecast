"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from compile_keras import *
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os

from eval_results import *
from random import *

def mae_mape(actual_y, prediction_y):
    mape = safe_mape(actual_y, prediction_y)
    mae = mean_absolute_error(actual_y, prediction_y)
    return mape * mae

def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log(np.absolute(return_vals) + 1)
    return_vals[neg_mask] *= -1.
    return return_vals

def safe_exp(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.exp(np.clip(np.absolute(return_vals), -7, 7)) - 1
    return_vals[neg_mask] *= -1.
    return return_vals

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
    diff = np.absolute((actual_y - prediction_y) / np.clip(np.absolute(actual_y), 1., None))
    return 100. * np.mean(diff)


def train_and_score(network):
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
    train_x = df_all_train_x.values
    test_actuals = df_all_test_actuals.values
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)
    test_x = df_all_test_x.values


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=26)
    csv_logger = CSVLogger('./logs/training.log')

    dimensions = train_x.shape[1]

    # Set use of log of y or y
    if network['log_y']:
        train_eval_y = train_y
        test_eval_y = test_y
    else:
        train_eval_y = train_actuals
        test_eval_y = test_actuals

    if 'epochs' in network:
        epochs = network['epochs']
    else:
        epochs = 10000

    results = {
        'mae': [],
        'mape': [],
        'maeape': [],
        'epochs': [],
    }

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])

    num_folds = 2

    for _ in range(num_folds):
        #  Clear all values
        s = None
        x_cv_train = None
        y_cv_train = None
        model = None
        history = None
        hist_epochs = None

        run_id = randint(1, 1000000)
        run_id = format(run_id, '06')
        weights_location = './weights/' + run_id + 'weights.hdf5'

        # Reorder array - get array index
        s = np.arange(train_x.shape[0])
        # Reshuffle index
        np.random.shuffle(s)

        # Create array using new index
        x_cv_train = train_x[s]
        y_cv_train = train_eval_y[s]

        model = compile_keras_model(network, dimensions)

        checkpointer = ModelCheckpoint(filepath=weights_location, verbose=0, save_best_only=True)

        history = model.fit(x_cv_train, y_cv_train,
                            batch_size=network['batch_size'],
                            epochs=epochs,  # using early stopping, so no real limit
                            verbose=0,
                            validation_split=0.2,
                            callbacks=[early_stopping, csv_logger, reduce_lr, checkpointer])

        hist_epochs = len(history.history['val_loss'])

        model.load_weights(weights_location)
        predictions = model.predict(test_x)
        prediction_results = predictions.reshape(predictions.shape[0], )

        # If using log of y, get exponent
        if network['log_y']:
            prediction_results = safe_exp(prediction_results)

        mae = mean_absolute_error(test_actuals, prediction_results)
        mape = safe_mape(test_actuals, prediction_results)
        maeape = mae_mape(test_actuals, prediction_results)


        results['mae'].append(mae)
        results['mape'].append(mape)
        results['maeape'].append(maeape)
        results['epochs'].append(hist_epochs)

        print('\rFold results')


        print('epochs:', hist_epochs)
        print('mae_mape:', maeape)
        print('mape:', mape)
        print('mae:', mae)
        print('-' * 20)

        eval_results({'bagged_predictions': {
            'actual_y': test_actuals,
            'y_predict': prediction_results
        }
        })

        range_results({
            'bagged_predictions': prediction_results,
        }, test_actuals)

        # Delete weights file, if found
        try:
            os.remove(weights_location)
        except:
            pass

    overall_scores = {
        'mae': np.mean(results['mae']),
        'mape': np.mean(results['mape']),
        'maeape': np.mean(results['maeape']),
        'epochs': np.mean(results['epochs']),
    }

    print('-' * 20)
    print('\rOverall Results')

    print('epochs:', overall_scores['epochs'])
    print('mae_mape:', overall_scores['maeape'])
    print('mape:', overall_scores['mape'])
    print('mae:', overall_scores['mae'])
    print('-' * 20)



def train_and_score_bagging(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """

    train_predictions = pd.read_pickle('data/train_predictions.pkl.gz', compression='gzip')
    test_predictions = pd.read_pickle('data/test_predictions.pkl.gz', compression='gzip')

    train_actuals = pd.read_pickle('data/train_actuals.pkl.gz', compression='gzip')
    test_actuals = pd.read_pickle('data/test_actuals.pkl.gz', compression='gzip')


    train_x = train_predictions.values
    train_y = train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_x = test_predictions.values
    test_y = test_actuals[0].values
    test_log_y = safe_log(test_y)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=36)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    input_shape = (train_x.shape[1],)


    model = compile_keras_model(network, input_shape)

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])

    history = model.fit(train_x, train_y,
    # history = model.fit(train_x, train_log_y,
                        batch_size=network['batch_size'],
                        epochs=10000,  # using early stopping, so no real limit
                        verbose=0,
                        validation_data=(test_x, test_y),
                        # validation_data=(test_x, test_log_y),
                        callbacks=[early_stopping, csv_logger, reduce_lr, checkpointer])


    print('\rResults')

    hist_epochs = len(history.history['val_loss'])
    # score = history.history['val_loss'][hist_epochs - 1]

    model.load_weights('weights.hdf5')

    predictions = model.predict(test_x)
    prediction_results = predictions.reshape(predictions.shape[0],)
    # prediction_results = safe_exp(prediction_results)

    eval_results({'bagged_predictions': {
                        'actual_y': test_y,
                        'y_predict': prediction_results
                }
    })

    range_results({
        'bagged_predictions': prediction_results,
    }, test_y)

def train_and_score_shallow_bagging(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """

    train_predictions = pd.read_pickle('data/train_predictions.pkl.gz', compression='gzip')
    test_predictions = pd.read_pickle('data/test_predictions.pkl.gz', compression='gzip')

    train_actuals = pd.read_pickle('data/train_actuals.pkl.gz', compression='gzip')
    test_actuals = pd.read_pickle('data/test_actuals.pkl.gz', compression='gzip')

    target_columns = ['xgboost_keras_log', 'xgboost_keras_log_log', 'xgboost_log', 'keras_mape']

    cols_to_drop = []
    for col in train_predictions.columns:
        if col not in target_columns:
            cols_to_drop.append(col)

    print('Dropping columns:', list(cols_to_drop))
    train_predictions.drop(cols_to_drop, axis=1, inplace=True)

    cols_to_drop = []
    for col in test_predictions.columns:
        if col not in target_columns:
            cols_to_drop.append(col)

    print('Dropping columns:', list(cols_to_drop))
    test_predictions.drop(cols_to_drop, axis=1, inplace=True)

    train_x = train_predictions.values
    train_y = train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_x = test_predictions.values
    test_y = test_actuals[0].values
    test_log_y = safe_log(test_y)

    # Set use of log of y or y
    if network['log_y']:
        train_eval_y = train_log_y
        test_eval_y = test_log_y
    else:
        train_eval_y = train_y
        test_eval_y = test_y

    if 'epochs' in network:
        epochs = network['epochs']
    else:
        epochs = 10000

    # Apply value scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    results = {
        'mae': [],
        'mape': [],
        'maeape': [],
        'epochs': [],
    }

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])

    num_folds = 2

    for _ in range(num_folds):
        #  Clear all values
        s = None
        x_cv_train = None
        y_cv_train = None
        model = None
        history = None
        hist_epochs = None

        # Delete weights file, if exists
        try:
            os.remove('weights.hdf5')
        except:
            pass

        # Reorder array - get array index
        s = np.arange(train_x_scaled.shape[0])
        # Reshuffle index
        np.random.shuffle(s)

        # Create array using new index
        x_cv_train = train_x_scaled[s]
        y_cv_train = train_eval_y[s]

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=2)
        early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        csv_logger = CSVLogger('./logs/training.log')
        checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

        input_shape = train_x_scaled.shape[1]

        model = compile_keras_model(network, input_shape)

        # history = model.fit(train_x, train_y,
        history = model.fit(x_cv_train, y_cv_train,
                            batch_size=network['batch_size'],
                            epochs=epochs,
                            verbose=0,
                            validation_split=0.2,
                            callbacks=[early_stopping, csv_logger, reduce_lr, checkpointer])


        model.load_weights('weights.hdf5')
        predictions = model.predict(test_x_scaled)
        prediction_results = predictions.reshape(predictions.shape[0],)

        # If using log of y, get exponent
        if network['log_y']:
            prediction_results = safe_exp(prediction_results)


        mae = mean_absolute_error(test_y, prediction_results)
        mape = safe_mape(test_y, prediction_results)
        maeape = mae_mape(test_y, prediction_results)

        hist_epochs = len(history.history['val_loss'])

        results['mae'].append(mae)
        results['mape'].append(mape)
        results['maeape'].append(maeape)
        results['epochs'].append(hist_epochs)

        print('\rFold results')


        print('epochs:', hist_epochs)
        print('mae_mape:', maeape)
        print('mape:', mape)
        print('mae:', mae)
        print('-' * 20)

        eval_results({'bagged_predictions': {
            'actual_y': test_y,
            'y_predict': prediction_results
        }
        })

        range_results({
            'bagged_predictions': prediction_results,
        }, test_y)

    overall_scores = {
        'mae': np.mean(results['mae']),
        'mape': np.mean(results['mape']),
        'maeape': np.mean(results['maeape']),
        'epochs': np.mean(results['epochs']),
    }


    print('\rResults')

    print('epochs:', overall_scores['epochs'])
    print('mae_mape:', overall_scores['maeape'])
    print('mape:', overall_scores['mape'])
    print('mae:', overall_scores['mae'])
    print('-' * 20)


def train_and_score_entity_embedding(network):
    # Creating the neural network
    embeddings = []
    inputs = []
    __Enc = dict()
    __K = dict()

    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout = network['dropout']
    batch_size = network['batch_size']



    train_x_df = pd.read_pickle('data/pp_train_x_df.pkl.gz', compression='gzip')
    train_y_df = pd.read_pickle('data/pp_train_y_df.pkl.gz', compression='gzip')
    test_x_df = pd.read_pickle('data/pp_test_x_df.pkl.gz', compression='gzip')
    test_y_df = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')

    __Lcat = train_x_df.dtypes[train_x_df.dtypes == 'object'].index
    # __Lnum = train_x_df.dtypes[train_x_df.dtypes != 'object'].index


    for col in __Lcat:
        exp_ = np.exp(-train_x_df[col].nunique() * 0.05)
        __K[col] = np.int(5 * (1 - exp_) + 1)

    for col in __Lcat:
        d = dict()
        levels = list(train_x_df[col].unique())
        nan = False


        if np.NaN in levels:
            nan = True
            levels.remove(np.NaN)

        for enc, level in enumerate([np.NaN] * nan + sorted(levels)):
            d[level] = enc

        __Enc[col] = d

        var = Input(shape=(1,))
        inputs.append(var)

        emb = Embedding(input_dim=len(__Enc[col]),
                        output_dim=__K[col],
                        input_length=1)(var)
        emb = Reshape(target_shape=(__K[col],))(emb)

        embeddings.append(emb)

    if (len(__Lcat) > 1):
        emb_layer = concatenate(embeddings)
    else:
        emb_layer = embeddings[0]



    # Add embedding layer as input layer
    outputs = emb_layer

    # Add each dense layer
    for i in range(nb_layers):

        outputs = Dense(nb_neurons, kernel_initializer='uniform', activation=activation)(outputs)

        # Add dropout for all layers after the embedding layer
        if i > 0:
            outputs = Dropout(dropout)(outputs)


    # Add final linear output layer.
    outputs = Dense(1, kernel_initializer='normal', activation='linear')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # Learning the weights
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, verbose=1, patience=4)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    csv_logger = CSVLogger('./logs/entity_embedding.log')

    train_x = [train_x_df[col].apply(lambda x: __Enc[col][x]).values for col in __Lcat]
    test_x = [test_x_df[col].apply(lambda x: __Enc[col][x]).values for col in __Lcat]

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])


    history = model.fit(train_x, train_y_df.values,
        validation_data=(test_x, test_y_df.values),
        epochs=500,
        verbose=0,
        batch_size=batch_size,
        callbacks=[early_stopping, csv_logger],
        )

    print('\rResults')

    hist_epochs = len(history.history['val_loss'])

def assess_models():

    df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_train_y = pd.read_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
    df_all_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    df_all_test_y = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
    df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_x = df_all_train_x.values
    test_actuals = df_all_test_actuals.values
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)
    test_x = df_all_test_x.values


    print('Fitting Keras mape model...')

    network = {
        'nb_neurons': 512,
        'nb_layers': 3,
        'activation': "relu",
        'optimizer': "adagrad",
        'batch_size': 256,
        'dropout': 0.05,
        'model_type': "mape"
    }

    input_shape = (train_x.shape[1],)

    p_model = compile_keras_model(network, input_shape)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    csv_logger = CSVLogger('./logs/actual-mape-training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    history = p_model.fit(train_x,
                          train_actuals,
                          validation_data=(test_x, test_actuals),
                          epochs=20000,
                          batch_size=network['batch_size'],
                          callbacks=[reduce_lr, early_stopping, csv_logger, checkpointer],
                          verbose=0)

    p_model.load_weights('weights.hdf5')

    predictions = p_model.predict(test_x)

    eval_results({'keras_mape': {
                        'actual_y': test_actuals,
                        'y_predict': predictions
                }
    })


    print('Building Keras mae model...')

    network = {
        'nb_layers': 4,
        'nb_neurons': 768,
        'activation': "relu",
        'optimizer': "adamax",
        'dropout': 0.05,
        'batch_size': 256,
        'model_type': "mae",
        'int_layer': 30
    }

    input_shape = (train_x.shape[1],)

    model = compile_keras_model(network, input_shape)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=26)
    csv_logger = CSVLogger('./logs/log-training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

    print('Fitting Keras mae model...')

    history = model.fit(train_x,
                        train_y,
                        validation_data=(test_x, test_y),
                        epochs=20000,
                        batch_size=network['batch_size'],
                        callbacks=[reduce_lr, early_stopping, checkpointer, csv_logger],
                        verbose=0)

    model.load_weights('weights.hdf5')


    print('Executing keras predictions...')

    log_y_predictions = model.predict(test_x)
    exp_predictions = safe_exp(log_y_predictions)

    eval_results({'keras_log_y': {
                        #'log_y': test_y,
                        'actual_y': test_actuals,
                        #'log_y_predict': log_predictions,
                        'y_predict': exp_predictions
                }
    })

    range_results({
        'keras_mape': predictions,
        'keras_log_y': exp_predictions,
    }, test_actuals)


    # save models
    p_model.save('models/mape-model.h5')
    model.save('models/mae-model.h5')


def main():
    # network = {
    #     "nb_neurons": 1024,
    #     "nb_layers": 4,
    #     "activation": "relu",
    #     "optimizer": "adadelta",
    #     "batch_size": 512,
    #     "dropout": 0.5,
    # }
    #
    # train_and_score(network)

    # network = {
    #     "nb_neurons": 768,
    #     "nb_layers": 5,
    #     "activation": "elu",
    #     "optimizer": "adamax",
    #     "batch_size": 256,
    #     "dropout": 0.05,
    #     "int_layer": 30,
    # }
    #
    # train_and_score(network)
    # assess_models()

    # network = {
    #     "nb_neurons": 768,
    #     "nb_layers": 5,
    #     "activation": "elu",
    #     "optimizer": "adamax",
    #     "batch_size": 256,
    #     "dropout": 0.05,
    #     "int_layer": 30,
    # }
    #
    # train_and_score(network)


    # network = {
    #     'activation': 'PReLU',
    #     'optimizer': 'Nadam',
    #     'batch_size': 1024,
    #     'dropout': 0,
    #     'model_type': 'mae_mape',
    #     'log_y': False,
    #     'kernel_initializer': 'normal',
    #     'hidden_layers': [5],
    # }
    #
    # train_and_score_shallow_bagging(network)

    # network = {
    #     'activation': 'PReLU',
    #     'optimizer': 'Nadam',
    #     'batch_size': 512,
    #     'dropout': 0,
    #     'model_type': 'mae',
    #     'log_y': False,
    #     'kernel_initializer': 'glorot_normal',
    #     'hidden_layers': [1, 1, 1, 1],
    # }

    # network = {
    #     'hidden_layers': [7, 7, 7, 7],
    #     'activation': 'relu',
    #     'optimizer': 'Adamax',
    #     'kernel_initializer': 'normal',
    #     'dropout': 0.05,
    #     'batch_size': 256,
    #     'model_type': 'mae',
    #     'int_layer': 30,
    #     'log_y': True,
    # }
    #
    # train_and_score(network)
    #
    # network = {
    #     'hidden_layers': [7, 7, 7, 7],
    #     'activation': "relu",
    #     'optimizer': "Adamax",
    #     'kernel_initializer': 'normal',
    #     'dropout': 0,
    #     'batch_size': 512,
    #     'model_type': "mae",
    #     'int_layer': 30,
    #     'log_y': False,
    # }
    #
    # train_and_score(network)

    network = {
        'hidden_layers': [5, 5, 5],
        'activation': "relu",
        'optimizer': "Adagrad",
        'kernel_initializer': 'glorot_uniform',
        'batch_size': 256,
        'dropout': 0.05,
        'model_type': "mape",
        'log_y': False,
    }

    train_and_score(network)


    network = {
        'hidden_layers': [5, 5, 5, 5],
        'int_layer': 30,
        'activation': "PReLU",
        'optimizer': "Adamax",
        'kernel_initializer': 'he_uniform',
        'batch_size': 512,
        'dropout': 0.0,
        'model_type': "mape",
        'log_y': False,
    }

    train_and_score(network)


if __name__ == '__main__':
    main()
