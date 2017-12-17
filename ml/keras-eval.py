"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, concatenate, Input
from keras.layers.embeddings import Embedding
from keras.models import Model

from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from eval_results import *

def k_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1., None))
    return 100. * K.mean(diff, axis=-1)

def k_mae_mape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1., None))
    mape = 100. * K.mean(diff, axis=-1)
    mae = K.mean(K.abs(y_true - y_pred), axis=-1)
    return mape * mae

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

def compile_model(network, input_shape, model_type=''):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout = network['dropout']

    if 'model_type' in network:
        model_type = network['model_type']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(dropout))


    if 'int_layer' in network:
        model.add(Dense(network['int_layer'], activation=activation, name="int_layer"))
        model.add(Dropout(dropout))


    # Output layer.
    model.add(Dense(1, activation='linear'))

    if model_type == "mape":
        model.compile(loss=k_mean_absolute_percentage_error, optimizer=optimizer, metrics=['mae'])
    elif model_type == "mae_mape":
        model.compile(loss=k_mae_mape, optimizer=optimizer, metrics=['mae', k_mean_absolute_percentage_error])
    else:
        model.compile(loss='mae', optimizer=optimizer, metrics=[k_mean_absolute_percentage_error])

    return model

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


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=10)
    early_stopping = EarlyStopping(monitor='val_loss', patience=36)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=0, save_best_only=True)

    input_shape = (train_x.shape[1],)


    model = compile_model(network, input_shape, "mae_mape")

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])

    history = model.fit(train_x, train_actuals,
                        batch_size=network['batch_size'],
                        epochs=10000,  # using early stopping, so no real limit
                        verbose=0,
                        validation_data=(test_x, test_actuals),
                        callbacks=[early_stopping, csv_logger, reduce_lr, checkpointer])

    model.load_weights('weights.hdf5')
    predictions = model.predict(test_x)
    predictions = predictions.reshape(predictions.shape[0], )
    # predictions = safe_exp(predictions)
    mae = mean_absolute_error(test_actuals, predictions)
    mape = safe_mape(test_actuals, predictions)
    maeape = mae_mape(test_actuals, predictions)

    score = maeape

    print('\rResults')

    hist_epochs = len(history.history['val_loss'])


    print('epochs:', hist_epochs)
    print('mae_mape:', maeape)
    print('mape:', mape)
    print('mae:', mae)
    print('-' * 20)

    eval_results({'bagged_predictions': {
                        'actual_y': test_actuals,
                        'y_predict': predictions
                }
    })

    range_results({
        'bagged_predictions': predictions,
    }, test_actuals)


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


    model = compile_model(network, input_shape, "mape")

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

    # Apply value scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    csv_logger = CSVLogger('./logs/training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    input_shape = (train_x_scaled.shape[1],)


    model = compile_model(network, input_shape, network['model_type'])

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])

    # history = model.fit(train_x, train_y,
    history = model.fit(train_x_scaled, train_eval_y,
                        batch_size=network['batch_size'],
                        epochs= network['epochs'],
                        verbose=0,
                        # validation_data=(test_x, test_y),
                        validation_data=(test_x_scaled, test_eval_y),
                        callbacks=[early_stopping, csv_logger, reduce_lr, checkpointer])


    print('\rResults')

    hist_epochs = len(history.history['val_loss'])
    # score = history.history['val_loss'][hist_epochs - 1]

    model.load_weights('weights.hdf5')
    predictions = model.predict(test_x_scaled)
    prediction_results = predictions.reshape(predictions.shape[0],)

    # If using log of y, get exponent
    if network['log_y']:
        prediction_results = safe_exp(prediction_results)


    mae = mean_absolute_error(test_y, prediction_results)
    mape = safe_mape(test_y, prediction_results)
    maeape = mae_mape(test_y, prediction_results)

    score = mape

    print('\rResults')

    hist_epochs = len(history.history['val_loss'])

    if np.isnan(score):
        score = 9999

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

    p_model = compile_model(network, input_shape)

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

    model = compile_model(network, input_shape)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=8)
    early_stopping = EarlyStopping(monitor='val_loss', patience=26)
    csv_logger = CSVLogger('./logs/log-training.log')
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

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

    network = {
        'nb_neurons': 16,
        'nb_layers': 2,
        'activation': 'selu',
        'optimizer': 'adagrad',
        'batch_size': 256,
        'dropout': 0.7,
        'model_type': 'mae',
        'epochs': 10000,
        'log_y': True
    }

    train_and_score_shallow_bagging(network)

if __name__ == '__main__':
    main()
