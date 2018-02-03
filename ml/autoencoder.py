from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pandas as pd

def build_encoder(x_data):
    n_layer1 = int(x_data.shape[1])
    n_layer2 = int(x_data.shape[1] / 2)
    n_layer3 = int(x_data.shape[1] / 4)
    # n_layer4 = int(ae_data.shape[1] / 8)

    print('Building Keras autoencoder model for', n_layer1 , 'features')

    model = Sequential()
    model.add(Dense(n_layer1, input_shape=(n_layer1,)))
    model.add(Activation('selu'))
    model.add(Dense(n_layer2))
    model.add(Activation('selu'))
    model.add(Dense(n_layer3, name="encoded"))
    model.add(Activation('selu'))
    # model.add(Dense(n_layer4, name="encoded"))
    # model.add(Activation('selu'))
    # model.add(Dense(n_layer3))
    # model.add(Activation('selu'))
    # model.add(Dense(n_layer2))
    model.add(Activation('selu'))
    model.add(Dense(n_layer1))
    model.add(Activation('selu'))

    model.compile(optimizer='adam', loss='mae')

    print('Fitting Keras autoencoder model...')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, verbose=1, patience=10)
    early_stopping = EarlyStopping(monitor='loss', patience=25)
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    history = model.fit(x_data,
                        x_data,
                        epochs=10000,
                        batch_size=512,
                        callbacks=[reduce_lr, early_stopping, checkpointer],
                        verbose=1)

    model.load_weights('weights.hdf5')


    # Make intermediate model which outputs the encoding stage results
    encoder_model = Model(inputs=model.input, outputs=model.get_layer('encoded').output)

    return encoder_model

def reduce_dimensions(encoder_model, x_data):

    return_x_data = encoder_model.predict(x_data)

    return return_x_data

def main():
    df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    all_x = np.row_stack([df_all_train_x.values, df_all_test_x.values])

    encoder_model = build_encoder(all_x)
    encoder_model.save('models/keras-encoder-model.h5')

    reduced_train_x = reduce_dimensions(encoder_model, df_all_train_x.values)
    reduced_test_x = reduce_dimensions(encoder_model, df_all_test_x.values)

    pd_reduced_train_x = pd.DataFrame(reduced_train_x)
    pd_reduced_test_x = pd.DataFrame(reduced_test_x)

    pd_reduced_train_x.to_pickle('data/ae_train_x.pkl.gz', compression='gzip')
    pd_reduced_test_x.to_pickle('data/ae_test_x.pkl.gz', compression='gzip')


if __name__ == "__main__":
    main()