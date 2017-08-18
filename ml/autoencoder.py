from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import gc

def reduce_dimensions(x_train, x_test):
    ae_data = np.row_stack([x_train, x_test])

    n_layer1 = int(ae_data.shape[1])
    n_layer2 = 256
    n_layer3 = 128
    n_layer4 = 64
    n_layer5 = 32

    print('Building Keras autoencoder model for', n_layer1 , 'features')

    model = Sequential()
    model.add(Dense(n_layer1, input_shape=(n_layer1,)))
    model.add(Activation('selu'))
    model.add(Dense(n_layer2))
    model.add(Activation('selu'))
    model.add(Dense(n_layer3))
    model.add(Activation('selu'))
    model.add(Dense(n_layer4, name="encoded"))
    model.add(Activation('selu'))
    # model.add(Dense(n_layer5, name="encoded"))
    # model.add(Activation('selu'))
    # model.add(Dense(n_layer4))
    # model.add(Activation('selu'))
    model.add(Dense(n_layer3))
    model.add(Activation('selu'))
    model.add(Dense(n_layer2))
    model.add(Activation('selu'))
    model.add(Dense(n_layer1))
    model.add(Activation('selu'))

    model.compile(optimizer='adam', loss='mae')

    print('Fitting Keras autoencoder model...')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, verbose=1, patience=10)
    early_stopping = EarlyStopping(monitor='loss', patience=50)

    history = model.fit(ae_data,
                        ae_data,
                        epochs=10000,
                        batch_size=512,
                        callbacks=[reduce_lr, early_stopping],
                        verbose=1)

    gc.collect()

    plt.plot(history.history['loss'])
    plt.title('Keras autoencoder model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ion()
    plt.show()

    # Make intermediate model which outputs the encoding stage results
    encoder_model = Model(inputs=model.input, outputs=model.get_layer('encoded').output)


    return_x_train = encoder_model.predict(x_train)
    return_x_test = encoder_model.predict(x_test)

    gc.collect()

    return return_x_train, return_x_test