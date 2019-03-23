from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU, ThresholdedReLU, ELU
from keras import optimizers
from AdamW import AdamW
from stats_operations import k_mae_mape, k_mean_absolute_percentage_error
from print_logger import print


def get_activation_layer(activation):
    if activation == 'LeakyReLU':
        return LeakyReLU()
    if activation == 'PReLU':
        return PReLU()
    if activation == 'ELU':
        return ELU()
    if activation == 'ThresholdedReLU':
        return ThresholdedReLU()

    return Activation(activation)


def get_optimizer(name='Adadelta', momentum=None, min_lr=0.0001, weight_decay=0.):
    if name == 'SGD':
        # return optimizers.SGD()
        return optimizers.SGD(lr=min_lr, momentum=momentum, decay=weight_decay, nesterov=True)
    if name == 'AdamW':
        return AdamW(weight_decay=weight_decay)
    if name == 'RMSprop':
        return optimizers.RMSprop()  # clipnorm=1.)
    if name == 'Adagrad':
        return optimizers.Adagrad()  # clipnorm=1.)
    if name == 'Adadelta':
        return optimizers.Adadelta()  # clipnorm=1.)
    if name == 'Adam':
        return optimizers.Adam()  # clipnorm=1.)
    if name == 'Adamax':
        return optimizers.Adamax()  # clipnorm=1.)
    if name == 'Nadam':
        return optimizers.Nadam()  # clipnorm=1.)

    return optimizers.Adam()  # clipnorm=1.)


def compile_keras_model(network, dimensions):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    activation = network.get('activation', 'relu')
    optimizer = network.get('optimizer', 'Adam')
    dropout = network.get('dropout', 0)
    kernel_initializer = network.get('kernel_initializer')
    model_type = network.get('model_type', 'mae')
    momentum = network.get('momentum')
    min_lr = network.get('min_lr', 0.0001)
    max_lr = network.get('max_lr', 0.1)
    weight_decay = network.get('weight_decay', 0.)
    num_classes = 0

    if model_type == "categorical_crossentropy":
        num_classes = network['num_classes']

    model = Sequential()

    # The hidden_layers passed to us is simply describing a shape.
    # it does not know the num_cols we are dealing with, it is simply
    # values of 0.5, 1, and 2, which need to be multiplied by the num_cols
    scaled_layers = []
    for layer in network['hidden_layers']:
        scaled_layers.append(max(int(dimensions * layer), 1))

    print('scaled_layers', scaled_layers)

    # Add input layers
    model.add(Dense(
        scaled_layers[0], kernel_initializer=kernel_initializer, input_dim=dimensions))
    model.add(get_activation_layer(activation))
    model.add(Dropout(dropout))

    # Add hidden layers
    for layer_size in scaled_layers[1:-1]:
        model.add(Dense(layer_size, kernel_initializer=kernel_initializer))
        model.add(get_activation_layer(activation))
        model.add(Dropout(dropout))

    if 'int_layer' in network:
        model.add(Dense(network['int_layer'], name="int_layer",
                        kernel_initializer=kernel_initializer))
        model.add(get_activation_layer(activation))
        model.add(Dropout(dropout))

    # Output layer.
    if model_type == "categorical_crossentropy":
        model.add(
            Dense(num_classes, kernel_initializer=kernel_initializer, activation='softmax'))
        model.compile(loss="categorical_crossentropy",
                      optimizer=get_optimizer(
                          optimizer, momentum, min_lr, weight_decay),
                      metrics=["categorical_accuracy"])
    else:
        model.add(
            Dense(1, kernel_initializer=kernel_initializer, activation='linear'))

        if model_type == "mape":
            model.compile(loss=k_mean_absolute_percentage_error,
                          optimizer=get_optimizer(
                              optimizer, momentum, min_lr, weight_decay),
                          metrics=['mae'])
        elif model_type == "mae_mape":
            model.compile(loss=k_mae_mape,
                          optimizer=get_optimizer(
                              optimizer, momentum, min_lr, weight_decay),
                          metrics=['mae', k_mean_absolute_percentage_error])
        else:
            model.compile(loss='mae',
                          optimizer=get_optimizer(
                              optimizer, momentum, min_lr, weight_decay),
                          metrics=[k_mean_absolute_percentage_error])

    return model
