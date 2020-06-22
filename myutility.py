import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

import golois


def load_data(is_dynamic, data_size, planes, moves):
    if is_dynamic:
        input_data = np.random.randint(2, size=(data_size, 19, 19, planes))
        input_data = input_data.astype('float32')

        policy = np.random.randint(moves, size=(data_size,))
        policy = keras.utils.to_categorical(policy, moves)

        value = np.random.randint(2, size=(data_size,))
        value = value.astype('float32')

        end = np.random.randint(2, size=(data_size, 19, 19, 2))
        end = end.astype('float32')

        golois.getBatch(input_data, policy, value, end)
    else:
        input_data = np.load('input_data.npy')
        policy = np.load('policy.npy')
        value = np.load('value.npy')
        end = np.load('end.npy')
    return input_data, policy, value, end


def build_input_layer(board, filters, deep):
    input_layer = board
    for i in range(deep):
        left = layers.Conv2D(filters, 3, activation='relu', padding='same')(input_layer)
        right = layers.Conv2D(filters, 1, activation='relu', padding='same')(input_layer)
        input_layer = layers.add([left, right])
        input_layer = layers.Activation('relu')(input_layer)
    return input_layer


def build_policy_layer(input_layer, filters, deep, moves):
    policy_layer = input_layer
    for i in range(deep):
        left = layers.Conv2D(filters, 3, activation='relu', padding='same')(policy_layer)
        left = layers.Conv2D(filters, 3, padding='same')(left)
        policy_layer = layers.add([left, policy_layer])
        policy_layer = layers.Activation('relu')(policy_layer)
        policy_layer = layers.BatchNormalization()(policy_layer)

    policy_layer = layers.Conv2D(24, 3, activation='relu', padding='same')(policy_layer)
    policy_layer = layers.MaxPooling2D()(policy_layer)
    policy_layer = keras.layers.Flatten()(policy_layer)
    policy_layer = layers.Dense(moves, activation='softmax', name='policy')(policy_layer)
    return policy_layer


def build_value_layer(input_layer):
    value_layer = input_layer
    value_layer = layers.Conv2D(30, 3, activation='relu', padding='same')(value_layer)
    value_layer = layers.MaxPooling2D()(value_layer)
    value_layer = layers.Flatten()(value_layer)
    value_layer = layers.Dense(1, activation='sigmoid', name='value')(value_layer)
    return value_layer

