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


def initialize_input_layers(input, z, depth):
    x = input
    for i in range(depth):
        x = layers.Conv2D(30, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        if i < depth - 1:
            x = layers.add([x, z])
    return x


def build_hidden_layers(head, nb_ahead, z, depth):
    for i in range(depth):
        head = layers.Conv2D(30, nb_ahead, activation='relu', padding='same')(head)
        head = layers.BatchNormalization()(head)
        if i < depth - 1:
            head = layers.add([head, z])
    head = layers.MaxPooling2D()(head)
    head = layers.Flatten()(head)
    return head
