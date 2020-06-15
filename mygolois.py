import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
import golois

planes = 8
moves = 361
epochs = 30
dynamicBatch = True

if dynamicBatch:
    N = 100000
    input_data = np.random.randint(2, size=(N, 19, 19, planes))
    input_data = input_data.astype('float32')

    policy = np.random.randint(moves, size=(N,))
    policy = keras.utils.to_categorical(policy, moves)

    value = np.random.randint(2, size=(N,))
    value = value.astype('float32')

    end = np.random.randint(2, size=(N, 19, 19, 2))
    end = end.astype('float32')

    golois.getBatch(input_data, policy, value, end)
else:
    input_data = np.load('input_data.npy')
    policy = np.load('policy.npy')
    value = np.load('value.npy')
    end = np.load('end.npy')

input = keras.Input(shape=(19, 19, planes), name='board')
z = layers.Conv2D(30, 1, activation='relu', padding='same')(input)

x = layers.Conv2D(30, 3, activation='relu', padding='same')(input)
x = layers.BatchNormalization()(x)
x = layers.add([x,z])
x = layers.Conv2D(30, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.add([x,z])
x = layers.Conv2D(30, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.add([x,z])
x = layers.Conv2D(30, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

policy_head = layers.Conv2D(30, 5, activation='relu', padding='same')(x)
policy_head = layers.BatchNormalization()(policy_head)
policy_head = layers.add([policy_head,z])
policy_head = layers.Conv2D(30, 5, activation='relu', padding='same')(policy_head)
policy_head = layers.BatchNormalization()(policy_head)
policy_head = layers.add([policy_head,z])
policy_head = layers.Conv2D(29, 5, activation='relu', padding='same')(policy_head)
policy_head = layers.BatchNormalization()(policy_head)
policy_head = layers.MaxPooling2D()(policy_head)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Dense(moves, activation='softmax', name='policy')(policy_head)


value_head = layers.Conv2D(30, 3, activation='relu', padding='same')(x)
value_head = layers.BatchNormalization()(value_head)
value_head = layers.add([value_head,z])
value_head = layers.Conv2D(30, 3, activation='relu', padding='same')(value_head)
value_head = layers.BatchNormalization()(value_head)
value_head = layers.add([value_head,z])
value_head = layers.Conv2D(30, 3, activation='relu', padding='same')(value_head)
value_head = layers.BatchNormalization()(value_head)
value_head = layers.MaxPooling2D()(value_head)
value_head = layers.Flatten()(value_head)
value_head= layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l1(0.01), name = 'value')(value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])

model.summary()

model.compile(optimizer=keras.optimizers.SGD(lr=0.1),
              loss={'value': 'mse', 'policy': 'categorical_crossentropy'},
              metrics=['accuracy'])

model.fit(input_data, {'policy': policy, 'value': value},
          epochs=epochs, batch_size=32, validation_split=0.1)

model.save('klouvi_kodjo_model.h5')
