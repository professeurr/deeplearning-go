import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from myutility import load_data, build_hidden_layer, build_input_layer

# input
planes = 8
moves = 361
dynamic_batch = True

board = keras.Input(shape=(19, 19, planes), name='board')
input_layer = board
for i in range(3):
    input_layer = build_input_layer(input_layer)

policy_layer = input_layer
for i in range(6):
    policy_layer = build_hidden_layer(policy_layer)
policy_layer = keras.layers.Flatten()(policy_layer)
policy_layer = layers.Dense(moves, activation='softmax', name='policy')(policy_layer)

value_layer = layers.Flatten()(input_layer)
value_layer = layers.Dense(1, activation='sigmoid', name='value')(value_layer)

model = keras.Model(inputs=board, outputs=[policy_layer, value_layer])
plot_model(model, to_file='model.png')

# input
N = 100000
sample_size = int(5000 / 0.9)
nb_sample = int(N / sample_size)
epochs = 50
mini_epochs = 1
mini_batch_size = 50
step = 0
model.compile(optimizer='nadam',
              loss={'value': 'mse', 'policy': 'categorical_crossentropy'}, metrics=['accuracy'])

file_name = 'metrics_{}.csv'.format(epochs)
output_file = open(file_name, 'w')
output_file.write('train_policy_loss,val_policy_loss,train_policy_accuracy,val_policy_accuracy\n')
output_file.close()

for epoch in range(epochs):
    print('running batch', (epoch + 1), '/', epochs, '...')
    # load data
    input_data, policy_data, value_data, end = load_data(dynamic_batch, N, planes, moves)
    for k in range(nb_sample):
        print('epoch:', (epoch + 1), '/', epochs, '- sample:', (k + 1), '/', nb_sample)
        start = k * sample_size
        stop = (k + 1) * sample_size
        # fitting the model with the current dataset
        result = model.fit(np.take(input_data, range(start, stop), axis=0),
                           {
                               'policy': np.take(policy_data, range(start, stop), axis=0),
                               'value': np.take(value_data, range(start, stop), axis=0)
                           },
                           epochs=mini_epochs, batch_size=mini_batch_size, validation_split=0.1, verbose=1)

        output_file = open(file_name, 'a')
        output_file.write('{},{},{},{}\n'.format(result.history['policy_loss'][-1], result.history['val_policy_loss'][-1],
                                                 result.history['policy_accuracy'][-1],
                                                 result.history['val_policy_accuracy'][-1]))
        output_file.close()

    model.save('klouvi_kodjo_model.h5')

print('Done')


data = pd.read_csv(file_name)
data.plot(y=['train_policy_loss', 'val_policy_loss', 'train_policy_accuracy', 'val_policy_accuracy'],
          style=['r', 'b', 'g', 'o'])
