import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from myutility import load_data, build_policy_layer, build_input_layer, build_value_layer

# input
planes = 8
moves = 361
dynamic_batch = True
filters = 30

board = keras.Input(shape=(19, 19, planes), name='board')
input_layer = build_input_layer(board, filters, 6)
policy_layer = build_policy_layer(input_layer, filters, 14, moves)
value_layer = build_value_layer(input_layer)

model = keras.Model(inputs=board, outputs=[policy_layer, value_layer])
plot_model(model, to_file='model.png')

print(model.summary())

# exit(0)

# input
N = 100000
sample_size = int(N / 0.9)
nb_sample = int(5000 / sample_size)
epochs = 100
mini_epochs = 3
mini_batch_size = 128
step = 0
model.compile(optimizer='nadam',
              loss={'value': 'mse', 'policy': 'categorical_crossentropy'}, metrics=['accuracy'])

file_name = 'metrics.csv'
with open(file_name, 'w') as output_file:
    output_file.write('train_policy_loss,val_policy_loss,train_policy_accuracy,val_policy_accuracy\n')

for epoch in range(epochs):
    print('running batch', (epoch + 1), '/', epochs, '...')
    # load data
    input_data, policy_data, value_data, end = load_data(dynamic_batch, N, planes, moves)
    # fitting the model with the current dataset
    result = model.fit(input_data, {'policy': policy_data, 'value': value_data},
                       epochs=mini_epochs, batch_size=mini_batch_size,
                       validation_split=0.1, verbose=1)

    with open(file_name, 'a') as output_file:
        for i in range(len(result.history['policy_loss'])):
            output_file.write(
                '{},{},{},{}\n'.format(result.history['policy_loss'][i], result.history['val_policy_loss'][i],
                                       result.history['policy_accuracy'][i],
                                       result.history['val_policy_accuracy'][i]))

    model.save('klouvi_riva_model.h5')

print('Done')

# data = pd.read_csv(file_name)
# data.plot(x=range(data.size), y=['train_policy_loss', 'val_policy_loss', 'train_policy_accuracy', 'val_policy_accuracy'])
# plt.show()
