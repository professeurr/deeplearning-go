import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

from myutility import load_data, build_hidden_layers, initialize_input_layer

# input
N = 100000
planes = 8
moves = 361
x_layers_depth = 4
hidden_layers_depth = 3
epochs = 8
dynamic_batch = True
global_loop = 100
mini_batch_size = 1000

# output
train_policy_accuracy = []
val_policy_accuracy = []

input = keras.Input(shape=(19, 19, planes), name='boardx')
input = layers.Conv2D(32, 3, activation='relu', padding='same')(input)
output = layers.Flatten()(input)
model = keras.Model(inputs=input, outputs=[output])
model.compile(optimizer='adam', loss={'value': 'mse', 'policy': 'categorical_crossentropy'}, metrics=['accuracy'])
plot_model(model, to_file='model.png')

exit(0)

# initialize the input layer
input, x, z = initialize_input_layers(planes, x_layers_depth)

# build policy network
policy_head = build_hidden_layers(x, 5, z, hidden_layers_depth)
policy_head = layers.Dense(moves, activation='softmax', name='policy')(policy_head)

# build value network
value_head = build_hidden_layers(x, 3, z, hidden_layers_depth)
value_head = layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l1(0.1), name='value')(
    value_head)

# create new model based on the above architecture
model = keras.Model(input, outputs=[policy_head, value_head])
model.compile(optimizer='adam', # keras.optimizers.SGD(lr=0.1),
              loss={'value': 'mse', 'policy': 'categorical_crossentropy'},
              metrics=['accuracy'])
plot_model(model, to_file='model.png')

for k in range(global_loop):
    print('running batch', (k + 1), '/', global_loop, '...')
    # load data
    input_data, policy, value, end = load_data(dynamic_batch, N, planes, moves)
    # fitting the model with the current dataset
    result = model.fit(input_data, {'policy': policy, 'value': value},
                       epochs=epochs, batch_size=mini_batch_size, validation_split=0.1, verbose=1)

    train_policy_accuracy = train_policy_accuracy + result.history['policy_accuracy']
    val_policy_accuracy = val_policy_accuracy + result.history['val_policy_accuracy']
    print('train policy accuracy:', train_policy_accuracy[-1], '- validation policy accuracy:', val_policy_accuracy[-1])

print('Done')

# plot the history of accuracy
plt.plot(train_policy_accuracy)
plt.plot(val_policy_accuracy)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.save('klouvi_kodjo_model.h5')


# for epoch in range(epochs):
#     print('running batch', (epoch + 1), '/', epochs, '...')
#     # load data
#     input_data, policy_data, value_data, end = load_data(dynamic_batch, N, planes, moves)
#     for k in range(nb_sample):
#         print('epoch:', (epoch + 1), '/', epochs, '- sample:', (k + 1), '/', nb_sample)
#         start = k * sample_size
#         stop = (k + 1) * sample_size
#         # fitting the model with the current dataset
#         result = model.fit(np.take(input_data, range(start, stop), axis=0),
#                            {
#                                'policy': np.take(policy_data, range(start, stop), axis=0),
#                                'value': np.take(value_data, range(start, stop), axis=0)
#                            },
#                            epochs=mini_epochs, batch_size=mini_batch_size, validation_split=0.1, verbose=1)
#
#         output_file = open(file_name, 'a')
#         output_file.write('{},{},{},{}\n'.format(result.history['policy_loss'][-1], result.history['val_policy_loss'][-1],
#                                                  result.history['policy_accuracy'][-1],
#                                                  result.history['val_policy_accuracy'][-1]))
#         output_file.close()
#
#     model.save('klouvi_riva_model.h5')