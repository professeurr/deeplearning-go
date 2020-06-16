import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

from myutility import load_data, build_hidden_layers, initialize_input_layers


N = 1000
planes = 8
moves = 361
x_layers_depth = 4
hidden_layers_depth = 3
epochs = 7
dynamicBatch = True
global_loop = 10
mini_batch_size = 32
val_policy_accuracy = []
test_policy_accuracy = []
globalModel = None

for k in range(global_loop):
    print('running batch', (k+1), '/', global_loop)

    input = keras.Input(shape=(19, 19, planes), name='board')
    z = layers.Conv2D(30, 1, activation='relu', padding='same')(input)

    # load data
    input_data, policy, value, end = load_data(True, N, planes, moves)

    # initialize the input layer
    x = initialize_input_layers(input, z, x_layers_depth)

    # build policy network
    policy_head = build_hidden_layers(x, 5, z, hidden_layers_depth)
    policy_head = layers.Dense(moves, activation='softmax', name='policy')(policy_head)

    # build value network
    value_head = build_hidden_layers(x, 3, z, hidden_layers_depth)
    value_head= layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l1(0.1), name = 'value')(value_head)

    model = keras.Model(input, outputs=[policy_head, value_head])
    if globalModel:
        model.set_weights(globalModel.get_weights())

    model.compile(optimizer='adam',
                  loss={'value': 'mse', 'policy': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    result = model.fit(input_data, {'policy': policy, 'value': value},
              epochs=epochs, batch_size=mini_batch_size, validation_split=0.1, verbose=0)

    globalModel = model

    test_policy_accuracy = test_policy_accuracy + result.history['test_policy_accuracy']
    val_policy_accuracy = val_policy_accuracy + result.history['val_policy_accuracy']
    print('validation policy accuracy:', val_policy_accuracy.tail)
    print('test policy accuracy:', test_policy_accuracy.tail)


# prevModel.summary()

# plot the history of accuracy
plt.plot(test_policy_accuracy)
plt.plot(val_policy_accuracy)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plot_model(globalModel, to_file='model.png')
globalModel.save('klouvi_kodjo_model.h5')


