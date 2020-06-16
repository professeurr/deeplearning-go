import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from myutility import load_data, build_hidden_layers, initialize_input_layers


N = 1000
planes = 8
moves = 361
x_layers_depth = 2
hidden_layers_depth = 2
epochs = 7
dynamicBatch = True
batchSize = 4

globalModel = None

for k in range(batchSize):
    print('running batch', (k+1), '/', batchSize)

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

    model.fit(input_data, {'policy': policy, 'value': value},
              epochs=epochs, batch_size=32, validation_split=0.1)

    globalModel = model

plot_model(globalModel, to_file='model.png')

globalModel.save('klouvi_kodjo_model.h5')

# prevModel.summary()

