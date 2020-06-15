import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
import golois

planes = 8
moves = 361
N = 100000

print ('Generating random data')
input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')

policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)

value = np.random.randint(2, size=(N,))
value = value.astype ('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')

print ('Getting batch of examples')
golois.getBatch (input_data, policy, value, end)

print ('Saving numpy arrays')
np.save ('input_data.npy', input_data, allow_pickle=False)
np.save ('policy.npy', policy, allow_pickle=False)
np.save ('value.npy', value, allow_pickle=False)
np.save ('end.npy', end, allow_pickle=False)
