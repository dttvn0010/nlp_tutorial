import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Bidirectional
from random import random

def get_random_bit():
    return 1 if random() >= 0.5 else 0
    
def and_bit(x, y):
    return 1 if x == 1 and y == 1 else 0

length = 20

X = []
y = []

for _ in range(10000):
    in_seq = [get_random_bit() for _ in range(length)]
    out_seq = [in_seq[0]] + [and_bit(in_seq[i-1], in_seq[i+1]) for i in range(1, length-1)] + [in_seq[-1]]
    X.append(np.array(in_seq).reshape(length, 1))
    y.append(np.array(out_seq).reshape(length, 1))

X = np.array(X)
y = np.array(y)

n_neurons = 2
n_batch = 32
n_epoch = 1000
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True))
#model.add(Bidirectional(LSTM(n_neurons, return_sequences=True), input_shape=(length, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
print(model.summary())

model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)

### test

X2 = []
y2 = []

for _ in range(1000):
    in_seq = [get_random_bit() for _ in range(length)]
    out_seq = [in_seq[0]] + [and_bit(in_seq[i-1], in_seq[i+1]) for i in range(1, length-1)] + [in_seq[-1]]
    X2.append(np.array(in_seq).reshape(length, 1))
    y2.append(np.array(out_seq).reshape(length, 1))

X2 = np.array(X2)
y2 = np.array(y2)

model.evaluate(X2, y2)
