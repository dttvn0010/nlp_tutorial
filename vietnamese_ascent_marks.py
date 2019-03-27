import numpy as np
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout, RepeatVector, Concatenate, Embedding
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from random import random, shuffle

import unicodedata
import os
import re


special_char_regex = "[\xad\u2008\xa0\u200b~!@#$%^&\+={}\[\]\\|/<>?“”\"‘’`…'_°*²ð]"
stop_chars = '.,:;()–-'

def clean_text(text):
    for ch in stop_chars:
        text = text.replace(ch, ' ' + ch + ' ')    
    words = text.split(' ')
    words = [word for word in words if word != '']
    text = re.sub(special_char_regex, '', ' '.join(words))    
    return text


def normalize(text):
    return ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))

topics = ['xahoi' , 'kinhdoanh', 'thethao', 'vanhoa']
topic_names = ['Xã hội', 'Kinh doanh', 'Thể thao', 'Văn hóa']


max_length = 22
train_docs = []
test_docs = []
input_words = set()

for i in range(len(topics)):
    fn = os.path.join('data/titles', topics[i] + '.txt')
    f = open(fn, encoding='utf8')
    lines = f.readlines()
    for line in lines[:7000]:
        text = clean_text(line.strip().lower())
        words = text.split(' ')
        words_norm = [normalize(word) for word in words]
        if len(words) <= max_length-2:
            train_docs.append((words, words_norm))
        for word in words:
            input_words.add(word)
        for word_norm in words_norm:
            input_words.add(word_norm)
    #
    for line in lines[7500:10000] :
        text = clean_text(line.strip().lower())
        words = text.split(' ')
        words_norm = [normalize(word) for word in words]
        if len(words) <= max_length-2:
            test_docs.append((words, words_norm))            
    f.close()

input_word_indexes = {w : i+3 for i, w in enumerate(list(input_words)) }
input_word_indexes['<UNK>'] = 0
input_word_indexes['<START_SEQ>'] = 1
input_word_indexes['<END_SEQ>'] = 2
word_map = {i:w for w,i in input_word_indexes.items()}
n_words = len(input_word_indexes)

with open('word_list.txt', 'w') as f:
    for w, i in input_word_indexes.items():
        f.write('{}~{}\n'.format(w, i))

def encode(doc):
    words, words_norm = doc
    x = np.zeros(max_length, dtype='int32')
    y = np.zeros(max_length, dtype='int32')
    n = len(words)
    x[0] = y[0] = 1    
    for i in range(n):
        y[i+1] = input_word_indexes.get(words[i], 0)        
        if random() < 0.2:
            x[i+1] = input_word_indexes.get(words_norm[i], 0)             
        else:
            x[i+1] = input_word_indexes.get(words[i], 0) 
    x[n+1] = y[n+1] = 2    
    return x, np.array([to_categorical(yi, n_words) for yi in y])
    
class DataGenerator():
    def __init__(self, docs, batch_size):
        self.docs = docs
        self.batch_size = batch_size
        self.current_index = 0
        self.N = len(docs)
    #
    def next_batch(self):
        Xb = np.zeros((self.batch_size, max_length), dtype=np.float32)
        Yb = np.zeros((self.batch_size, max_length, n_words), dtype=np.float32)
        while True:            
            for i in range(self.batch_size):
                x, y = encode(self.docs[self.current_index])
                Xb[i] = x
                Yb[i] = y
                #
                self.current_index += 1
                if self.current_index >= self.N:
                    shuffle(self.docs)
                    self.current_index = 0                
            yield Xb, Yb        
        
def decode_sequence(seq):
    indexes = [np.argmax(x) for x in seq]
    return ' '.join([word_map[index] for index in indexes])
    
           
embeding_vector_length = 64

model = Sequential()
model.add(Embedding(n_words, embeding_vector_length, input_length=max_length))
model.add(Bidirectional(LSTM(50, return_sequences=True, input_shape=(max_length, embeding_vector_length))))
model.add(TimeDistributed(Dense(n_words, activation='softmax')))
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=adam)

model.summary()

batch_size = 256
train_generator = DataGenerator(train_docs, batch_size)
test_generator = DataGenerator(test_docs, batch_size)

early_stopping = EarlyStopping(monitor='val_acc', patience=5)
filepath = 'model-{epoch:03d}-loss{loss:.3f}-val_acc{val_acc:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks = [early_stopping , checkpoint ]

model.fit_generator(generator=train_generator.next_batch(), 
                    steps_per_epoch=5*len(train_docs)//batch_size, 
                    epochs=20,
                    validation_data=test_generator.next_batch(),
                    validation_steps=len(test_docs)//batch_size,
                    callbacks=callbacks, verbose=True)


Xtrain = []
Ytrain =[]

for i in range(5000):
    x,y = encode(train_docs[i])
    Xtrain.append(x)
    Ytrain.append(y)
    
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
    
model.fit(Xtrain, Ytrain, epochs=1, batch_size=128, verbose=True)
