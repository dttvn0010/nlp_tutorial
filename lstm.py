import os
import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from random import shuffle
from pyvi import ViTokenizer, ViPosTagger

special_char = [chr(c + ord('0')) for c in range(10)]
special_char.extend([' ', '~', '!', '@', '#', '$', '%', '^', '&', '-', '+', '=', 
                     '{', '}', '[', ']', '\\', '|', '/', '<', '>', '?', '“', '”', '"',
                    '‘', '’'])

def is_valid_word(word):
    return all(c not in word for c in special_char)

def word_tokenize(sentence):
    words, postags = ViPosTagger.postagging(ViTokenizer.tokenize(sentence.lower()))
    return [word for word in words if is_valid_word(word)]
	

topics = ['xahoi' , 'kinhdoanh', 'thethao', 'vanhoa']
topic_names = ['Xã hội', 'Kinh doanh', 'Thể thao', 'Văn hóa']

num_classes = len(topics)

word_doc_counts = {}

train_data = []
test_data = []

for i in range(len(topics)):
    fn = os.path.join('data/headlines', topics[i] + '.txt')
    f = open(fn, encoding='utf8')
    lines = f.readlines()
    #    
    for line in lines[:2000:10]:
        tokens = word_tokenize(line.strip())
        train_data.append((tokens, i))
        #        
        for token in set(tokens):
            word_doc_counts[token] = word_doc_counts.get(token, 0) + 1
            
    for line in lines[5000:7000]:
        tokens = word_tokenize(line.strip())
        test_data.append((tokens, i))
    #    
    f.close()    
	

word_items = list(word_doc_counts.items())
word_items = sorted(word_items, key=lambda x : x[1])


word_index = {item[0]:i+3 for i,item in enumerate(word_items[-10000:])}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

def encode_tokens(tokens):    
    return [word_index.get(token, 0) for token in tokens]
	
Xtrain = [encode_tokens(x[0]) for x in train_data]
Xtrain = pad_sequences(Xtrain, value=0, maxlen=128)
ytrain = np.array([to_categorical(x[1], num_classes) for x in train_data])


Xtest = [encode_tokens(x[0]) for x in test_data]
Xtest = pad_sequences(Xtest, value=0, maxlen=128)
ytest = np.array([to_categorical(x[1], num_classes) for x in test_data])


vocab_size = len(word_index)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_length=128))
model.add(keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(Xtrain, ytrain, epochs=20, batch_size=64, verbose=1)

model.evaluate(Xtest, ytest)