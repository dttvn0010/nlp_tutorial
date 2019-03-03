import os
import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from random import shuffle
from pyvi import ViTokenizer, ViPosTagger

from gensim.models import Word2Vec

wv_model = Word2Vec.load('vi.bin')

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
        

maxlen = 128
embedding_vecor_length = 100 
	
def encode_tokens(tokens):    
    sequence = []
    #    
    if len(tokens) < maxlen:
        sequence = [np.zeros(embedding_vecor_length)] * (maxlen - len(tokens))
    #
    for token in tokens[:maxlen]:
        if token in wv_model:
            sequence.append(wv_model[token])
        else:
            sequence.append(np.zeros(embedding_vecor_length))
    #
    return np.array(sequence)
   

Xtrain = np.array([encode_tokens(x[0]) for x in train_data])
ytrain = np.array([to_categorical(x[1], num_classes) for x in train_data])

Xtest = np.array([encode_tokens(x[0]) for x in test_data])
ytest = np.array([to_categorical(x[1], num_classes) for x in test_data])


model = keras.Sequential()
model.add(keras.layers.LSTM(100, input_shape=(maxlen, embedding_vecor_length), dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(Xtrain, ytrain, epochs=8, batch_size=64, verbose=1)

model.evaluate(Xtest, ytest)