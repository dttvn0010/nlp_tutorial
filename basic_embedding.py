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

train_docs = []
train_labels = []

test_docs = []
test_labels = []

word_count = {}

for i in range(len(topics)):
    fn = os.path.join('data/titles', topics[i] + '.txt')
    f = open(fn, encoding='utf8')
    lines = f.readlines()
    #
    for line in lines[:5000]:
        tokens = word_tokenize(line.strip())
        if ':' in tokens:
            continue
        train_docs.append(tokens)
        train_labels.append(i)    
        for token in set(tokens):
            word_count[token] = word_count.get(token, 0) + 1        
    #
    for line in lines[5000:10000]:
        tokens = word_tokenize(line.strip())
        if ':' in tokens:
            continue
        test_docs.append(tokens)
        test_labels.append(i)
    f.close()

    
word_items = word_count.items()
word_items = sorted(word_items, key=lambda x : x[1], reverse=True)

word_index = {item[0]:i+3 for i,item in enumerate(word_items[:10000])}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

def encode_tokens(tokens):    
    return [word_index.get(token, 0) for token in tokens]

maxlen = 24
Xtrain = [encode_tokens(doc) for doc in train_docs]
Xtrain = pad_sequences(Xtrain, value=0, maxlen=maxlen)
ytrain = np.array([to_categorical(label, num_classes) for label in train_labels])

Xtest = [encode_tokens(doc) for doc in test_docs]
Xtest = pad_sequences(Xtest, value=0, maxlen=maxlen)
ytest = np.array([to_categorical(label, num_classes) for label in test_labels])


vocab_size = len(word_index)
embeding_vector_length = 32

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embeding_vector_length, input_length=maxlen))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(Xtrain, ytrain, epochs=10, batch_size=64, verbose=1)

model.evaluate(Xtest, ytest)
