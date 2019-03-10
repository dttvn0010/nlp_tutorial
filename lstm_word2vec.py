import os
import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from random import shuffle
from pyvi import ViTokenizer, ViPosTagger

from gensim.models import Word2Vec
import re

special_char_regex = '.*[0-9~!@#$%^&\-\+={}\[\]\\|/<>?“”"‘’].*'

def is_valid_word(word):
    return re.match(special_char_regex, word) == None

def word_tokenize(sentence):
    words, postags = ViPosTagger.postagging(ViTokenizer.tokenize(sentence.lower()))
    return [word for word in words if is_valid_word(word)]
    
wv_model = Word2Vec.load('vi.bin')
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
        train_labels.append(1)
        clone_tokens = list(tokens)
        shuffle(clone_tokens)
        train_docs.append(clone_tokens)
        train_labels.append(0)
        for token in set(tokens):
            word_count[token] = word_count.get(token, 0) + 1
    #            
    for line in lines[5000:10000]:
        tokens = word_tokenize(line.strip())
        if ':' in tokens:
            continue
        test_docs.append(tokens)
        test_labels.append(1)
        clone_tokens = list(tokens)
        shuffle(clone_tokens)        
        test_docs.append(clone_tokens)
        test_labels.append(0)
    #    
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
        

vocab_size = len(word_index)
embedding_vector_length = 100 

# Initial embedding_matrix
embedding_matrix = np.zeros((vocab_size, embedding_vector_length))

for word, index in word_index.items():
	if word in wv_model:
		embedding_matrix[index] = wv_model[word]

# Create data
maxlen = 24
Xtrain = [encode_tokens(doc) for doc in train_docs]
Xtrain = pad_sequences(Xtrain, value=0, maxlen=maxlen)
ytrain = np.array(train_labels)

Xtest = [encode_tokens(doc) for doc in test_docs]
Xtest = pad_sequences(Xtest, value=0, maxlen=maxlen)
ytest = np.array(test_labels)

		
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embeding_vector_length, input_length=maxlen, weights=[np.zeros((vocab_size, embeding_vector_length))], trainable=True))
model.add(keras.layers.LSTM(24, input_shape=(maxlen, embeding_vector_length), dropout=0.1, recurrent_dropout=0.1))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(Xtrain, ytrain, epochs=10, batch_size=64, verbose=1)

model.evaluate(Xtest, ytest)
