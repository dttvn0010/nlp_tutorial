import os
import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from random import shuffle
from pyvi import ViTokenizer, ViPosTagger
import gensim
import re

special_char_regex = '.*[0-9~!@#$%^&\-\+={}\[\]\\|/<>?“”"‘’].*'

def is_valid_word(word):
    return re.match(special_char_regex, word) == None

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
    #
    for line in lines[5000:10000]:
        tokens = word_tokenize(line.strip())
        if ':' in tokens:
            continue
        test_docs.append(tokens)
        test_labels.append(i)
    f.close()
	
    
dictionary = gensim.corpora.Dictionary(train_docs)
dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=10000)
vocab_size = len(dictionary)

bow_corpus = [dictionary.doc2bow(doc) for doc in train_docs]
tfidf = gensim.models.TfidfModel(bow_corpus)

def createData(docs, labels):
    X = []
    y = []
    #
    for doc, label in zip(docs, labels):
        bow_vector = tfidf[dictionary.doc2bow(doc)] #dictionary.doc2bow(doc)
        wordvec = np.zeros(vocab_size)    
        for index, value in bow_vector:
            wordvec[index] = value
        X.append(wordvec)
        y.append(to_categorical(label, num_classes))
    #
    return np.array(X), np.array(y)

Xtrain, ytrain = createData(train_docs, train_labels)
Xtest, ytest = createData(test_docs, test_labels)


model = keras.Sequential()
model.add(keras.layers.Dense(5, input_dim=vocab_size, activation='sigmoid'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.summary()

model.fit(Xtrain, ytrain, epochs=20, shuffle=True)

model.evaluate(Xtest, ytest)
