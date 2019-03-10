import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pyvi import ViTokenizer, ViPosTagger
import re

special_char_regex = '.*[0-9~!@#$%^&\-\+={}\[\]\\|/<>?“”"‘’].*'

def is_valid_word(word):
    return re.match(special_char_regex, word) == None

def word_tokenize(sentence):
    words, postags = ViPosTagger.postagging(ViTokenizer.tokenize(sentence.lower()))
    return [word for word in words if is_valid_word(word)]
    

topics = ['xahoi' , 'kinhdoanh', 'thethao', 'vanhoa']
topic_names = ['Xã hội', 'Kinh doanh', 'Thể thao', 'Văn hóa']

docs = []

for i in range(len(topics)):
    fn = os.path.join('data/headlines', topics[i] + '.txt')
    f = open(fn, encoding='utf8')
    docs.extend([line.strip() for line in f.readlines()[:2000]])
    f.close()

tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[str(i)]) for i, doc in enumerate(docs)]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)


for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")

test_data = word_tokenize(docs[0])
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])