import pandas as pd 
df=pd.read_json('Sarcasm_Headlines_Dataset.json',lines=True)
df=df.drop(['article_link'],axis=1)
df['len'] = df['headline'].apply(lambda x: len(x.split(" ")))
print(df.head())

import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,CuDNNLSTM,Embedding,Dropout,Activation,Flatten,Bidirectional,GlobalMaxPool1D
from keras.models import Model,Sequential

max_features=10000
max_len=25
embedding_size=200

tokenizer=Tokenizer(num_words=max_features)
#only 9999 most frequent words will be kept
tokenizer.fit_on_texts(list(df['headline']))
#it needs all sent in a lsit
X=tokenizer.texts_to_sequences(df['headline'])
#print(X)
#you get a list of list of all the sent and in numbers each word in that.
#seq is a list of list where each ele is a seq.
X=pad_sequences(X,maxlen=max_len)
#print(X)
#pads the seq to the same length
#maxlen=max length of all the sequences
y=df['is_sarcastic']

Embeddings_file='glove.6B.200d.txt'

def get_coefs(word,*arr):
	return word,np.asarray(arr,dtype='float32')
embeddings_index=dict(get_coefs(*o.split(" "))for o in open(Embeddings_file) if len(o)>100)
print(embeddings_index)
#it is a dict of word and the vector containing its embeddings
count=0
for key,value in embeddings_index.items():
	count=count+1
print(count)
#40000 word embeddings are there in that

all_embs=np.stack(embeddings_index.values())
#join the seq of arrays along a new axis.If axis=0 then 1st axis and if -1 the along last axis
#getting all the embeddings in all_embs

emb_mean,emb_std=all_embs.mean(),all_embs.std()
embed_size=all_embs.shape[1]
print(emb_mean)
#-0.008671842
print(emb_std)
#0.3818617
print(all_embs.shape)
#(400000, 200)
word_index=tokenizer.word_index
print(word_index)
#all words index is given
#it is a dict
nb_words=min(max_features,len(word_index))
print(len(word_index))
#29656
print(nb_words)
#10000
embedding_matrix=np.random.normal(emb_mean,emb_std,(nb_words,embedding_size))
print(embedding_matrix.shape)
#10000,200

for word,i in word_index.items():
	if i>=max_features:
		continue
	#stating that less than 10000
	embedding_vector=embeddings_index.get(word)
	#dict.get(key)
	#returns the value for the key
	if embedding_vector is not None:
		embedding_matrix[i]=embedding_vector

#you take the word from word_index
#pass it to embeddings index to get the value
#store that in embeddings matrix

#emb_file is for word and its vector
#shape is 40000,200
#word_index is word and its index number
#len is 29656





model = Sequential()
model.add(Embedding(max_features, embedding_size, weights = [embedding_matrix]))
model.add(Bidirectional(128, return_sequences = True))
model.add(GlobalMaxPool1D())
model.add(Dense(40, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size=100
epochs=5
model.fit(X,y,batch_size=batch_size,epochs=epochs
	,validation_split=0.2)
