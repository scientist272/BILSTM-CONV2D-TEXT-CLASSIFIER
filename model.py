#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, LSTM, Embedding,Conv2D,Bidirectional,MaxPool2D,Reshape
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,tokenizer_from_json
from keras.callbacks import ModelCheckpoint
import json
import os
import datetime as dt
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

class Model():
    
    def __init__(self,seq_len):
        with open('./pre-trained/tokenizer.json', 'r', encoding='utf-8') as f1:
            tokenizer_config = json.load(f1)
        with open('./pre-trained/label_tokenizer_json.json', 'r', encoding='utf-8') as f2:
            label_tokenizer_config = json.load(f2)
        self.tokenizer = tokenizer_from_json(tokenizer_config)
        self.label_tokenizer = tokenizer_from_json(label_tokenizer_config)
        self.train_sequences= np.load('./pre-trained/train_sequences.npy')
        self.train_label= np.load('./pre-trained/train_label.npy')
        self.test_sequences = np.load('./pre-trained/test_sequences.npy')
        self.test_label= np.load('./pre-trained/test_label.npy')
        self.embeddings_matrix= np.load('./pre-trained/embeddings_matrix.npy')
        self.embedding_dim = 100
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index)
        self.max_len = seq_len
        self.rnn_units = self.embedding_dim
        self.category_num = len(set(self.test_label[:,0]))

    def get_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size+1,self.embedding_dim,input_length=self.max_len,weights=[self.embeddings_matrix], trainable=False))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(self.rnn_units,return_sequences=True),merge_mode='sum'))
        self.model.add(Dropout(0.2))
        self.model.add(Reshape((-1,self.rnn_units,1)))
        self.model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(self.category_num+1,activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
        self.model.summary()

    #load pre-trained model then keeping training or just using
    def load_model(self,model_dir):
        self.model = load_model(model_dir)

    def classify(self,texts):
        seq = np.array(pad_sequences(self.tokenizer.texts_to_sequences(texts),truncating='post',padding='post',maxlen=self.max_len))
        result = np.argmax(self.model.predict(seq),axis=1)[:,None].tolist()
        out = self.label_tokenizer.sequences_to_texts(result)
        print(out)
                           
    def train_model(self,epochs,save_dir,batch_size):
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)]
        history = self.model.fit(self.train_sequences,self.train_label,epochs=epochs,validation_data=(self.test_sequences,self.test_label),callbacks=callbacks,batch_size = batch_size)
        self.plot_graphs(history, "accuracy")
        self.plot_graphs(history, "loss")


    def plot_graphs(self,history, string):
          plt.figure()
          plt.plot(history.history[string])
          plt.plot(history.history['val_'+string])
          plt.xlabel("Epochs")
          plt.ylabel(string)
          plt.legend([string, 'val_'+string])
          plt.savefig('./train_history/'+string+'.png')

