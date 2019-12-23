#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,tokenizer_from_json
import pandas as pd
from nltk.corpus import stopwords
import re
import json
STOPWORDS = set(stopwords.words('english'))
import os

class DataLoader():
    def __init__(self,train_test_split_ratio,seq_len):
        path = './pre-trained'
        self.file_num = 0
        for file in os.listdir(path):
            self.file_num+=1
        
        if not self.file_num == 7:
            self.embedding_dim=100
            self.oov_token = '<OOV>'
            self.train_test_split = train_test_split_ratio
            self.max_len = seq_len
            self.trunc_type = 'post'
            self.padding_type = 'post'
    
    def load_data(self,data_dir,label_col,text_col):
        if not self.file_num == 7:
            df = pd.read_csv(data_dir)
            self.labels = df.get(label_col).values.tolist()
            self.articles = df.get(text_col).values.tolist()
            for i in range(len(self.articles)):
                for word in STOPWORDS:
                    token = ' '+word+' '
                    self.articles[i] = self.articles[i].replace(token,' ')
                    self.articles[i] = re.sub(r'\s+', ' ',self.articles[i])
        
    def get_train_test_data(self):
        if not self.file_num == 7:
            #Get sequence Tokenzier
            tokenizer = Tokenizer(oov_token=self.oov_token)
            tokenizer.fit_on_texts(self.articles)
            sequences = tokenizer.texts_to_sequences(self.articles)
            article_sequences = pad_sequences(sequences,maxlen=self.max_len,truncating=self.trunc_type,padding=self.padding_type)
            word_index = tokenizer.word_index
            vocab_size = len(word_index)
            # Train test split
            split_index = int(len(article_sequences)*self.train_test_split)
            train_sequences = np.array(article_sequences[0:split_index])
            test_sequences = np.array(article_sequences[split_index:])
            # Get label Tokenzier
            label_tokenizer = Tokenizer()
            label_tokenizer.fit_on_texts(self.labels)
            label_sequences = label_tokenizer.texts_to_sequences(self.labels)
            train_label = np.array(label_sequences[0:split_index])
            test_label = np.array(label_sequences[split_index:])
            # Get Glove pre-trained word embedding
            embeddings_index = {};
            with open('glove/glove.6B.100d.txt',encoding='utf-8') as f:
                for line in f:
                    values = line.split();
                    word = values[0];
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs;

            embeddings_matrix = np.zeros((vocab_size+1, self.embedding_dim))
            for word, i in word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embeddings_matrix[i] = embedding_vector
            # seralize objects to local storage            
            tokenizer_json = tokenizer.to_json()
            label_tokenizer_json = label_tokenizer.to_json()
            # Save Tokenizer
            if not os.path.exists('./pre-trained/tokenizer.json'):
                with open('./pre-trained/tokenizer.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
            if not os.path.exists('./pre-trained/label_tokenizer_json.json'):
                with open('./pre-trained/label_tokenizer_json.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(label_tokenizer_json, ensure_ascii=False))
            # Save train test sequence and embeddings matrix
            if not os.path.exists('./pre-trained/train_sequences.npy'):
                np.save('./pre-trained/train_sequences.npy',train_sequences)
            if not os.path.exists('./pre-trained/train_label.npy'):
                np.save('./pre-trained/train_label.npy',train_label)
            if not os.path.exists('./pre-trained/test_sequences.npy'):
                np.save('./pre-trained/test_sequences.npy',test_sequences)
            if not os.path.exists('./pre-trained/test_label.npy'):
                np.save('./pre-trained/test_label.npy',test_label)
            if not os.path.exists('./pre-trained/embeddings_matrix.npy'):
                np.save('./pre-trained/embeddings_matrix.npy',embeddings_matrix)

