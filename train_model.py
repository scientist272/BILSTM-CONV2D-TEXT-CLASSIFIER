#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from model import Model
from data_preprocessor import DataLoader

def main():
    config = json.load(open('./config/config.json','r'))
    
    train_test_split_ratio = config['train_test_split_ratio']
    seq_len = config['seq_len']
    data_dir = config['data']
    label_col = config['label_col']
    text_col = config['text_col']
    batch_size = config['batch_size']
    epochs = config['epochs']
    save_dir = config['save_dir']
    
    data_loader = DataLoader(train_test_split_ratio,seq_len)
    data_loader.load_data(data_dir,label_col,text_col)
    data_loader.get_train_test_data()
    
    
    model = Model(seq_len)
    if config['use_exist_model']['use_or_not'] == False:
        model.get_model()
    else:
        model.load_model(config['use_exist_model']['model_dir'])
    model.train_model(epochs,save_dir,batch_size)

if __name__ == '__main__':
    main()

