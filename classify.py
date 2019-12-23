import numpy as np
import json
from model import Model
def main():
    config = json.load(open('./config/config.json','r'))
    seq_len = config['seq_len']
    if config['use_model_classify']['user_or_not'] == True:
        model = Model(seq_len)
        model.load_model(config['use_model_classify']['model_dir'])
        texts = json.load(open('./config/text.json','r'))['texts']
        model.classify(texts)
if __name__ == '__main__':
    main()
        