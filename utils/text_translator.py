import clip
import json
import torch
from transformers import AutoTokenizer, AutoModel

'''
This file is used for translating the text_json file:
format is: 
{
    class_name1:[text1, text2, ... ,textN],
    ...
    class_namex:[text1, text2, ... ,textN]
}
'''

class ClipTranslator():
    '''
    function: Translate the sentence input to embedding using CLip_textencoder
    '''
    def __init__(self, pretrained_path, ):
        #self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = 'cpu'

        self.clip, _ = clip.load(pretrained_path, device = self._device)

    def preprocess_for_dataset(self, json_path, out_feature_path):
        '''
        Read the info about one dataset from json file and translate it into embeddings
        The input and output feature all organalized as json
        '''
        embeddings = {}
        with open(json_path, 'r') as f:
            info = json.load(f)
        classes = info.keys()
        for i in classes:
            texts = clip.tokenize(info[i]).to(self._device)
            with torch.no_grad():
                text_features = self.clip.encode_text(texts)
            embeddings[i] = text_features.data.cpu().tolist()
        with open(out_feature_path, 'w') as f: 
            json.dump(embeddings, f, indent=1)
        print('Complete the encoding')

class BertTanslater:
    '''
    translate the sentence input to embedding using Bert Model
    '''
    def __init__(self, model_path, key='pooler_output'):
        self.key = key
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
    
    def preprocess_for_dataset(self, json_path, out_feature_path):
        embeddings = {}
        with open(json_path, 'r') as f:
            info = json.load(f)
        classes = info.keys()
        for i in classes:
            texts = [self.tokenizer(t, return_tensors='pt') for t in info[i]]
            with torch.no_grad():
                if self.key == 'pooler_output':
                    text_features = torch.cat([self.model(**inputs)[self.key] for inputs in texts], dim=0)
                elif self.key == 'last_hidden_state':
                    text_features = torch.cat([self.model(**inputs)[self.key][:, 0] for inputs in texts], dim=0)
                else:
                    raise NotImplemented
            
            print(f'class {i}:shape {text_features.shape}')
            embeddings[i] = text_features.data.cpu().tolist()
        with open(out_feature_path, 'w') as f:
            json.dump(embeddings, f, indent=1)
        print('Complete the encoding')

if __name__ == '__main__':
    bert_path = ''
    bert = BertTanslater(bert_path)
    bert.preprocess_for_dataset('data/dataset_info/text/organmnist_short_info.json', 'data/dataset_info/features/organmnist_short_info.json')
