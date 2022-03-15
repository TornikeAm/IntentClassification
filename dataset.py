from torch.utils.data import Dataset, DataLoader
import json
import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
with open('demo-intents.json','r+') as f:
  data=json.load(f)



df=pd.read_csv('dataset.csv',encoding='utf-8')

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

y=to_categorical(df['tag'],53)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize(sentences):
  encodings = tokenizer.batch_encode_plus(sentences,
                            add_special_tokens=True,
                            truncation=True,
                            return_token_type_ids=True,
                            padding=True,
                            return_attention_mask=True,
                            return_tensors='pt')
  return encodings
  
tokenized_patterns = tokenize(df['patterns'].values)

class IntentData(Dataset):
  def __init__(self,df,tokenize,y):
    self.df=df
    self.tokenize=tokenize
    self.y=y

  def __getitem__(self,index):
    pattern=self.df.iloc[index]['patterns']
    tag=self.df.iloc[index]['tag']
    return {
        # 'review': pattern,
        'input_ids' : self.tokenize['input_ids'][index].flatten(),
        'attention_mask' : self.tokenize['attention_mask'][index].flatten(),
        'token_type_ids' : self.tokenize['token_type_ids'][index].flatten(),
        'labels': torch.tensor(self.y[index]).float()

        }

  def __len__(self):
    return self.df.shape[0]

dataset=IntentData(df,tokenized_patterns,y)


train_set,val_set=torch.utils.data.random_split(dataset,[int(len(dataset)*0.7),round(len(dataset)*0.3)])
train_loader=DataLoader(train_set,batch_size=64,shuffle=True)
val_loader=DataLoader(val_set,batch_size=16,shuffle=False)

