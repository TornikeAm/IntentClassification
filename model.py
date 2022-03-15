import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

if torch.cuda.is_available():
  device=torch.device('cuda:0')
  print('Cuda')
else:
  device=torch.device('cpu')
  print('cpu')
  
  
class BertIntentClassification(torch.nn.Module):
    def __init__(self):
        super(BertIntentClassification, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased',return_dict=False)
        self.l2 = torch.nn.Linear(768, 53)
 
    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))
    
    def forward(self, ids, mask,token_type_ids):
        _, output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output = self.l2(output_1)
        return self.sigmoid(output)
 
model = BertIntentClassification().to(device)

