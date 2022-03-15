
import warnings
warnings.simplefilter('ignore')
import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from transformers import BertModel,BertTokenizer,get_linear_schedule_with_warmup,AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import train_loader,val_loader
from model import model
from tqdm import tqdm
from categorical_acc import cat_acc

if torch.cuda.is_available():
  device=torch.device('cuda:0')
  print('Cuda')
else:
  device=torch.device('cpu')
  print('cpu')


print('\n Preparing To Train')


optimizer = AdamW(model.parameters(),
                  lr=0.0001, 
                  eps=1e-8)
                  
epochs=1

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_loader)*epochs)

criterion=nn.BCEWithLogitsLoss().to(device)



def train(epoch,trainig_loader,val_loader=None,evalution=False):
    losses=[]
    val_losses=[]
    val_accs=[]
    model.train()
    for _,data in enumerate(trainig_loader, 0):
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['labels'].to(device)

        outputs = model(ids, mask, token_type_ids)
        

        optimizer.zero_grad()
        loss = criterion(torch.sigmoid(outputs), targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()           
        scheduler.step()
    if evalution == True:
      preds,targets = evaluate(model,val_loader,cat_acc)
      val_loss = criterion(preds,targets)
      val_losses.append(val_loss.item())
      preds= preds.clone().detach() >= 0.5
      acc=cat_acc((preds,targets))
      val_accs.append(acc)
    torch.save(model.state_dict(),f"model.pt")
    print(f'Train Loss : {np.mean(losses)} \n validation Loss : {np.mean(val_losses)}\n Validation Accuracy : {np.mean(val_accs)}')

def evaluate(model,val_loader,loss_f):
  print('evaluation Started ')
  model.eval()
  all_outputs=[]
  all_targets=[]
  
  for step,batch in enumerate(val_loader):
    ids = batch['input_ids'].to(device, dtype = torch.long)
    mask = batch['attention_mask'].to(device, dtype = torch.long)
    token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
    targets = batch['labels'].to(device)

    with torch.no_grad():
      outputs= model(ids,mask,token_type_ids)
      all_targets.extend(targets.cpu().detach().numpy())
      all_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy())
  all_targets=np.array(all_targets)
  all_outputs=np.array(all_outputs)
  
  return torch.tensor(all_outputs), torch.tensor(all_targets)


for epoch in tqdm(range(epochs)):
	train(train_loader,val_loader,True)


