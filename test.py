from model import BertIntentClassification
import json
import warnings
from transformers import BertTokenizer
import torch
warnings.simplefilter('ignore')

exec(open('PrepareData.py').read())

exec(open('training.py').read())
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
models = BertIntentClassification()

models.load_state_dict(torch.load('model.pt',map_location = torch.device('cpu')))

models.eval()
testing = True
while testing:
	print('input Sentence// Stop = N \n')
	sentence = input()
	if sentence == 'N' or sentence =='n':
		testing = False

	tokenized=tokenizer(sentence)
	logits=models(torch.tensor(tokenized['input_ids']).unsqueeze(0), 			
		  torch.tensor(tokenized['attention_mask']).unsqueeze(0),
		  torch.tensor(tokenized['token_type_ids']))
	class_ = torch.argmax(logits,dim=1).flatten()
	for intent in data['intents']:
  		if intent['label'] == class_:
    			print(f"{class_} : {random.choice(intent['responses'])}")
    	
	
	



