import nlpaug
import nlpaug.augmenter.char as nac

import nlpaug.flow as nafc
from tqdm import tqdm
import json

with open('demo-intents.json','r+') as f:
  data=json.load(f)

class Augmentation:
  def __init__(self,data):
    self.data=data
  
  def augment_types(self):
    

    aug_ocr = nac.OcrAug(name='OCR_Aug', aug_char_min=1, aug_char_max=10, aug_char_p=0.3, 	   		aug_word_p=0.3, aug_word_min=1, aug_word_max=10, stopwords=None, tokenizer=None, reverse_tokenizer=None, verbose=0, stopwords_regex=None,min_char=1)
    
    rand = nac.RandomCharAug(action='substitute', name='RandomChar_Aug', aug_char_min=1, aug_char_max=10, aug_char_p=0.3, 
                        aug_word_p=0.3, aug_word_min=1, aug_word_max=10, include_upper_case=True, include_lower_case=True, 
                        include_numeric=True, min_char=4, swap_mode='adjacent', spec_char='!@#$%^&*()_+', stopwords=None, 
                        tokenizer=None, reverse_tokenizer=None, verbose=0, stopwords_regex=None, candidiates=None)
 
  
    aug = nafc.Sequential([
    aug_ocr,rand
    ])
    
    return aug
  
  def Augment(self):
    tags= []
    patterns = []
    responses = []
    print('started Augmentation')
    for i in tqdm(range(5)):
      for intent in self.data['intents']:
        for pattern in intent['patterns']:
          tags.append(intent['tag'])
          patterns.append(pattern)
          responses.extend(intent['responses'])
          augmentation = self.augment_types().augment(pattern, n=700)
          patterns.extend(augmentation)
          tags.extend([intent['tag']]*700)
          responses.extend(intent['responses']*700)
    final_dict = {'Tags' : tags , 'patterns' : patterns, 'responses' :responses } 
    return final_dict

augmentation_class =Augmentation(data)


