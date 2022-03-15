from augmentation import augmentation_class
import pandas as pd 
import numpy as np
import json
with open('demo-intents.json','r+') as f:
  data=json.load(f)


augmentation_dict = augmentation_class.Augment()

class PrepareData:
  def __init__(self,data,aug_data):
    self.data=data
    self.aug_data=aug_data
    self.dataframe= self.create_dataframe()
    self.labels= self.convert_labels()
  
  def create_dataframe(self):
    df_valuess=list(zip(self.aug_data['Tags'],self.aug_data
    		 ['patterns'],self.aug_data['responses']))
    df=pd.DataFrame(df_valuess,columns=['tags','patterns','responses'])
    return df
  
  def convert_labels(self):
    classes=self.dataframe['tags'].unique()
    labels = {}
    for index, label in enumerate(classes):
      labels[label] = index
    
    return labels
  

  def add_labels_to_json(self,data):
    for key,value in self.labels.items():
      for intent in data['intents']:
        if intent['tag'] ==key:
          intent['label'] = value
    return data
  
  def create_csv(self):
    dataframe=self.dataframe
    dataframe['tags'] = dataframe['tags'].replace(self.labels)
    dataframe=dataframe.iloc[np.random.permutation(len(dataframe))]
    dataframe=dataframe.reset_index(drop=True)
    dataframe.to_csv('dataset.csv',index=False,header=('tag','patterns','responses'))
    return 'Created Dataframe'

create_csv_inst=PrepareData(data,augmentation_dict)
create_csv=create_csv_inst.create_csv()

labeled_data=create_csv_inst.add_labels_to_json(data)

