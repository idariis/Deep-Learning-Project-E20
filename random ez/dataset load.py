# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:26:22 2020

@author: elisa
"""

from datasets import load_dataset

test = load_dataset("wiki_qa", split = 'test')
#%%
train = load_dataset("wiki_qa", split = 'train')

#%% 
dev = load_dataset("wiki_qa", split = 'validation')

#%%

for i in range(100):
    if(train[i]['label'] == 1):
        print(train[i]['question'])
        print(train[i]['answer'])
    
