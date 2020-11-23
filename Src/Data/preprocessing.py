# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:28:40 2020

@author: elisa

Script contains the functions needed to pre-process the data for use in 
BERT or tf-idf. 

"""

from Src.Data.preprocessing_function import *
import inspect

#%%
src_file_path = inspect.getfile(lambda: None)
path = Path(src_file_path).parent.parent.parent
#%%
data = load_from_disk(path/'Data/Raw/')

#%%



# Work in progress below, might be useful later


#def find_best_paragraph_with_answer(top_k_paragraphs, answer):
#    k = len(top_k_paragraphs)
#    for i in range(k):
#        tmp = top_k_paragraphs[i].find(answer)
#        if tmp != -1:
#            return top_k_paragraphs[i]
#    return top_k_paragraphs[0]

#%% Usage:
data_preprocessed = preprocess_data(data, 128, False)

#%% Quick look at pre-processed data
print(data_preprocessed.features)

