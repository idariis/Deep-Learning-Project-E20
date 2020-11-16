# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:28:40 2020

@author: elisa
"""

from pathlib import Path
from datasets import load_from_disk

#%%
path = Path(__file__).parent.parent.parent
#%%
data = load_from_disk(path/'Data/Raw/')

#%%
def preprocess_data(data, text_length, remove_search_results):
    """ 
    Takes the raw data and creates paragraph dataset

    
    """
    if remove_search_results:
        print('Hovsa, det skal du selv goere.')
    
    out_data = data.map(lambda example: {'wiki_text': example['entity_pages']['wiki_context'], 
                                             'answer': example['answer']['normalized_value']}, 
                            remove_columns=['question_source', 'answer', 'entity_pages'])
    out_data = out_data.filter(lambda example: len(example['wiki_text']) > 0)
    out_data = out_data.map(lambda example: {'paragraph': get_paragraph(example)})
    out_data = out_data.map(remove_columns = ['wiki_text'])
    
    return(out_data)

def get_paragraph(example):
    text_length = 128
    n_docs = len(example['wiki_text'])
    for i in range(n_docs):
        idx = example['wiki_text'][i].lower().find(example['answer'])
        if idx != -1:
            idx_lwr = idx-text_length
            idx_upr = idx+text_length
            if idx_lwr < 0:
                idx_lwr = 0
                idx_upr = 2*text_length
            elif idx_upr > len(example['wiki_text'][i]):
                idx_upr = len(example['wiki_text'][i])
                idx_lwr = idx_upr - 2*text_length
            paragraph = example['wiki_text'][i].lower()[idx_lwr:idx_upr]
            return(paragraph)
    

#%%
tmp = preprocess_data(data, 128, False)

#%%
example = tmp[1]
text_length = 64
paragraph = ''
n_docs = len(example['wiki_text'])
for i in range(n_docs):
    idx = example['wiki_text'][i].lower().find(example['answer'])
    if idx != -1:
        paragraph = example['wiki_text'][i].lower()[(idx-text_length):(idx+text_length)]
        break

#%%






