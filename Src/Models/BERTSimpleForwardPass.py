# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:41:52 2020

@author: Caroline
"""

#%%
import inspect
src_file_path = inspect.getfile(lambda: None)
#%% Elisabeths and Idas docs 

from pathlib import Path
from datasets import load_from_disk

#path = Path(__file__).parent.parent.parent

import inspect
src_file_path = inspect.getfile(lambda: None)
path = Path(src_file_path).parent.parent.parent

dataRaw = load_from_disk(path/'Data/Raw/')

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
    

data = preprocess_data(dataRaw, 128, False)


#%% 
# Small test data 
from datasets import load_from_disk
path = 'C:/Users/Caroline/Documents/DTU/9_semester/02456_DeepLearning/deep_learning_project/Data/Processed' # Din path i stedet
test = load_from_disk(path)

smalldat = data[0:100]

# BERT Fast tokenizing 
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, \
    BertTokenizerFast

#%% FastBERT 
tokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
text = test[0]["wiki_text"][0][0:1000]
text2 = test[0]["wiki_text"][0][201:250]
encoded_dict = tokenizerFast([text, text2], padding = True)
tokenizerFast.decode(encoded_dict["input_ids"][0])

#%% BERT MODEL FOR ENCODING 
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor(encoded_dict["input_ids"]) #(batchsize, sequence_length)
segments_tensor = torch.tensor(encoded_dict["token_type_ids"]) #(batchsize, sequence_length)
attention_tensor = torch.tensor(encoded_dict["attention_mask"]) #(batchsize, sequence_length)

#%%
# Predict hidden states features for each layer
with torch.no_grad():
    # Using BERT with the inputs from out tokenizer, the position embedding is done automatically incrementing from left to right
    outputs = model(input_ids = tokens_tensor, attention_mask = attention_tensor, \
                    token_type_ids = segments_tensor)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]
    
# Encoding of first text encoded_layers[0,0,:]
    
#%% Trying to find similarities with question and answers 









