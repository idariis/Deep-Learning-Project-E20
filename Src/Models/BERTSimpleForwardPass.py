# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:41:52 2020

@author: Caroline
"""
# Small test data 
from datasets import load_from_disk
path = 'C:/Users/Caroline/Documents/DTU/9_semester/02456_DeepLearning/deep_learning_project/Data/Processed' # Din path i stedet
test = load_from_disk(path)

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
    
    
    