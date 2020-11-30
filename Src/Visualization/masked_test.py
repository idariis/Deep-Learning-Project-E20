#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:58:04 2020

@author: amalieduejensen
"""
import torch
from transformers import BertTokenizer, BertForMaskedLM
from pathlib import Path
from inspect import getfile

src_file_path = getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent#.parent.parent

#%%
# Masked prediction with initial Bert

vocab = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(vocab)

text = "[CLS] Where in England was Dame Judi Dench born ? [SEP] Dame Judi was born in york [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a word for prediction
mask1 = 19
tokenized_text[mask1] = "[MASK]"
assert tokenized_text == ['[CLS]','where','in','england','was','dame','ju','##di','den','##ch','born','?','[SEP]','dame','ju','##di','was','born','in','[MASK]','[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to Q and A
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(vocab)
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

#prediction for mask1
predicted_index = torch.argmax(predictions[0, mask1]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token) #returns "england"


#%%
# Masked prediction with fine-tuned Bert

model = BertForMaskedLM.from_pretrained(deep_learning_dir/'Src/Models/saved_model1')
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

#prediction for mask1
predicted_index = torch.argmax(predictions[0, mask1]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token) #returns random word


