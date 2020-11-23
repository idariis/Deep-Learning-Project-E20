# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

@author: Amalie
"""

#%%
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM

def tokenize_BERT(text):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return tokenizer(text, padding = True)

def encoder_BERT(tokenized_text):
    tokens_tensor = torch.tensor(tokenized_text["input_ids"])
    segments_tensor = torch.tensor(tokenized_text["token_type_ids"])
    attention_tensor = torch.tensor(tokenized_text["attention_mask"])
    
    model = BertModel.from_pretrained('bert-base-uncased')
    
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=tokens_tensor, attention_mask=attention_tensor, token_type_ids=segments_tensor)
        encoded_layers = outputs[0]
    
    return encoded_layers

#%%
from Src.Data.preprocessing_function import *
import inspect

src_file_path = inspect.getfile(lambda: None)
path = Path(src_file_path).parent#.parent.parent

data = load_from_disk(path/'Data/Raw/')


data_preprocessed = preprocess_data(data, 128, False)

#%%

paragraph = data_preprocessed[:]["paragraph"]
question = data_preprocessed[:]["question"]

#%%
encoded_dict = tokenize_BERT(question)

encoded_layers = encoder_BERT(encoded_dict)

print(encoded_layers)







