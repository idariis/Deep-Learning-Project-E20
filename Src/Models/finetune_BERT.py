# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:41:09 2020

@author: elisa
"""
import torch
from datasets import load_from_disk
from pathlib import Path
from inspect import getfile
from transformers import AutoTokenizer
from Src.Data.preprocessing_function import preprocess_data


src_file_path = getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent.parent.parent

#%% Functions 
def get_loss(sim):
    nll = -(torch.diagonal(sim) - torch.logsumexp(sim, dim = 1))
    return nll# return negative loss

#%% Load and preprocess data
data = load_from_disk(deep_learning_dir/'Data/Raw/')
data_preprocessed = preprocess_data(data)

#%% Set params
n_sample = 100
batch_size = 10


#%% Prepare data for minibatch training
train_sample = data_preprocessed.select(range(n_sample))

#%% Attempt to tokenize (not quite sure about how this should be done)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', padding = True)
def tokenize_all(examples):
    return tokenizer(examples['question'], examples['paragraph'])
train_sample = train_sample.map(tokenize_all)

#%% Change to pytorch format. 
# columns selects columns to keep - not sure what is needed. 
# No strings allowed in a tensor? 
train_sample.set_format(type = 'torch', columns = ['token_type_ids'])
#%% 
trainloader = torch.utils.data.DataLoader(train_sample, batch_size=batch_size)
trainloader = iter(trainloader)

#%% Yay, training here
for i, batch in enumerate(trainloader):
    print(batch.shape)