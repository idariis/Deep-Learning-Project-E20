# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:41:09 2020

@author: elisa
"""
import torch
from datasets import load_from_disk
from pathlib import Path
from inspect import getfile
from transformers import AutoTokenizer, BertModel, AdamW
from Src.Data.preprocess_3 import preprocess_data
from math import ceil


src_file_path = getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent.parent.parent

#%% Functions 
def get_loss(sim):
    nll = -(torch.diagonal(sim) - torch.logsumexp(sim, dim = 1))
    return sum(nll)# return negative loss

#%% Load and preprocess data
train_data = load_from_disk(deep_learning_dir/'Data/Raw/')
train_data_preprocessed = preprocess_data(train_data)

validation_data = load_from_disk(deep_learning_dir/'Data/Raw/validation/')
validation_data_preprocessed = preprocess_data(validation_data)

#%% Set params
n_sample = 9
batch_size = 3
n_batches = ceil(n_sample/batch_size)

#%% Attempt to tokenize (not quite sure about how this should be done)
train_sample = train_data_preprocessed.select(range(n_sample))
validation_sample = validation_data_preprocessed.select(range(n_sample))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', padding = True)

#%%
train_sample = train_sample.map(lambda example: {
    'Q_input_ids': tokenizer(list(example['question']), padding = 'max_length')['input_ids'],
    'Q_attention_mask': tokenizer(list(example['question']), padding = 'max_length')['attention_mask'],
    'Q_token_type_ids': tokenizer(list(example['question']), padding = 'max_length')['token_type_ids'],
    'P_input_ids': tokenizer(list(example['paragraph']), padding = 'max_length')['input_ids'],
    'P_attention_mask': tokenizer(list(example['paragraph']), padding = 'max_length')['attention_mask'],
    'P_token_type_ids': tokenizer(list(example['paragraph']), padding = 'max_length')['token_type_ids']},
    batched = True, batch_size= batch_size)

validation_sample = validation_sample.map(lambda example: {
    'Q_input_ids': tokenizer(list(example['question']), padding = 'max_length')['input_ids'],
    'Q_attention_mask': tokenizer(list(example['question']), padding = 'max_length')['attention_mask'],
    'Q_token_type_ids': tokenizer(list(example['question']), padding = 'max_length')['token_type_ids'],
    'P_input_ids': tokenizer(list(example['paragraph']), padding = 'max_length')['input_ids'],
    'P_attention_mask': tokenizer(list(example['paragraph']), padding = 'max_length')['attention_mask'],
    'P_token_type_ids': tokenizer(list(example['paragraph']), padding = 'max_length')['token_type_ids']},
    batched = True, batch_size= batch_size)


#%% Change to pytorch format. 
# columns selects columns to keep - not sure what is needed. 
# No strings allowed in a tensor? 
train_sample.set_format(type = 'torch', 
                        columns = ['Q_input_ids', 'Q_attention_mask', 'Q_token_type_ids',
                                   'P_input_ids', 'P_attention_mask', 'P_token_type_ids'])

validation_sample.set_format(type = 'torch', 
                        columns = ['Q_input_ids', 'Q_attention_mask', 'Q_token_type_ids',
                                   'P_input_ids', 'P_attention_mask', 'P_token_type_ids'])

#%%
lr = 5e-5
n_epochs = 3

#%%
model = BertModel.from_pretrained('bert-base-uncased')
optim = AdamW(model.parameters(), lr = lr)

#%% Yay, training here
epoch_train_loss = [None]*n_batches
epoch_validation_loss = [None]*n_batches
train_loss = [None]*n_epochs
validation_loss = [None]*n_epochs

for epoch in range(n_epochs):
    print(f'epoch: {epoch}')
    trainloader = torch.utils.data.DataLoader(train_sample, batch_size=batch_size)
    trainloader = iter(trainloader)
    
    validationloader = torch.utils.data.DataLoader(validation_sample, batch_size=batch_size)
    validationloader = iter(validationloader)
    
    for i, batch in enumerate(trainloader):
        print(i)
        
        # (get extra observation in training set)
        
        # Make forward pass
        P_outputs = model(input_ids=batch['P_input_ids'], 
                          attention_mask=batch['P_attention_mask'], 
                          token_type_ids=batch['P_token_type_ids'])
        Q_outputs = model(input_ids=batch['Q_input_ids'], 
                          attention_mask=batch['Q_attention_mask'], 
                          token_type_ids=batch['Q_token_type_ids'])
        P_encoded_layers = P_outputs[0][:, 0, :]
        Q_encoded_layers = Q_outputs[0][:, 0, :]
    
    
        # Calculate similarity matrix
        sim_matrix = torch.matmul(Q_encoded_layers, P_encoded_layers.T)
        
        # Get loss
        loss = get_loss(sim_matrix)
        epoch_train_loss[i] = loss
        print(loss)
        
        # Update weights
        loss.backward()
        optim.step()
        
    for i, batch in enumerate(validationloader):
        print(i)
        
        # (get extra observation in training set)
        
        # Make forward pass
        P_outputs = model(input_ids=batch['P_input_ids'], 
                          attention_mask=batch['P_attention_mask'], 
                          token_type_ids=batch['P_token_type_ids'])
        Q_outputs = model(input_ids=batch['Q_input_ids'], 
                          attention_mask=batch['Q_attention_mask'], 
                          token_type_ids=batch['Q_token_type_ids'])
        P_encoded_layers = P_outputs[0][:, 0, :]
        Q_encoded_layers = Q_outputs[0][:, 0, :]
    
    
        # Calculate similarity matrix
        sim_matrix = torch.matmul(Q_encoded_layers, P_encoded_layers.T)
        
        # Get loss
        loss = get_loss(sim_matrix)
        epoch_validation_loss[i] = loss
        print(loss)
        
        # Update weights
        #loss.backward()
        #optim.step()
    
    train_loss[epoch] = sum(epoch_train_loss)
    validation_loss[epoch] = sum(epoch_validation_loss)
    
    
    print(f'train loss: {train_loss[epoch]}')
    print(f'validation loss: {validation_loss[epoch]}')
    

#%% Saving model
model.save_pretrained(deep_learning_dir/'Src/Models/saved_model1')

#%%
test_model = BertModel.from_pretrained(deep_learning_dir/'Src/Models/saved_model1')
#%%
P_outputs = test_model(input_ids=batch['P_input_ids'], 
                          attention_mask=batch['P_attention_mask'], 
                          token_type_ids=batch['P_token_type_ids'])
#%%
base_model = BertModel.from_pretrained('bert-base-uncased')
