#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:00:56 2020

@author: amalieduejensen
"""

from datasets import load_dataset
from pathlib import Path
from inspect import getfile

from Src.Data.preprocessing_function import preprocess_data

src_file_path = getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent.parent.parent

#%% Load data
train_data = load_dataset('trivia_qa', name = 'rc', split = 'train')
validation_data = load_dataset('trivia_qa', name = 'rc', split = 'validation')

train_all = train_data.map(remove_columns=(['search_results']))
validation_all = validation_data.map(remove_columns=(['search_results']))

#%% Preprocess data
train_preprocessed = preprocess_data(train_all)
validation_preprocessed = preprocess_data(validation_all)

#%% Save data
train_preprocessed.save_to_disk(deep_learning_dir/'Data/Raw/train_all/')
validation_preprocessed.save_to_disk(deep_learning_dir/'Data/Raw/validation_all/')

#%%
from datasets import load_from_disk

test = load_from_disk(deep_learning_dir/'Data/Raw/train_all/')
