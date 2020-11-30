#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:19:32 2020

@author: amalieduejensen
"""

from datasets import load_dataset

from datasets import load_from_disk
from pathlib import Path
from inspect import getfile
from transformers import AutoTokenizer, BertModel, AdamW
from Src.Data.preprocessing_function import preprocess_data


src_file_path = getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent.parent.parent

#%%
data_validation = load_dataset('trivia_qa', name = 'rc', split = 'validation')

#%%
# Saving sample

sample_validation = data_validation.select(range(1000))
sample_validation = sample_validation.map(remove_columns=(['search_results']))


sample_validation.save_to_disk(deep_learning_dir/'Data/Raw/validation/')

#%%
data = load_from_disk(deep_learning_dir/'Data/Raw/validation/')
data_preprocessed = preprocess_data(data)

#%%
data_train = load_from_disk(deep_learning_dir/'Data/Raw/')
data_preprocessed_train = preprocess_data(data_train)
