#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:21:36 2020

@author: IdaRiis
"""
from datasets import load_dataset
from pathlib import Path

#%%
data_train = load_dataset('trivia_qa', name = 'rc', split = 'train')

#%%
# Saving sample

sample = data_train.select(range(1000))
sample = sample.map(remove_columns=(['search_results']))

path = Path(__file__).parent.parent.parent

sample.save_to_disk(path/'Data/Raw/')