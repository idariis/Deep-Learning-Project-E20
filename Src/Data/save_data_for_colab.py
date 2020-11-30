# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:09:03 2020

@author: elisa
"""

import torch
from datasets import load_from_disk
from pathlib import Path
from inspect import getfile
from Src.Data.preprocessing_function import preprocess_data


src_file_path = getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent.parent.parent

#%% Load and preprocess data
data = load_from_disk(deep_learning_dir/'Data/Raw/')
data_preprocessed = preprocess_data(data)

#%% Save data
data_preprocessed.save_to_disk(deep_learning_dir/'Data/Processed/colab_data')