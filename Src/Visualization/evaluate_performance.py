# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:41:54 2020

@author: elisa
"""
from datasets import load_from_disk
from pathlib import Path
from inspect import getfile

from Src.Data.preprocessing_function import preprocess_data
import Src.Models.DPR_functions as DPR
#from Src.Models.DPR_functions import get_tfidf_similarity, get_top_k,\
#    get_accuracy, get_BERT_similarity

src_file_path = getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent.parent.parent

#%% Load and preprocess data
data = load_from_disk(deep_learning_dir/'Data/Raw/')
data_preprocessed = preprocess_data(data)

#%% Define inputs
k = 5
n = 100

questions = data_preprocessed['question'][0:n]
question_ids = data_preprocessed['question_id'][0:n]
paragraphs = data_preprocessed['paragraph'][0:n]


#%% Get tfidf results
tfidf_sim = DPR.get_tfidf_similarity(questions, paragraphs)
top_k = DPR.get_top_k(tfidf_sim, question_ids, question_ids, k)
acc_tfidf = DPR.get_accuracy(top_k)
print(f'Accuracy of tfidf: {acc_tfidf}')

#%% Get BERT results
sim_BERT = DPR.get_BERT_similarity(questions, paragraphs)
top_k_BERT = DPR.get_top_k(sim_BERT.numpy(), question_ids, question_ids, k)
acc_BERT = DPR.get_accuracy(top_k_BERT)
print(f'Accuracy of BERT (no training): {acc_BERT}')


