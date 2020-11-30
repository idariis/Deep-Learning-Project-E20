# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:41:54 2020

@author: elisa
"""
from datasets import load_from_disk
from pathlib import Path
from inspect import getfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from Src.Data.preprocessing_function import preprocess_data
import Src.Models.DPR_functions as DPR

src_file_path = getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent.parent.parent

#%% Load and preprocess data
data = load_from_disk(deep_learning_dir/'Data/Raw/')
data_preprocessed = preprocess_data(data)

#%% Define inputs
n = 1000

questions = data_preprocessed['question'][0:n]
question_ids = data_preprocessed['question_id'][0:n]
paragraphs = data_preprocessed['paragraph'][0:n]


#%% Get similarity matrices
tfidf_sim = DPR.get_tfidf_similarity(questions, paragraphs)
sim_BERT = DPR.get_BERT_similarity(questions, paragraphs)
sim_BERT_normalized = StandardScaler().fit_transform(sim_BERT)

#%% Get accuracies for a range of ks
k_list = [i+1 for i in range(int(n/4))]
acc_tfidf = DPR.get_accuracy_vector(k_list, tfidf_sim, question_ids, question_ids)
acc_bert = DPR.get_accuracy_vector(k_list, sim_BERT, question_ids, question_ids)
acc_bert_normalized = DPR.get_accuracy_vector(k_list, sim_BERT_normalized, question_ids, question_ids)
acc_random = DPR.get_random_accuracy(k_list, n)

#%%
plt.plot(k_list, acc_tfidf, label = 'TF-IDF')
plt.plot(k_list, acc_bert, label = 'BERT INIT')
plt.plot(k_list, acc_random, label = 'RANDOM')
plt.plot(k_list, acc_bert_normalized, label = 'BERT INIT NORMALIZED')
plt.legend()
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

#%%
plt.matshow(sim_BERT)
plt.title('BERT similarity matrix')
plt.matshow(tfidf_sim)
plt.title('TF-IDF similarity matrix')
plt.matshow(sim_BERT_normalized)
plt.title('BERT similarity matrix - normalized')







