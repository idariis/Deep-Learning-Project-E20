# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:46:03 2020

@author: elisa
"""
#%%
from datasets import load_from_disk
from pathlib import Path
deep_learning_dir = Path(__file__).parent.parent.parent

#%%
# Load data
data = load_from_disk(deep_learning_dir/'Data/Processed/')

#%%
data = data.map(lambda example: {'question_lower': example['question'].lower()})

#%% Make tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

#%%
vectorizer = TfidfVectorizer(lowercase = False)
vectorizer.fit(data['wiki_text_lower'])
X_text = vectorizer.transform(data['wiki_text_lower'])
X_question = vectorizer.transform(data['question_lower'])

#%% 
tmp = X_text*X_question.T

#%%