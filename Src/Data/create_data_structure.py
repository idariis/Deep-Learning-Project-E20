# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:28:40 2020

@author: elisa

"""

from pathlib import Path
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

#%%
path = Path(__file__).parent.parent.parent
#%%
data = load_from_disk(path/'Data/Raw/')

#%%
def preprocess_data(data, paragraph_len = 128, remove_search_results = False):
    """ 
    NB jeg mangler at skrive al dokumentationen - det sker senere :)   
    """
   
    # Select appropiate columns
    if remove_search_results:
        data = data.map(remove_columns = ['search_results'])
    out_data = data.map(lambda example: {'wiki_text': example['entity_pages']['wiki_context'], 
                                         'wiki_title': example['entity_pages']['title'],
                                             'answer': example['answer']['normalized_value']}, 
                            remove_columns=['question_source', 'answer', 'entity_pages'])
    
    # Remove entries without wiki text
    out_data = out_data.filter(lambda example: len(example['wiki_text']) > 0)
    
    # Get paragraph with the answer
    out_data = out_data.map(lambda example: {
        'paragraph': get_paragraph_with_answer(example, paragraph_len)
        }, remove_columns = ['wiki_text', 'wiki_title'])
    
    return(out_data)

def get_paragraph_with_answer(example, paragraph_len):
    paragraphs = get_all_paragraphs(example, paragraph_len)
    target = example['question'] + example['answer']
    sim = get_tfidf_similarity([target], paragraphs)
    idx_answer_paragraph = np.argmax(sim)
    answer_paragraph = paragraphs[idx_answer_paragraph]
    return answer_paragraph

def get_all_paragraphs(example, paragraph_len):
    """
    Later: implement sliding window?
    """
    n_texts = len(example['wiki_text'])
    all_paragraphs = []
    for i in range(n_texts):
        text_lower = example['wiki_text'][i].lower()
        text_no_space = re.sub(r'[\t\n\r]', ' ', text_lower)
        text_no_space = re.sub(r'[  ]', ' ', text_no_space)
        words = re.sub(r'[^\w ]', '', text_no_space).split(' ')
        words = [word for word in words if word]
        paragraphs = [words[i:(i+paragraph_len)] for i in range(0, len(words), paragraph_len)]
        paragraphs = [example['wiki_title'][i].lower() + ' ' + ' '.join(paragraph) for paragraph in paragraphs]
        all_paragraphs += paragraphs 
    return all_paragraphs
    
    
def get_tfidf_similarity(questions, paragraphs):
    vectorizer = TfidfVectorizer(lowercase = False)
    all_text = questions + paragraphs
    vectorizer.fit(all_text)
    similarity = vectorizer.transform(questions) * vectorizer.transform(paragraphs).T
    return similarity.todense()


# This one is not quite done. Used simpler version instead
def get_top_k(similarity, paragraphs, k):
    idxs = np.argpartition(similarity, -k)[:, -k:].tolist()
    top_k_paragraphs = [[paragraphs[idx] for idx in question_idxs] for question_idxs in idxs]
    return top_k_paragraphs

def find_best_paragraph_with_answer(top_k_paragraphs, answer):
    k = len(top_k_paragraphs)
    for i in range(k):
        tmp = top_k_paragraphs[i].find(answer)
        if tmp != -1:
            return top_k_paragraphs[i]
    return top_k_paragraphs[0]

#%%
data_preprocessed = preprocess_data(data, 128, False)

#%% Quick look at pre-processed data
print(data_preprocessed.features)
#%%

