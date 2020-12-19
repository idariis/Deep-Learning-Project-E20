# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:36:50 2020

@author: elisa
"""
# Imports required to run functions
from nltk.tokenize import RegexpTokenizer
from Src.Models.DPR_functions import get_tfidf_similarity
import numpy as np

def preprocess_data(data, paragraph_len = 128, remove_search_results = False):
    """
    Pre-processes the data by selecting relevant columns and finding the 
    paragraph with the correct answer .

    Parameters
    ----------
    data : datasets.Dataset
        The raw wiki_qa data - maybe with feature 'search_results' removed for 
        more efficient storage.
    paragraph_len : int, optional
        Number of words in paragraph. The default is 128.
    remove_search_results : bool, optional
        Remove feature 'search_results' from the data. The default is False.

    Returns
    -------
        datasets.dataset
    Pre processed dataset with columns 
    ['answer', 'paragraph', 'question', 'question_id'].
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
    
    out_data = out_data.map(lambda example: {
        'question': append_Q_token(example)
        })
    return(out_data)

def get_paragraph_with_answer(example, paragraph_len):
    """
    Return paragraph of paragraph_len from example['wiki_text'] with highest 
    similary to question + answer
    """
    paragraphs = get_all_paragraphs(example, paragraph_len)
    
    # joining title and text without the '[P]' token 
    paragraphs_joined = [paragraphs[0][0][4:] + ' ' + paragraph[1] for paragraph in paragraphs]
    
    # similarities
    target = example['question'] + example['answer'] 
    sim = get_tfidf_similarity([target], paragraphs_joined) 
        
    # finding most similar containing the answer
    n_para = sim.size
    
    idxs = np.argsort(sim)[0][-n_para:]
    for p in range(len(idxs)):
        idx = idxs[-p]
        if example['answer'] in paragraphs[idx][1]:
            break 
    
    answer_paragraph = paragraphs[idx]
    
    return answer_paragraph


def get_all_paragraphs(example, paragraph_len):
    """
    Splits all wiki_texts of example into paragraphs of paragraph_len
    """
    n_texts = len(example['wiki_text'])
    all_paragraphs = []
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(n_texts):
        tokens = tokenizer.tokenize(example['wiki_text'][i].lower())
        paragraphs = [tokens[i:(i+paragraph_len)] \
                      for i in range(0, len(tokens), paragraph_len)]
        
        #Old paragraph for concatenating title with paragraph
        #paragraphs = [example['wiki_title'][i].lower() + \
        #              ' ' + ' '.join(paragraph) for paragraph in paragraphs]
        
        # Paragraph as list of title and text 
        paragraphs = [['[P] ' + example['wiki_title'][i].lower()] + [' '.join(paragraph)] for paragraph in paragraphs]
        
        all_paragraphs += paragraphs 
    return all_paragraphs
    
    
def append_Q_token(example):
    """
    Appends P or Q to the paragraph or Question
    """
    return(['[Q]'] + [example['question']])
   
    

    
