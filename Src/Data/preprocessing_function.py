# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:36:50 2020

@author: elisa
"""
# Import required 
from pathlib import Path
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
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
    
    return(out_data)

def get_paragraph_with_answer(example, paragraph_len):
    """
    Return paragraph of paragraph_len from example['wiki_text'] with highest 
    similary to question + answer
    """
    paragraphs = get_all_paragraphs(example, paragraph_len)
    target = example['question'] + example['answer'] 
    sim = get_tfidf_similarity([target], paragraphs)
    idx_answer_paragraph = np.argmax(sim)
    answer_paragraph = paragraphs[idx_answer_paragraph]
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
        paragraphs = [example['wiki_title'][i].lower() + \
                      ' ' + ' '.join(paragraph) for paragraph in paragraphs]
        all_paragraphs += paragraphs 
    return all_paragraphs
    
    
def get_tfidf_similarity(questions, paragraphs):
    """
    Returns a similarity matrix based on the distance in the tf-idf space

    Parameters
    ----------
    questions : list of strings
        Lists of all questions.
    paragraphs : list of strings
        Lists of all paragraphs
    
    Returns :
    -------
        Similarity matrix of dimension  
        (number of questions, number of paragraphs). 
    """
    vectorizer = TfidfVectorizer(lowercase = False)
    all_text = questions + paragraphs
    vectorizer.fit(all_text)
    similarity = vectorizer.transform(questions) * vectorizer.transform(paragraphs).T
    return similarity.todense()