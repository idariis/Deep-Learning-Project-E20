# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:19:26 2020

@author: elisa
"""

# Imports required to run functions
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import BertTokenizerFast, BertModel


def get_top_k(similarity, question_ids, paragraph_ids, k):
    """
    Return the top k documents for based on similarity matrix

    Parameters
    ----------
    similarity : ndarray
        similarity matrix with shape (n_questions, n_paragraphs)
    question_ids : list of str
        List of the question ids
    paragraph_ids : list of str
        List of the paragraph ids (corresponding to the question id that the 
                                   paragraph belongs to)
    k : int
        Number or paragraphs to return

    Returns
    -------
    out : dict
        keys are the question ids, and each element contains list of the ids of the
        k nearest paragraphs
    """
    n_questions = similarity.shape[0]
    idxs = [np.argsort(similarity[row,:])[-k:][::-1] for row in range(n_questions)]
    out = {question_ids[i]:np.array(paragraph_ids)[idxs[i]] for i in range(n_questions)}
    return out

def get_accuracy(top_k):
    """ Returns accuracy. top_k is a dict as returned by get_top_k().  """
    n_correct = [(question in paragraphs) for question, paragraphs in top_k.items()]
    accuracy = sum(n_correct)/len(top_k)*100
    return accuracy


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
    return np.array(similarity.todense())

def get_BERT_similarity(questions, paragraphs):
    """
    Returns a similarity matrix based on the distance in the BERT encoded space

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
    E_Q = encoder_BERT(tokenize_BERT(questions))
    E_P = encoder_BERT(tokenize_BERT(paragraphs))
    sim = torch.matmul(E_Q, E_P.T)
    return sim


def tokenize_BERT(text):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return tokenizer(text, padding = True)

def encoder_BERT(tokenized_text):
    tokens_tensor = torch.tensor(tokenized_text["input_ids"])
    segments_tensor = torch.tensor(tokenized_text["token_type_ids"])
    attention_tensor = torch.tensor(tokenized_text["attention_mask"])
    
    model = BertModel.from_pretrained('bert-base-uncased')
    
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=tokens_tensor, attention_mask=attention_tensor, token_type_ids=segments_tensor)
        encoded_layers = outputs[0]
    
    return encoded_layers[:, 0, :]

def get_random_accuracy(k, n):
    # Still neds to be implemented
    pass