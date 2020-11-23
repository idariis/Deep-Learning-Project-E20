# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:46:03 2020

@author: elisa
"""
#%%
from datasets import load_from_disk
from pathlib import Path
import inspect
from Src.Data.preprocessing_function import *

src_file_path = inspect.getfile(lambda: None)
deep_learning_dir = Path(src_file_path).parent.parent.parent

#%% Load data
data = load_from_disk(deep_learning_dir/'Data/Raw/')
#%%
data_preprocessed = preprocess_data(data)

#%% 
tfidf_sim = get_tfidf_similarity(data_preprocessed['question'], data_preprocessed['paragraph'])

#%%
def get_top_k(similarity, question_ids, paragraph_ids, k):
   """ 
   Get the top k paragraphs with highest similary with question + answer
   Currently does
   """
   n_questions = similarity.shape[0]
   #idxs = np.argpartition(similarity, -k)[:, -k:].tolist()
   #top_k_paragraphs = [[paragraphs[idx] for idx in question_idxs] for question_idxs in idxs]
   idxs = [np.argsort(similarity[row,:])[-k:][::-1] for row in range(n_questions)]
   out = {question_ids[i]:np.array(paragraph_ids)[idxs[i]] for i in range(n_questions)}
   return out

top_k = get_top_k(tfidf_sim, data_preprocessed['question_id'], data_preprocessed['question_id'], 5)

#%% Get accuracy
def get_accuracy(top_k):
    n_correct = [(question in paragraphs) for question, paragraphs in top_k.items()]
    accuracy = sum(n_correct)/len(top_k)*100
    return accuracy

acc_tfidf = get_accuracy(top_k)
print(acc_tfidf)

#%% functions for BERT
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM

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

#%% 
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


questions = data_preprocessed['question'][0:100]
question_ids = data_preprocessed['question_id'][0:100]
paragraphs = data_preprocessed['paragraph'][0:100]
sim_BERT = get_BERT_similarity(questions, paragraphs)
#%%
top_k_BERT = get_top_k(sim_BERT.numpy(), question_ids, question_ids, 5)
acc_BERT = get_accuracy(top_k_BERT)
print(acc_BERT)

#%% 
#def get_accuracy_random()


#%% Making loss function
def get_loss(sim):
    nll = -(torch.diagonal(sim) - torch.logsumexp(sim, dim = 1))
    return nll# return negative loss


