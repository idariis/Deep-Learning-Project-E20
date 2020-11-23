# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

@author: Amalie
"""

#%%
from datasets import load_dataset

dataset = load_dataset('trivia_qa', name = 'rc')

train = dataset['train']

tmp_data = train.select(range(11))
tmp_data = tmp_data.map(lambda example: {'wiki_text': example['entity_pages']['wiki_context'], 'answer': example['answer']['normalized_value']}, remove_columns=['search_results', 'question_source', 'answer', 'entity_pages'])

tmp_data = tmp_data.filter(lambda example: len(example['wiki_text']) == 1)

tmp_data = tmp_data.map(lambda example: {'wiki_text_lower': example['wiki_text'][0].lower()})

#%%
#import torch
#from transformers import BertModel, BertTokenizerFast

#tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

text1 = tmp_data[0]["wiki_text"][0][0:200]
text2 = tmp_data[0]["wiki_text"][0][201:400]

texts = [text1,text2]

#%%
#text = [text1]
#encoded_dict = tokenize_BERT(text)

#print(encoded_dict)

#hidden_layer = encoder_BERT(encoded_dict[0])



#%%
#encoded_dict = tokenizer(text1, padding = True)
encoded_dict = tokenize_BERT(texts)
print(encoded_dict)
#tokenizer.decode(encoded_dict["input_ids"])

tokens_tensor = torch.tensor(encoded_dict[0]["input_ids"])
segments_tensor = torch.tensor(encoded_dict[0]["token_type_ids"])
attention_tensor = torch.tensor([encoded_dict[0]["attention_mask"]])

#%%
model = BertModel.from_pretrained('bert-base-uncased')

model.eval()

with torch.no_grad():
    outputs = model(inputs_ids=tokens_tensor, attention_mask=attention_tensor, token_type_ids=segments_tensor)
    encoded_layers = outputs[0]
    
assert tuple(encoded_layers.shape) == (1, len(encoded_dict["input_ids"]), model.config.hidden_size)

#%%
print(encoded_layers.shape)


#%%
import torch
from transformers import BertTokenizerFast

def tokenize_BERT(text):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    encoded_dict = []
    for i in range(len(text)):
        encoded_dict.append(tokenizer(text[i], padding = True))
    return encoded_dict

#%%
from transformers import BertModel

def encoder_BERT(tokenized_text):
    #print(tokenized_text["input_ids"])
    tokens_tensor = torch.tensor(tokenized_text["input_ids"])
    segments_tensor = torch.tensor(tokenized_text["token_type_ids"])
    attention_tensor = torch.tensor([tokenized_text["attention_mask"]])
    
    model = BertModel.from_pretrained('bert-base-uncased')
    
    model.eval()

    with torch.no_grad():
        outputs = model(inputs_ids=tokens_tensor, attention_mask=attention_tensor, token_type_ids=segments_tensor)
        encoded_layers = outputs[0]
        
    assert tuple(encoded_layers.shape) == (1, len(encoded_dict["input_ids"]), model.config.hidden_size)
    
    return encoded_layers






