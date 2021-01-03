# Open Domain Question Answering - Deep Learning Project

> This repository contains our exam project in the course 02456 Deep Learning Fall 2020 at DTU. **Note that the notebook containing the main results can be found in [Presentation of Main Results.ipynb](https://github.com/elisabethzinck/deep_learning_project/blob/master/Presentation%20of%20Main%20Results.ipynb).** 

## Structure

The [Src](https://github.com/elisabethzinck/deep_learning_project/tree/master/Src)-folder contains three folders:

* The [Data](https://github.com/elisabethzinck/deep_learning_project/tree/master/Src/Data)-folder in which data is loaded and preprocessed.
* The [Models](https://github.com/elisabethzinck/deep_learning_project/tree/master/Src/Models)-folder in which all the models that have been developed, can be found.
* The [Visualization](https://github.com/elisabethzinck/deep_learning_project/tree/master/Src/Visualization)-folder in which scripts that evaluate the performance can be found. Furthermore, the file [predictions_masked.py](https://github.com/elisabethzinck/deep_learning_project/blob/master/Src/Visualization/predictions_masked.py) that contains experiments with predicting masked words, can be found.

## Usage 

### Importing Modules

First the necessary modules are installed and imported, the drive is mounted, and torch is setup to use the GPU

```python
# Install modules from Hugging face
! pip install datasets;
! pip install transformers;

# Import used modules
import torch
import numpy as np
import matplotlib.pyplot as plt

# Setup Google Drive, where the raw data is located
import sys
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
sys.path.append(os.path.join('/content/drive/My Drive/deep_learning_project'))

# Setup GPU
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
```

### Loading Data

The data comes from the dataset TriviaQA from Hugging Face. Below two small samples of training and validation data are loaded.

```python
from datasets import load_from_disk

train_path = 'drive/My Drive/deep_learning_project/train_mini'
val_path = 'drive/My Drive/deep_learning_project/validation_mini'

train_raw = load_from_disk(train_path)
validation_raw = load_from_disk(val_path)
```

An example of what the data looks like can be seen below:

```python
train_raw[1]
```

```python
Out[]:
        {'answer': {'aliases': ['Park Grove (1895)',
        'York UA',
        'Yorkish',
        'UN/LOCODE:GBYRK',
        'York, UK',
        'Eoforwic',
        'Park Grove School',
        'York Ham',
        'The weather in York',
        'City of York',
        'York, England',
        'York, Yorkshire',
        'York ham',
        'County Borough of York',
        'YORK',
        'Eoferwic',
        'Park Grove Primary School',
        'York, North Yorkshire',
        'Yoisk',
        'York',
        'York (England)'],
        'matched_wiki_entity_name': '',
        'normalized_aliases': ['york yorkshire',
        'eoferwic',
        'park grove primary school',
        'park grove school',
        'weather in york',
        'park grove 1895',
        'eoforwic',
        'county borough of york',
        'york uk',
        'un locode gbyrk',
        'city of york',
        'york england',
        'york ua',
        'york ham',
        'york',
        'yorkish',
        'yoisk',
        'york north yorkshire'],
        'normalized_matched_wiki_entity_name': '',
        'normalized_value': 'york',
        'type': 'WikipediaEntity',
        'value': 'York'},
        'entity_pages': {'doc_source': ['TagMe', 'TagMe'],
        'filename': ['England.txt', 'Judi_Dench.txt'],
        'title': ['England', 'Judi Dench'],
        'wiki_context': ['England is a country that is part of the United Kingdom.   It shares 
        'land borders with Scotland to the north and Wales to the west. The Irish Sea lies' 
        'northwest of England and the Celtic Sea lies to the southwest. England is separated' 
        'from continental Europe by the North Sea to the east and the English Channel to the' 
        'south. The country covers much of the central and southern part of the island of Great' 
        'Britain, which lies in the North Atlantic; and includes over 100 smaller islands such' 
        'as the Isles of Scilly, and the Isle of Wight.\n\nThe area now called England was first' 
        'inhabited by modern humans during the Upper Palaeolithic period, but takes its name from' 
        'the Angles, one of the Germanic tribes who settled during the 5th and 6th centuries.' 
        'England became a unified state in the 10th century, and since the Age of Discovery, which' 
        'began during the 15th century, has had a significant cultural and legal impact on the' 
        'wider world.  The English language, the Anglican Church, and English law – the basis for' 
        'the common law legal systems of many other countries around the world – developed in' 
        'England, and the country\'s parliamentary system of government has been widely adopted by' 
        'other nations.  The Industrial Revolution began in 18th-century England, transforming its' 
        'society into the world\'s first industrialised nation. \n\nEngland\'s terrain mostly' 
        'comprises low hills and plains, especially in central and southern England. However,' 
        'there are uplands in the north (for example, the mountainous Lake District, Pennines,' 
        'and Yorkshire Dales) and in the south west (for example, Dartmoor and the Cotswolds).' 
        'The capital is London, which is the largest metropolitan area in both the United Kingdom' 
        'and the European Union.According to the European Statistical Agency, London is the largest' 
        'Larger Urban Zone in the EU, a measure of metropolitan area which comprises a city\'s' 
        'urban core as well as ...']},
        'question': 'Where in England was Dame Judi Dench born?',
        'question_id': 'tc_3',
        'question_source': 'http://www.triviacountry.com/'}

```

### Pre-process Data

The data is pre-processed before using in the Dense Passage Retrieval. The process is described in section *3.2 Wikipedia Data Pre-processing* in the article

```python
train_processed = preprocess_data(train_raw)
validation_processed = preprocess_data(validation_raw)
```

### Training Models

```python
from math import ceil
from transformers import AutoTokenizer, BertModel, AdamW

# Define data parameters
batch_size = 16

# Train
n_sample_train = 1024 
n_batches_train = ceil(n_sample_train/batch_size)
# Validation
n_sample_validation = 128 
n_batches_validation = ceil(n_sample_validation/batch_size)

# Subset data
train_data = train_processed.select(range(n_sample_train))
validation_data = validation_processed.select(range(n_sample_validation))

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', padding = True)
train_data = train_data.map(lambda example: {
    'Q_input_ids': tokenizer(example['question'], padding = 'max_length')['input_ids'],
    'Q_attention_mask': tokenizer(example['question'], padding = 'max_length')['attention_mask'],
    'Q_token_type_ids': tokenizer(example['question'], padding = 'max_length')['token_type_ids'],
    'P_input_ids': tokenizer(example['paragraph'], padding = 'max_length')['input_ids'],
    'P_attention_mask': tokenizer(example['paragraph'], padding = 'max_length')['attention_mask'],
    'P_token_type_ids': tokenizer(example['paragraph'], padding = 'max_length')['token_type_ids']},
    batched = True, batch_size= batch_size)

validation_data = validation_data.map(lambda example: {
    'Q_input_ids': tokenizer(example['question'], padding = 'max_length')['input_ids'],
    'Q_attention_mask': tokenizer(example['question'], padding = 'max_length')['attention_mask'],
    'Q_token_type_ids': tokenizer(example['question'], padding = 'max_length')['token_type_ids'],
    'P_input_ids': tokenizer(example['paragraph'], padding = 'max_length')['input_ids'],
    'P_attention_mask': tokenizer(example['paragraph'], padding = 'max_length')['attention_mask'],
    'P_token_type_ids': tokenizer(example['paragraph'], padding = 'max_length')['token_type_ids']},
    batched = True, batch_size= batch_size)

#%% Change to pytorch format. 
train_data.set_format(type = 'torch', 
                        columns = ['Q_input_ids', 'Q_attention_mask', 'Q_token_type_ids',
                                   'P_input_ids', 'P_attention_mask', 'P_token_type_ids'])

validation_data.set_format(type = 'torch', 
                        columns = ['Q_input_ids', 'Q_attention_mask', 'Q_token_type_ids',
                                   'P_input_ids', 'P_attention_mask', 'P_token_type_ids'])
```

```python
# Get pre-trained model from hugging face
model = BertModel.from_pretrained('bert-base-uncased')

# Move model to cuda to train there
model.to(device)

optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = lr) 
# filter object works as a generator 
```

The model can now be passed into a training loop, see [Presentation of Main Results.ipynb](https://github.com/elisabethzinck/deep_learning_project/blob/master/Presentation%20of%20Main%20Results.ipynb) for more details.

```python
#%% Saving model
model_path = 'drive/some path...'
model.save_pretrained(model_path)
```

### Evaluate Performance

Validation data is now used in order to evaluate performance.

```python

n_evaluation = 1024
k_list = [i+1 for i in range(100)]

validation_data = validation_processed.select(range(n_evaluation))

#%% Define inputs
questions_BERT = validation_data['question']
question_ids = validation_data['question_id']
paragraphs_BERT = validation_data['paragraph']
#%%
questions = [entry[0] + ' ' + entry[1] for entry in questions_BERT]
paragraphs = [entry[0] + ' ' + entry[1] for entry in paragraphs_BERT]
```

```python
sim_tfidf = get_tfidf_similarity(questions, paragraphs)
```

```python
# Default base model (no finetuning)
sim_BERT = get_BERT_similarity(validation_data, concatenate_9thTo11thLayer = True)

# Fine-tuned model
sim_BERT_finetuned = get_BERT_similarity(validation_data, finetuned = True, model_name = 'presentation_notebook_model', concatenate_9thTo11thLayer = True)
```

```python
#%% Get accuracies for a range of ks
acc_tfidf = get_accuracy_vector(k_list, sim_tfidf, question_ids, question_ids)
acc_bert = get_accuracy_vector(k_list, sim_BERT, question_ids, question_ids)
acc_bert_finedtuned = get_accuracy_vector(k_list, sim_BERT_finetuned, question_ids, question_ids)
acc_random = get_random_accuracy(k_list, n_evaluation)
```

## API

### Methods for pre-processing data

* `preprocess_data(data, paragraph_len = 128, remove_search_results = False)`:

Pre-processes the data by selecting relevant columns and finding the paragraph with the correct answer.

* `get_tfidf_similarity(questions, paragraphs)`:

Returns a similarity matrix based on the distance in the tf-idf space.

* `get_paragraph_with_answer(example, paragraph_len)`:

Return paragraph of paragraph_len from example['wiki_text'] with highest similary to question + answer.

* `get_all_paragraphs(example, paragraph_len)`:

Splits all wiki_texts of example into paragraphs of paragraph_len.

* `append_Q_token(example)`

Appends P or Q to the paragraph or Question.

### Methods for training models

* `get_loss(sim)`:

Returns negative loss.

### Methods for evaluating performance

* `get_top_k(similarity, question_ids, paragraph_ids, k)`:

Return the top k documents for based on similarity matrix.

* `get_accuracy(top_k)`:

Returns accuracy. top_k is a dict as returned by get_top_k().

* `get_tfidf_similarity(questions, paragraphs)`:

Returns a similarity matrix based on the distance in the tf-idf space.

* `get_random_accuracy(k_list, n)`:

Returns random accuracy.

* `get_accuracy_vector(k_list, sim, question_ids, paragraph_ids)`:

Returns accuracy vector.

* `get_BERT_similarity(data, finetuned = False, model_name = 'bert-base-uncased', concatenate_9thTo11thLayer = False)`:

Returns a similarity matrix based on the distance in the BERT encoded space.