# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 21:05:55 2022

@author: A. Lichtenberg
@purpose: Converts CONLL output into a huggingface dataset to be used for
transfer learning
"""

# Read in data
import os 

conll_filepath = r'..\data\conll_labels'
text = str()

for filename in os.listdir(conll_filepath):
    with open(os.path.join(conll_filepath,filename), 'r', encoding = 'utf-8') as f:
        text += f.read()



#get from conll to dataset format   
span_counter = 0
dict_list = [] 
span_id = span_counter
span_dict = {'id':span_id,
            'ner_tags': [],
            'tokens': []} 

for line in text.splitlines():
    if line.startswith('-DOCSTART-'):
        continue

    
    elif len(line) == 0:
        dict_list.append(span_dict)
        span_counter += 1
        span_id = span_counter
        span_dict = {'id':span_id,
                    'ner_tags': [],
                    'tokens': []} 
        
    else:
        token, tag = line.split(' -X- _ ')
        
        s_punctuation = ['(', '"']
        e_punctuation = [',', ';', ':', '"', ')']
        sent_end_punct = ['.', '!', '?']
        
        #check for extra punctuation at start of token
        if token.startswith(tuple(s_punctuation)):
            span_dict['ner_tags'].append('O')
            span_dict['tokens'].append(token[0])
            token = token[1:]
        
        #check for extra punctuation at end of token
        if token.endswith(tuple(e_punctuation)):
            span_dict['ner_tags'].append(tag)
            span_dict['tokens'].append(token[:-1])
            
            span_dict['ner_tags'].append('O')
            span_dict['tokens'].append(token[-1])
            
        # Create a new span if this is the end of a sentence    
        elif token.endswith(tuple(sent_end_punct)):
            span_dict['ner_tags'].append(tag)
            span_dict['tokens'].append(token[:-1])
            
            span_dict['ner_tags'].append('O')
            span_dict['tokens'].append(token[-1])
            
            dict_list.append(span_dict)
            span_counter += 1
            span_id = span_counter
            span_dict = {'id':span_id,
                        'ner_tags': [],
                        'tokens': []} 
                                 
        else:
            span_dict['ner_tags'].append(tag)
            span_dict['tokens'].append(token)

#do tokenization 


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-uncased-contracts")

e_type_dict = {
                "O":0,       # Outside of a named entity
                "B-MISC":1,  # Beginning of a miscellaneous entity right after another miscellaneous entity
                "I-MISC":2,  # Miscellaneous entity
                "B-PER":3,   # Beginning of a person's name right after another person's name
                "I-PER":4,   # Person's name
                "B-ORG":5,   # Beginning of an organisation right after another organisation
                "I-ORG":6,   # Organisation
                "B-LOC":7,   # Beginning of a location right after another location
                "I-LOC":8,    # Location
                "B-Issuer":9,
                "I-Issuer":10,
                "B-Card Name":11,
                "I-Card Name":12,
                "B-Jurisdiction":13,
                "I-Jurisdiction":14,
                "B-Interest Rate":15,
                "I-Interest Rate":16,
                "B-Credit Card":11,
                "I-Credit Card":12
                }

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:  # Set the special tokens to -100.

            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])

            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels

    return tokenized_inputs


from datasets import Dataset

for d in dict_list:
    d['ner_tag_int'] = []
    for tag in d['ner_tags']:
        if tag in e_type_dict.keys():
            d['ner_tag_int'].append(e_type_dict[tag]) 
        else:
            d['ner_tag_int'].append(0) 
            
custom_dict = {}
custom_dict['train'] = {}
custom_dict['train']['id'] = [d['id'] for d in dict_list]
custom_dict['train']['ner_tags'] = [d['ner_tag_int'] for d in dict_list]
custom_dict['train']['tokens'] = [d['tokens'] for d in dict_list]
dataset = Dataset.from_dict(custom_dict['train'])
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched = True, batch_size=1000)
tokenized_dataset.features[f'ner_tags'].feature.names = e_type_dict

#dataset_json = tokenized_dataset.to_json(r'../data/hf_datasets/TG_dataset_final.json')

dataset.features[f'ner_tags'].feature.names = e_type_dict
#dataset.to_json(r'../data/hf_datasets/TG_dataset_w_IR_not_tokenized.json')


### Create Visualizations 
# Only counting tags that were the beginning of a tag, cut out the 'B-'
# that prepends all tags
all_tags = [tag[2:] for d in dict_list for tag in d['ner_tags'] if tag.startswith('B-')]
all_tags = list(map(lambda x: x.replace('Credit Card', 'Card Name'), all_tags))
import pandas as pd
import seaborn as sns
entity_df = pd.DataFrame(all_tags, columns = ['label'])

count_plot = sns.countplot(x = 'label', data = entity_df, order = entity_df['label'].value_counts().index,
              color = 'black')
count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation = 45, horizontalalignment='right')
count_plot.set(xlabel = 'Label',
               ylabel = 'Count',
               title = "Count of Entity Types Manually Tagged")
