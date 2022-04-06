# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 21:05:55 2022

@author: A. Lichtenberg
@purpose: Converts CONLL output into a huggingface dataset to be used for
transfer learning
"""

# Read in data
import os 
import pandas as pd

relation_df = pd.read_csv(r'..\data\all_relations_with_span.csv', encoding = 'utf-8')

conll_filepath = r'..\data\conll_labels'
text = str()

for filename in os.listdir(conll_filepath):
    with open(os.path.join(conll_filepath,filename), 'r', encoding = 'utf-8') as f:
        text += f.read()



#get from conll to dataset format   
span_counter = 0
doc_counter = 0
dict_list = [] 
span_id = span_counter
span_dict = {'id':span_id,
             'doc_id':doc_counter,
            'ner_tags': [],
            'tokens': [],
            'full_text': str(),
            'token_starts': [],
            'token_ends': [],
            'rels': []} 

for line in text.splitlines():
    if line.startswith('-DOCSTART-'):
        continue

    
    elif len(line) == 0:
        dict_list.append(span_dict)
        span_counter += 1
        doc_counter += 1
        span_id = span_counter
        span_dict = {'id':span_id,
                     'doc_id':doc_counter,
                    'ner_tags': [],
                    'tokens': [],
                    'full_text': str(),
                    'token_starts': [],
                    'token_ends': [],
                    'rels': []} 
        
    else:
        token, tag = line.split(' -X- _ ')
        
        s_punctuation = ['(', '"']
        e_punctuation = [',', ';', ':', '"', ')']
        sent_end_punct = ['.', '!', '?']
        
        #check for extra punctuation at start of token
        if token.startswith(tuple(s_punctuation)):
            
            span_dict['ner_tags'].append('O')
            span_dict['tokens'].append(token[0])
            span_dict['token_starts'].append(len(span_dict['full_text']))
            span_dict['full_text'] += token[0]
            span_dict['token_ends'].append(len(span_dict['full_text']))
            token = token[1:]
            
        
        #check for extra punctuation at end of token
        if token.endswith(tuple(e_punctuation)):
            span_dict['ner_tags'].append(tag)
            span_dict['tokens'].append(token[:-1])
            span_dict['token_starts'].append(len(span_dict['full_text']))
            span_dict['full_text'] += token[:-1]
            span_dict['token_ends'].append(len(span_dict['full_text']))
            
            span_dict['ner_tags'].append('O')
            span_dict['tokens'].append(token[-1])
            span_dict['token_starts'].append(len(span_dict['full_text']))
            span_dict['full_text'] += token[-1]
            span_dict['token_ends'].append(len(span_dict['full_text']))
            span_dict['full_text'] += ' '
            
        # Create a new span if this is the end of a sentence    
        elif token.endswith(tuple(sent_end_punct)):
            span_dict['ner_tags'].append(tag)
            span_dict['tokens'].append(token[:-1])
            span_dict['token_starts'].append(len(span_dict['full_text']))
            span_dict['full_text'] += token[:-1]
            span_dict['token_ends'].append(len(span_dict['full_text']))
            
            span_dict['ner_tags'].append('O')
            span_dict['tokens'].append(token[-1])
            span_dict['token_starts'].append(len(span_dict['full_text']))
            span_dict['full_text'] += token[-1]
            span_dict['token_ends'].append(len(span_dict['full_text']))
            
            dict_list.append(span_dict)
            span_counter += 1
            span_id = span_counter
            span_dict = {'id':span_id,
                         'doc_id':doc_counter,
                        'ner_tags': [],
                        'tokens': [],
                        'full_text': str(),
                        'token_starts': [],
                        'token_ends': [],
                        'rels': []} 
                                 
        else:
            span_dict['ner_tags'].append(tag)
            span_dict['tokens'].append(token)
            span_dict['token_starts'].append(len(span_dict['full_text']))
            span_dict['full_text'] += token
            span_dict['token_ends'].append(len(span_dict['full_text']))
            span_dict['full_text'] += ' '


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
import re

for d in dict_list:
    d['ner_tag_int'] = []
    for tag in d['ner_tags']:
        if tag in e_type_dict.keys():
            d['ner_tag_int'].append(e_type_dict[tag]) 
        else:
            d['ner_tag_int'].append(0) 
    # Get full text
      
custom_dict = {}
custom_dict['train'] = {}
custom_dict['train']['id'] = [d['id'] for d in dict_list]
custom_dict['train']['ner_tags'] = [d['ner_tag_int'] for d in dict_list]
custom_dict['train']['tokens'] = [d['tokens'] for d in dict_list]

### Create HF Datasets (For NER)
dataset = Dataset.from_dict(custom_dict['train'])
#tokenized_dataset = dataset.map(tokenize_and_align_labels, batched = True, batch_size=1000)
#tokenized_dataset.features[f'ner_tags'].feature.names = e_type_dict

#dataset_json = tokenized_dataset.to_json(r'../data/hf_datasets/TG_dataset_final.json')

dataset.features[f'ner_tags'].feature.names = e_type_dict
#dataset.to_json(r'../data/hf_datasets/TG_dataset_w_IR_not_tokenized.json')

### Create Spacy Binaries (For Relationships)

rel_dict = {'Interest Rate-Card Name':'Interest Rate-Card Name',
            'Card Name-Interest Rate':'Interest Rate-Card Name',
            'LOC-ORG':'LOC-ORG',
            'ORG-LOC':'LOC-ORG',
            'Issuer-Card Name': 'Issuer-Card Name',
            'Card Name-Issuer': 'Issuer-Card Name',
            'LOC-Issuer': 'LOC-Issuer',
            'Issuer-LOC': 'LOC-Issuer',
            'ORG-Issuer': 'ORG-Issuer',
            'Issuer-ORG': 'ORG-Issuer',
            'ORG-Card Name': 'ORG-Card Name',
            'Card Name-ORG': 'ORG-Card Name',
            'Jurisdiction-ORG': 'ORG-Jurisdiction',
            'ORG-Jurisdiction':'ORG-Jurisdiction',
            'Card Name-Card Name' : 'Card Name-Card Name',
            'LOC-LOC':'LOC-LOC',
            'ORG-ORG':'ORG-ORG',
            'Interest Rate-Interest Rate': 'Interest Rate-Interest Rate'}

MAP_LABELS = {'Interest Rate-Card Name':'CARD INTEREST RATE',
            'LOC-ORG':'LOCATED AT',
            'Issuer-Card Name': 'ISSUED BY',
            'LOC-Issuer': 'LOCATED AT',
            'ORG-Issuer': 'RELATED',
            'ORG-Card Name': 'ISSUED BY',
            'ORG-Jurisdiction':'JURISDICTION IN'}

relation_df['relationship_type'] = relation_df['to_label'] + '-' + relation_df['from_label']
relation_df['relationship_type'] = relation_df['relationship_type'].apply(lambda x: x.replace(x, rel_dict[x]))
relation_df = relation_df.loc[relation_df['relationship_type'].isin(MAP_LABELS.keys())].copy()
relation_df['relationship_type'] = relation_df['relationship_type'].apply(lambda x: x.replace(x, MAP_LABELS[x]))

entity_df = pd.read_csv(r'..\data\all_entities.csv')
import more_itertools

for d in dict_list:
    l_df = relation_df.loc[((relation_df['span_x'] == d['id']) & (relation_df['span_y'] == d['id']))]
    e_df = entity_df.loc[entity_df['span'] == d['id']]
    e_lookup = {}
    for i, row in enumerate(e_df.itertuples()):
        e_lookup[row.id] = i
    for row in l_df.itertuples():
        to_order = e_lookup[row.to_id]
        from_order = e_lookup[row.from_id]
        d['rels'].append((to_order, from_order, row.relationship_type))

close_df = relation_df.loc[abs(relation_df['span_x'] - relation_df['span_y']) == 1]
for row in close_df.itertuples():
    from_span = row.span_y
    to_span = row.span_x
    from_dict = [d for d in dict_list if d['id'] == from_span][0]
    to_dict = [d for d in dict_list if d['id'] == to_span][0]
    to_token_starts = [s + len(from_dict['full_text']) + 1 for s in to_dict['token_starts']]
    to_token_ends = [s + len(from_dict['full_text']) + 1 for s in to_dict['token_ends']]
    
    combined_dict = {'id': from_span + to_span,
                 'doc_id':row.ls_file,
                 'ner_tags': from_dict['ner_tags'] + to_dict['ner_tags'],
                'full_text': from_dict['full_text'] + ' ' + to_dict['full_text'],
                'token_starts': from_dict['token_starts'] + to_token_starts,
                'token_ends': from_dict['token_ends'] + to_token_ends,
                'tokens': from_dict['tokens'] + to_dict['tokens'],
                'rels': []} 
    e_lookup = {}
    from_e_df = entity_df.loc[entity_df['span'] == from_dict['id']]
    to_e_df = entity_df.loc[entity_df['span'] == to_dict['id']]
    for i, r in enumerate(from_e_df.itertuples()):
        e_lookup[r.id] = i
    for i, r in enumerate(to_e_df.itertuples()):
        e_lookup[r.id] = i + len(from_e_df) - 1
    
    to_order = e_lookup[row.to_id]
    from_order = e_lookup[row.from_id]
    
    combined_dict['rels'].append((to_order, from_order, row.relationship_type))

    dict_list.append(combined_dict)
 
rel_dict_list = [d for d in dict_list if len(d['rels']) > 0]      

import spacy
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab

#Creating the corpus from the Label Studio annotations
#Doc.set_extension("rel", default={})
vocab = Vocab()

train_file=r'..\data\spacy_binaries\relations_0401_train.spacy'
dev_file=r'..\data\spacy_binaries\relations_0401_dev.spacy'
test_file=r'..\data\spacy_binaries\relations_0401_test.spacy'

spacy_docs = {"train": [], "dev": [], "test": []}
ids = {"train": set(), "dev": set(), "test": set()}
count_all = {"train": 0, "dev": 0, "test": 0}
count_pos = {"train": 0, "dev": 0, "test": 0}

nlp = spacy.load("en_core_web_sm")

for d in rel_dict_list:
    doc = nlp(d['full_text'])
    doc.ents = []
    entities = []
    
    ent_token_starts = []
    for i, t in enumerate(d['ner_tags']):
        if t.startswith('B'):
            s_start = d['token_starts'][i]
            s_label = t[2:]
            mid_token = True
            j = i
            if t.endswith('LOC'):
                if (d['ner_tags'][i+1] == 'O') and (d['ner_tags'][i+2] == 'I-LOC'):
                    s_end = d['token_ends'][i+2]
                    mid_token = False
            while mid_token == True:
                j += 1
                check_suf= d['ner_tags'][j][2:]
                check_pre= d['ner_tags'][j][:1]
                if check_suf == s_label and check_pre == 'I':
                    continue
                else:
                    mid_token = False
                    s_end = d['token_ends'][j-1]
            
            ent = doc.char_span(s_start,s_end,label = s_label)
            if ent:
                entities.append(ent)
            else:
                # in case of weird punctuation
                ent = doc.char_span(s_start,s_end+1,label = s_label)
                entities.append(ent)
                print('Added a char')
            ent_token_starts.append(ent.start)
                
    doc.ents = entities
    order_lookup = {}
    for i, ent in enumerate(doc.ents):
        order_lookup[i] = ent.start
    rels = {}   
    # needs to be entity spans, not all token starts@       
    for x1 in ent_token_starts:
        for x2 in ent_token_starts:
            rels[(x1, x2)] = {}
    for r in d['rels']:
        s,e,l = r
        
        s_ent = order_lookup[s]
        e_ent = order_lookup[e]

        if l not in rels[(s_ent, e_ent)]:
            rels[(s_ent, e_ent)][l] = 1.0
            

        if l not in rels[(e_ent, s_ent)]:
            rels[(e_ent, s_ent)][l] = 1.0
            
        for x1 in ent_token_starts:
            for x2 in ent_token_starts:
                for label in MAP_LABELS.values():
                    if label not in rels[(x1, x2)]:
                        
                        rels[(x1, x2)][label] = 0.0
        doc._.rel = rels
        
        # Each of these should be about 10% of docs
    if str(d['id']).endswith('4'):
        spacy_docs["dev"].append(doc)

    elif str(d['id']).endswith('6'):
        spacy_docs["test"].append(doc)
    else:
        spacy_docs["train"].append(doc)

docbin = DocBin(docs=spacy_docs["train"], store_user_data=True)
#docbin.to_disk(train_file)


docbin = DocBin(docs=spacy_docs["dev"], store_user_data=True)
#docbin.to_disk(dev_file)

docbin = DocBin(docs=spacy_docs["test"], store_user_data=True)
#docbin.to_disk(test_file)


# Visuals 

import seaborn as sns
ids = [int(str(d['id'])[-1:]) for d in rel_dict_list]
id_distplot = sns.distplot(ids)
id_distplot.set(xlabel = 'Character',
                ylabel = 'Density',
                title = "End Character of Doc IDs")


### Create Visualizations 
# Only counting tags that were the beginning of a tag, cut out the 'B-'
# that prepends all tags

# =============================================================================
# all_tags = [tag[2:] for d in dict_list for tag in d['ner_tags'] if tag.startswith('B-')]
# all_tags = list(map(lambda x: x.replace('Credit Card', 'Card Name'), all_tags))
# import pandas as pd
# import seaborn as sns
# entity_df = pd.DataFrame(all_tags, columns = ['label'])
# 
# count_plot = sns.countplot(x = 'label', data = entity_df, order = entity_df['label'].value_counts().index,
#               color = 'black')
# count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation = 45, horizontalalignment='right')
# count_plot.set(xlabel = 'Label',
#                ylabel = 'Count',
#                title = "Count of Entity Types Manually Tagged")
# =============================================================================