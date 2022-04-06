# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:17:39 2022

@author: A. Lichtenberg
@purpose: The fundamental issue this script attempts to solve is that we have
data in two formats from label studio. The CONLL files have each token's text
and appropriate NER tag, but lack information on extracted relationships. The
JSON files have information on what entities are related, but none of the token
text or correct character indicies. The script assumes that the number/order/type
of entities extracted will be unique to a document and attempts to use that
as a key to pull together the required information into one training data set. 
"""


# Importing packages 
import os 
import json 
import pandas as pd

# Read in jsons
json_filepath = r'../data/tagged_entities'
input_jsons = []
for filename in os.listdir(json_filepath):
    with open(os.path.join(json_filepath,filename), 'r') as f:
        data=f.read()
        j = json.loads(data)
        input_jsons.append(j)

# Creating lists for entities and relationships from jsons
entities = []
relationships = []

# Each tagger had their own json
for tagger in input_jsons:
    # Iterate through docs annotated by each tagger
    for doc in tagger:
        file = doc['data']['text']
        #doc_ents = doc['annotations'][0]['result']

        for e in doc['annotations'][0]['result']:
            # only two options in type, labels and relation
            if e['type'] == 'labels':
                tup = (e['id'], file, e['value']['labels'][0], e['value']['start'], e['value']['end'])
                entities.append(tup)
            else:
                tup = (e['from_id'], e['to_id'])
                relationships.append(tup)

            
# Read in CONLL data
conll_filepath = r'..\data\conll_labels'
text = str()

for filename in os.listdir(conll_filepath):
    with open(os.path.join(conll_filepath,filename), 'r', encoding = 'utf-8') as f:
        text += f.read()


#get from conll to dataset format   
span_counter = 0
dict_list = [] 
span_dict = {'id':span_counter,
            'ner_tags': [],
            'tokens': []} 

for line in text.splitlines():
    # a useless first line to throw away
    if line.startswith('-DOCSTART-'):
        continue

    # Line breaks are present between documents in conll files
    # If at a line break, append the tags and entities from the previous doc
    elif len(line) == 0:
        dict_list.append(span_dict)
        span_counter += 1
        span_dict = {'id':span_counter,
                    'ner_tags': [],
                    'tokens': []} 
        
    else:
        token, tag = line.split(' -X- _ ') #standard text between tag and token
        
        # Accounting for punctuation that was included in the file
        s_punctuation = ['(', '"']
        e_punctuation = [',', ';', ':', '"', ')', '.', '!', '?']
        #sent_end_punct = ['.', '!', '?']
        
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
            
        else:
            span_dict['ner_tags'].append(tag)
            span_dict['tokens'].append(token)
            
# TB had data exported in a different format   
# This format had everything we needed but was not reproducable with
# anyone else's instance of label studio              
tb_filepath = r'..\data\ls_json_fixed\tb_data.json' 
with open(tb_filepath, 'r', encoding = 'utf-8') as f:
    tb_json= f.read()
    tb_data = json.loads(tb_json)

tb_ents = []
file_counter = 0
for doc in tb_data:
    file = doc['file_upload']
    # TB data has each file twice- once with the full text but no file name (this one)
    if file.startswith('e2fcfd47'):
        full_text = doc['data']['text']
        # file name isn't useful, so add suffix to create a unique file
        file += ('_tb_' + str(file_counter))
        file_counter += 1
        for e in doc['annotations'][0]['result']:
            # only two options in type, labels and relation
            if (e['type'] == 'labels') and ('text' in e['value']):
                tup = (e['id'], file, e['value']['labels'][0], e['value']['start'], e['value']['end'], e['value']['text'])
                tb_ents.append(tup)
            elif e['type'] == 'labels':
                tup = (e['id'], file, e['value']['labels'][0], e['value']['start'], e['value']['end'])
                entities.append(tup)
            else:
                tup = (e['from_id'], e['to_id'])
                relationships.append(tup)    
    # The other instance of the file will have the original name of the file,
    # But not the text
    else:
        continue

tb_entity_df = pd.DataFrame(tb_ents, columns = ['id', 'ls_file', 'label', 'char_start', 'char_end', 'token'])
              
entity_df = pd.DataFrame(entities, columns = ['id', 'ls_file', 'label', 'char_start', 'char_end'])

docs = entity_df['ls_file'].unique()

# Pulling the unique sequence of entities tagged from each JSON doc
seq_dict = {}
for doc in docs:
    ent_list = []
    loop_df = entity_df.loc[entity_df['ls_file'] == doc]
    loop_df.sort_values(by='char_start', inplace=True, axis=0)
    for row in loop_df.itertuples():
        ent_list.append(row.label)
    seq_dict[str(ent_list).strip()] = doc

# Pulling the unique sequence of entities tagged from each CONLL doc
tag_to_token = {}

for d in dict_list:
    actual_text = []
    actual_tags = []  
    entities = d['ner_tags']
    tokens = d['tokens']
    # I only want tags that are something other than 0
    tags = [(i, e.split('-')) for i, e in enumerate(entities) if len(e) > 3]
    tag_text = str()
    for j,t in enumerate(tags):
        i = t[0]
        pre = t[1][0]
        tag = t[1][1]
        
        # Special case for end of the document
        if j == len(tags) - 1:
            if pre == 'I':
                tag_text += (' ' + tokens[i])
                actual_text.append(tag_text)
            else:
                actual_tags.append(tag)
                actual_text.append(tag_text)
                tag_text = str()
                tag_text += tokens[i]
                actual_text.append(tag_text)
                
        else:   
            if pre == 'I':
                tag_text += (' ' + tokens[i])
            # Special case for beginning of the document
            elif j==0:
                tag_text += tokens[i]
                actual_tags.append(tag)
            else:
                actual_tags.append(tag)
                actual_text.append(tag_text)
                tag_text = str()
                tag_text += tokens[i]
               
    tag_to_token[str(actual_tags).strip()] = actual_text

# Create a counter of sequences not found- need to track how many docs
# this didn't work on. 18 of the results are from TB's docs, which we pull in
# below, so only 2 docs actually get dropped in this process
seq_not_found = 0
doc_to_tokens = {} 
for t in tag_to_token.keys():
    if t.strip() in seq_dict:
        doc_to_tokens[seq_dict[t.strip()]] = tag_to_token[t.strip()]
    else:
        seq_not_found += 1

          
    
df_list = []
docs_for_final = [doc for doc in docs if doc in doc_to_tokens.keys()]
for doc in docs_for_final:
    l_df = entity_df.loc[entity_df['ls_file'] == doc].copy()
    doc_tokens = doc_to_tokens[doc]
    l_df.sort_values(by='char_start', inplace = True)
    l_df['token'] = doc_tokens
    df_list.append(l_df)
    
final_df = pd.concat(df_list)
final_df = pd.concat([final_df, tb_entity_df])

relation_df = pd.DataFrame(relationships, columns = ['to_id', 'from_id'])        
to_df = relation_df.merge(final_df, left_on = 'to_id', right_on ='id', how = 'left')
to_df.rename(columns = {'label':'to_label', 'token':'to_token'}, inplace = True)
to_df.drop('id', axis = 1, inplace = True)
relation_df_final = to_df.merge(final_df, left_on = 'from_id', right_on ='id', how = 'left')
relation_df_final.drop(['id','ls_file_y'], axis = 1, inplace = True)
relation_df_final.rename(columns = {'ls_file_x':'ls_file','label':'from_label','token':'from_token'}, inplace = True)
relation_df_final.dropna(thresh = 3, inplace= True)
relation_df_final['to_label'] = relation_df_final['to_label'].apply(lambda x: x.replace('Credit Card', 'Card Name'))
relation_df_final['from_label'] = relation_df_final['from_label'].apply(lambda x: x.replace('Credit Card', 'Card Name'))

labels_to_remove = ['Termination Date','Execution Date','Credit Limit', 'Annual Fee', 'Payment Due Date', 'PER']

relation_df_final = relation_df_final[~relation_df_final['to_label'].isin(labels_to_remove)]
relation_df_final = relation_df_final[~relation_df_final['from_label'].isin(labels_to_remove)]

relation_df_final.to_csv('../data/all_relations.csv', index = False)
