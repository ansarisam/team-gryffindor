# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 13:16:30 2022

@author: Alex Lichtenberg
"""
import os 
import json 
import pandas as pd

filepath = r'../data/tagged_entities'
input_jsons = []
for filename in os.listdir(filepath):
    with open(os.path.join(filepath,filename), 'r') as f:
        data=f.read()
        j = json.loads(data)
        input_jsons.append(j)
        
entities = []
relationships = []

for tagger in input_jsons:
    for doc in tagger:
        file = doc['data']['text']
        doc_ents = doc['annotations'][0]['result']

        for e in doc_ents:
            if e['type'] == 'labels':
                tup = (e['id'], file, e['value']['labels'][0], e['value']['start'], e['value']['end'])
                entities.append(tup)
            else:
                tup = (e['from_id'], e['to_id'])
                relationships.append(tup)


entity_df = pd.DataFrame(entities, columns = ['id', 'ls_file', 'label', 'char_start', 'char_end'])

entity_df[['userinfo','bank','contract']] = entity_df['ls_file'].str.split('-', n = 2, expand=True)
entity_df['clean_file'] = entity_df['bank'] + '-' + entity_df['contract']
entity_df['clean_file'] = entity_df['clean_file'].str.replace('_', ' ')

import glob
unique_contracts = entity_df['clean_file'].unique()
bad_guesses = []
data_location = r'../data/cleaned_texts'
corpus_dict = {}
for contract in unique_contracts:
    
    guess = r'../data/cleaned_texts/{0}*.txt'.format(contract[0:len(contract)-12])
    poss_match = glob.glob(guess)
    if len(poss_match) == 1:
        with open(poss_match[0], 'r', encoding = 'utf-8') as f:
            # repl 0xc2 w whitespace
            data = f.read()
            corpus_dict[contract] = data
    else:
        print("Not good guess")
        bad_guesses.append(contract)

