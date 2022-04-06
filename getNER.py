# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:13:37 2022

@author: A. Lichtenberg
"""
# Imports needed for NER
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
# Getting relationships with spaCy 
from spacy.tokens import DocBin, Doc 
from spacy.training.example import Example
from rel_pipe import make_relation_extractor, score_relations
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


def tokenization_fixer(doc, ent_start):
    toks = [t.idx for t in doc]
    for i, t in enumerate(toks):
       if ent_start > t:
           continue
       if ent_start == t:
           ent_end = toks[i+1]
           return ent_start, ent_end
       else:
           ent_start = toks[i-1]
           ent_end = toks[i]
           return ent_start, ent_end
       
def get_relationships(doc, threshold = 0.5):
    relations = []              
    for value, rel_dict in doc._.rel.items():        
            for e in doc.ents:
                for b in doc.ents:
                    if e.start == value[0] and b.start == value[1]:
                        #checker = rel_dict
                        max_v = 0
                        for k, v in rel_dict.items():
                            if v > max_v:
                                max_v = v
                                max_k = k
                        if max_v >= threshold:
                            rd = {'from':e.text,
                                  'to': b.text,
                                  'label': max_k,
                                  'p':max_v}
                            relations.append(rd)
    return relations

#example = "First National Credit Union issues the Super Awesome Rewards Card with an introductory APR of 15.00%"

def getNER(text, relation_threshold = 0.5):
    # Getting Entities with our custom huggingface model
    tokenizer = AutoTokenizer.from_pretrained("timhbach/Team-Gryffindor-bert-base-finetuned-NER-creditcardcontract")
    model = AutoModelForTokenClassification.from_pretrained("timhbach/Team-Gryffindor-bert-base-finetuned-NER-creditcardcontract")
    
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    
    ner_results = nlp(text)

    # Loading rel model from file
    rel_nlp = spacy.load("rel_component/TG-Relation-Model")
    
    # Take HF output and turn into useable spaCy format    
    doc = rel_nlp(text)
    doc.ents = []
    entities = []
    
    for i, d in enumerate(ner_results):
        if d['entity'].startswith('B'):
            label = d['entity'][2:]
            ent_start = d['start']
            mid_token = True
            j = i
            while mid_token == True:
                j += 1
                if j >= len(ner_results) - 1:
                    ent_end = d['start'] + len(d['word'])
                    mid_token = False
                else:
                    check_suf= ner_results[j]['entity'][2:]
                    check_pre = ner_results[j]['entity'][:1]
                    if check_suf == label and check_pre == 'I':
                        continue
                    else:
                        mid_token = False
                        ent_end = ner_results[j-1]['start'] + len(ner_results[j-1]['word'])
            
            ent = doc.char_span(ent_start,ent_end,label = label)
            if ent:
                entities.append(ent)
            else:
                ent_start, ent_end = tokenization_fixer(doc, ent_start)
                ent = doc.char_span(ent_start,ent_end,label = label)
                entities.append(ent)
    
    # get rid of any duplicates            
    doc.ents = set(entities)
            
    for name, proc in rel_nlp.pipeline:
        doc = proc(doc)
    
    #ent_labels = [e.label_ for e in doc.ents]
    #ents_for_output = [e.text for e in doc.ents]
    ents_for_output = []
    for ent in doc.ents:
         e_dict = {'name':ent.text,
                    'label': ent.label_,
                    'start':ent.start_char}
         ents_for_output.append(e_dict)
        
    
    relationships = get_relationships(doc, threshold = relation_threshold)
    
    return ents_for_output, relationships
    