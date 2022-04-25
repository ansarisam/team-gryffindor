# George Mason University DAEN690 Spring 2022
## Team Gryffindor

This repository represents Team Gryffindor's contribution to the class' problem: extract entities and relationships from legal documents.
This repository contains both the "produciton" function/classes and the code that got us there- everything from OCR/text extraction of PDFs, extensive data cleaning and model training, to the final result. 

The actual production portion of this code is the `getNER` module/function, which is implemented in the `gryffindorner` package. `getNER` uses custom trained NER and relationship extraction transformer models to extract entities and relationships found in credit card agreements. `getNER` takes one argument, a string that you wish to parse for entities and relationships. There is an optional argument called `relation_threshold` that allows you to set the threshold for a relation prediction. In entity sparse texts you may see improved performance from a lower threshold, while in entity rich texts you may want to increase the threshold. An example of how to implement the code is below:

```
from gryffindorner.getNER import getNER

example = 'lorem ipsum'

ents, rels = getNER(example)
```

## Entity Extraction 
`getNER` calls our [custom-trained NER model](https://huggingface.co/timhbach/Team-Gryffindor-distilbert-base-finetuned-NER-creditcardcontract-100epoch) which tags a number of unique entity types: `Credit Card Name`, `Issuer`, `Jurisdiction`, and `Interest Rate`. Additionally, standard entity types like `PER`, `ORG`, and `LOC` are tagged. 

## Relation Extraction
`getNER` also calls a custom trained model to classify relationships between extracted entities. Relationship types identified by the model are: `Card Interest Rate`, `Located At`, `Issued By`, `Jurisdiction In`, and a generic `Related`. This model can be installed through the `rel_pipeline` whl file. While you can call the model whenever you'd like after installing, you do not need to specially call it for it to funtion in `getNER`- the function is hardcoded to look for that spaCy model installed on your machine. 

## Outputs
`getNER` returns two lists: `ents` and `rels`. `ents` is a list of dictionaries with the following keys: `name`, a string representing the entity's name, `label`, a string for how it was classified, and `start`, an integer corresponding to the start character of `name`. `rels` is also a list of dictionaries and contains the following keys: `from`, a string showing the `name` of the recieving end of the relationship, `to`, a string showing the `name` of the originating end of the relationship, `label`, a string showing the type of relationship extracted, and `p`, a float representing the probability of that relationship existing, according to the model. 
