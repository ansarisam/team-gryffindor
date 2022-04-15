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

