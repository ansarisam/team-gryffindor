# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:06:25 2022

@author: A. Lichtenberg
"""

from getNER import getNER

example = "First National Credit Union issues the Super Awesome Rewards Card with an introductory APR of 15.00%. Wisconsin residents note that there are special rules."

ents, rels = getNER(example, 0.45)