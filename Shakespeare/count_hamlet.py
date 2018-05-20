#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:19:07 2018

@author: verani
"""

import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import os
import pysentiment as ps
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from os import path


d = os.getcwd()

raw_text= open(path.join(d, 'hamlet.txt'), encoding = "utf8").read()
#excludes = {"ham","thy","hor","pol","laer","laertes","horatio","polonius","thou","ophelia","exeunt","ber","dost","ho"}

text_file= raw_text

cleantextprep = str(text_file)


# keep only letters, numbers and whitespace
expression = "[^a-zA-Z0-9 ]" 
cleantextCAP = re.sub(expression, '', cleantextprep) # apply regex
cleantext = cleantextCAP.lower() # lower case 


raw_words=nltk.word_tokenize(cleantext)
#raw1=cleantext.split()

porter_stemmer = PorterStemmer()
words_stem = [porter_stemmer.stem(raw_words) for raw_words in raw_words]


wordnet_lematizer = WordNetLemmatizer()
words1 = [wordnet_lematizer.lemmatize(raw_words) for raw_words in raw_words]

#words  = cleantext.split()
words = [word for word in words1 if word not in stopwords.words('english')]

counts = {}
for word in words:			
    counts[word] = counts.get(word,0) + 1
#for word in excludes:
#    del(counts[word])    
    
items = list(counts.items())
items.sort(key=lambda x:x[1], reverse=True) 
for i in range(10):
    word, count = items[i]
    print ("{0:<10}{1:>5}".format(word, count))