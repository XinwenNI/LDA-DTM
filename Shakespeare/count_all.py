#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:13:58 2018

@author: verani
"""

#import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import os
#import pysentiment as ps
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
#from nltk.corpus import stopwords
from os import path
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS


d = os.getcwd()

raw_text= open(path.join(d, 'alllines.txt'), encoding = "utf8").read()
raw_text= raw_text.replace("\n"," ")

cleantextprep = str(raw_text)


# keep only letters, numbers and whitespace
expression = "[^a-zA-Z0-9 ]" 
cleantextCAP = re.sub(expression, '', cleantextprep) # apply regex
cleantext = cleantextCAP.lower() # lower case 



text_file = open("Output_all.txt", "w")
text_file.write(str(cleantext))
text_file.close()



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
    
    
