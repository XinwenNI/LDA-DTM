#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:43:54 2018

@author: verani
"""

import pandas as pd
import bs4
import urllib

# Further Analysis
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import os

from os import path


d = os.getcwd()

text_file = open(path.join(d, 'XMAS SONG.txt'), encoding = "utf8").read()
text_file1 = open(path.join(d, 'jingle bells.txt'), encoding = "utf8").read()
text_file2 = open(path.join(d, 'silent night.txt'), encoding = "utf8").read()
text_file_temp= text_file.replace("\n"," ")
text_file1_temp= text_file1.replace("\n"," ")
text_file2_temp= text_file2.replace("\n"," ")


# Convert to string
cleantextprep = str(text_file_temp)
cleantextprep1 = str(text_file1_temp)
cleantextprep2 = str(text_file2_temp)

# Regex cleaning
expression = "[^a-zA-Z0-9 ]" # keep only letters, numbers and whitespace
cleantextCAP = re.sub(expression, '', cleantextprep) # apply regex
cleantext = cleantextCAP.lower() # lower case 

cleantextCAP1 = re.sub(expression, '', cleantextprep1) # apply regex
cleantext1= cleantextCAP1.lower() # lower case 

cleantextCAP2 = re.sub(expression, '', cleantextprep2) # apply regex
cleantext2 = cleantextCAP2.lower() # lower case 

# Save dictionaries for wordcloud
text_file = open("Output_total.txt", "w")
text_file.write(str(cleantext))
text_file.close()

words=cleantext.split()
dict1 = {}
for word in words:			
    dict1[word] = dict1.get(word,0) + 1


# Filter Stopwords
keys = list(dict1)
import nltk
nltk.download('stopwords')
filtered_words = [word for word in keys if word not in stopwords.words('english')]
dict2 = dict((k, dict1[k]) for k in filtered_words if k in filtered_words)

#lst = [(value, key) for (key, value) in dict2.items()]
#lst.sort(reverse=True)
#
#top_word=lst[: 10 ]


# Resort in list
# Reconvert to dictionary
def SequenceSelection(dictionary, length, startindex = 0): # length is length of highest consecutive value vector
    
    # Test input
    lengthDict = len(dictionary)
    if length > lengthDict:
        return print("length is longer than dictionary length");
    else:
        d = dictionary
        items = [(v, k) for k, v in d.items()]
        items.sort()
        items.reverse()   
        itemsOut = [(k, v) for v, k in items]
    
        highest = itemsOut[startindex:startindex + length]
        dd = dict(highest)
        wanted_keys = dd.keys()
        dictshow = dict((k, d[k]) for k in wanted_keys if k in d)

        return dictshow;
    
dictshow = SequenceSelection(dictionary = dict2, length = 10, startindex = 0)


# Plot most frequent words
n = range(len(dictshow))
plt.bar(n, dictshow.values(), align='center')
plt.xticks(n, dictshow.keys())
plt.title("Most frequent Words")
plt.savefig("FrequentWords.png", transparent=True)

# Overview
overview =  SequenceSelection(dictionary = dict2, length = 400, startindex = 0)
nOverview = range(len(overview.keys()))
plt.bar(nOverview, overview.values(), color = "g", tick_label = "")
plt.title("Word Frequency Overview")
plt.xticks([])
#plt.savefig("overview.png")
plt.savefig("overview.png", transparent = True)
#plt.savefig('overview.png', transparent=True)



top_words=list(dictshow)
words1=cleantext1.split()
dict11 = {}
for word in words1:			
    dict11[word] = dict11.get(word,0) + 1

dict_S1={}
for word in top_words:
    if word in words1:
        dict_S1[word] = dict11[word]       
    else: 
       dict_S1[word] = dict11.get(word,0)

    
words2=cleantext2.split()
dict21 = {}
for word in words2:		
    dict21[word] = dict21.get(word,0) + 1

dict_S2={}
for word in top_words:
    if word in words2:
        dict_S2[word] = dict21[word]       
    else: 
       dict_S2[word] = dict21.get(word,0)









