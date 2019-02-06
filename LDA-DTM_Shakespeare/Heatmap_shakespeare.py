#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:08:48 2018
The heatmap of Shakespeare analysis is missing from the Quantlet. This code is try to generate that. 
@author: verani
"""
#please install these module before you run the code :
#!pip install matplotlib
#!pip install nltk
#!pip install pandas

import re
from nltk.corpus import stopwords
import os
from os import path
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

# Please change the working directory to your path!
#os.chdir("/Users/xinwenni/LDA-DTM/Shakespeare") 

def BasicCleanText(raw_text):
    cleantextprep = str(raw_text)
    
    expression = "[^a-zA-Z0-9 ]" # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep) # apply regex
    cleantext = cleantextCAP.lower() # lower case 
    
    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cleantext)

    # create English stop words list
    #en_stop = get_stop_words('en')
    stop = set(stopwords.words('english'))
    # remove stop words from tokens
    #stopped_tokens = [i for i in tokens if not i in en_stop]
    stopped_tokens = [i for i in tokens if not i in stop]
 
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # stem token
    texts_clean = [p_stemmer.stem(i) for i in stopped_tokens]
    return texts_clean;

d = os.getcwd()
#inport the texts
text_file = open(path.join(d, 'three_tragedy.txt'), encoding = "utf8").read()
text_hamlet = open(path.join(d, 'hamlet_lines.txt'), encoding = "utf8").read()
text_julius = open(path.join(d, 'Julius_Caesar_lines.txt'), encoding = "utf8").read()
text_Romeo = open(path.join(d, 'Romeo_and_Juliet_lines.txt'), encoding = "utf8").read()


Clean_threeTragedy = BasicCleanText(raw_text=text_file)
Clean_H= BasicCleanText(raw_text=text_hamlet)
Clean_J = BasicCleanText(raw_text=text_julius)
Clean_R = BasicCleanText(raw_text=text_Romeo)

text_file = open("Output_three_tragedy.txt", "w")
text_file.write(str(Clean_threeTragedy))
text_file.close()
    
def WordFrequency(words):

    dict1 = {}
    for word in words:			
         dict1[word] = dict1.get(word,0) + 1

    d = dict1
    items = [(v, k) for k, v in d.items()]
    items.sort()
    items.reverse()   
    itemsOut = [(k, v) for v, k in items]

    dd = dict(itemsOut)
    return dd;
      
Fre_threeTragedy = WordFrequency(words=Clean_threeTragedy)
Fre_h = WordFrequency(words=Clean_H)
Fre_j = WordFrequency(words=Clean_J)
Fre_r = WordFrequency(words=Clean_R)
    
    
#these 34 words are given as an example in the book from L.Borke & W. Haerdle
doc_34words="art   away  blood  day  dead  dear  death  eye  exit  fair  father  fear god good great heart heaven ill king live lie  like life love lord make man men must night sweet think time well"
words_34=doc_34words.split()
Tragedys=['Romeo and Juliet ', 'Hamlet', 'Julius Caesar']

dict_34words_all3={}
dict_34words_r={}
dict_34words_h={}
dict_34words_j={}
for word in words_34:
    dict_34words_all3[word]=Fre_threeTragedy[word]
    dict_34words_r[word]=Fre_r[word]
    dict_34words_h[word]=Fre_h[word]
    dict_34words_j[word]=Fre_j[word]

data_all3=pd.DataFrame.from_dict(dict_34words_all3, orient='index')
data_r=pd.DataFrame.from_dict(dict_34words_r, orient='index')
data_h=pd.DataFrame.from_dict(dict_34words_h, orient='index')
data_j=pd.DataFrame.from_dict(dict_34words_j, orient='index')


new_data=pd.DataFrame(index=words_34)
new_data['Romeo and Juliet']=data_r
new_data['Hamlet']=data_h
new_data['Julius Caesar']=data_j

data_heatmap=new_data.T

import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(figsize = (10, 1))
#cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
#sns.heatmap(data_heatmap, cmap = cmap, linewidths = 0.05, ax = ax)
sns.heatmap(data_heatmap,cmap='gray',linewidths = 0.05, ax = ax)

f.savefig('Tragedy_heatmap.jpg', bbox_inches='tight')


