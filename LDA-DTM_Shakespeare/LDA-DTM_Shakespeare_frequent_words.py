#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:15:06 2018

@author: verani
"""

# please install these modules before run this code : 
#!pip install matplotlib
#!pip install nltk

import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import os
from os import path
import nltk
nltk.download('stopwords')

# Please change the working directory to your path!
#os.chdir("/Users/xinwenni/LDA-DTM/Shakespeare") 

d = os.getcwd()

text_file = open(path.join(d, 'three_tragedy.txt'), encoding = "utf8").read()
text_hamlet = open(path.join(d, 'hamlet_lines.txt'), encoding = "utf8").read()
text_julius = open(path.join(d, 'Julius_Caesar_lines.txt'), encoding = "utf8").read()
text_Romeo = open(path.join(d, 'Romeo_and_Juliet_lines.txt'), encoding = "utf8").read()

#text_file_temp= text_file.replace("\n"," ")


# Convert to string
cleantextprep = str(text_file)
cleantextprep_h = str(text_hamlet)
cleantextprep_j = str(text_julius)
cleantextprep_r = str(text_Romeo)

# Regex cleaning
expression = "[^a-zA-Z0-9 ]" # keep only letters, numbers and whitespace
cleantextCAP = re.sub(expression, '', cleantextprep) # apply regex
cleantext = cleantextCAP.lower() # lower case 

cleantextCAP_h = re.sub(expression, '', cleantextprep_h) # apply regex
cleantext_h= cleantextCAP_h.lower() # lower case 
cleantextCAP_j = re.sub(expression, '', cleantextprep_j) # apply regex
cleantext_j= cleantextCAP_j.lower() # lower case 
cleantextCAP_r = re.sub(expression, '', cleantextprep_r) # apply regex
cleantext_r= cleantextCAP_r.lower() # lower case 

# Save dictionaries for wordcloud
text_file = open("Output_three_tragery.txt", "w")
text_file.write(str(cleantext))
text_file.close()

words=cleantext.split()
dict1 = {}
for word in words:			
    dict1[word] = dict1.get(word,0) + 1


    
# Filter Stopwords
keys = list(dict1)

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
    
dictshow = SequenceSelection(dictionary = dict2, length = 90, startindex = 0)

#these 34 words are given as an example in the book from L.Borke & W. Haerdle
doc_34words="art   away  blood  day  dead  dear  death  eyes  exit  fair  father  fear god good great heart heaven ill king leave lady like life love lord make man men must night sweet think time well"
words_34=doc_34words.split()
dict_34words={}
for word in words_34:
    dict_34words[word]=dictshow[word]



# Plot most frequent words
#n = range(len(dictshow))
#plt.bar(n, dictshow.values(), align='center')
#plt.xticks(n, dictshow.keys())
#plt.title("Most frequent Words")
#plt.savefig("FrequentWords.png", transparent=True)
    
n = range(len(dictshow))
plt.figure(figsize=(20,10))
plt.bar(n, dictshow.values(), align='center')
plt.xticks(n, dictshow.keys(), rotation = 'vertical')
plt.title("Most frequent Words")
plt.savefig("FrequentWords.png", transparent=True)
