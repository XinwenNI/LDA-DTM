#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:35:05 2018

@author: verani
"""#

import os
import json

import re
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from os import path

import matplotlib.pyplot as plt

import nltk

from PIL import Image


root_path= os.getcwd()

with open(root_path + '/NASDAQ_News_2016.json', 'r') as json_file:
    nasdaq_news_2016 = json.load(json_file)
    

TempTry=nasdaq_news_2016
n=len(TempTry)
AllTitles=''.split()
for i in range(n):
    temp1=TempTry[i]
    temp2=temp1['article_title']
    AllTitles.append(temp2)

string= ""
allTit=string.join(AllTitles)


expression = "[^a-zA-Z0-9 ]" # keep only letters, numbers and whitespace
cleantextCAP = re.sub(expression, '', allTit) # apply regex
cleantext = cleantextCAP.lower() # lower case 
#raw_words=nltk.word_tokenize(allTit)
##raw1=cleantext.split()
#
#porter_stemmer = PorterStemmer()
#words_stem = [porter_stemmer.stem(raw_words) for raw_words in raw_words]
#
#
#wordnet_lematizer = WordNetLemmatizer()
#words1 = [wordnet_lematizer.lemmatize(raw_words) for raw_words in raw_words]
#
##words  = cleantext.split()
#words = [word for word in words1 if word not in stopwords.words('english')]




text_file = open("Output_titles.txt", "w")
text_file.write(str(cleantext))
text_file.close()



#wordcloud 
# Read the whole text.
with open(path.join(root_path, 'Output_titles.txt'), 'r', encoding='utf-8', errors='ignore') as outout_file:
    text = outout_file.readlines()

# Mask
nasdaq_pic = np.array(Image.open(path.join(root_path, "nasdaq_bull.png")))

# Optional additional stopwords
stopwords = set(STOPWORDS)


# Construct Word Cloud
# no backgroundcolor and mode = 'RGBA' create transparency
wc = WordCloud(max_words=1000, mask=nasdaq_pic,
               stopwords=stopwords, mode='RGBA', background_color=None)

# Pass Text
wc.generate(text[0])

# store to file
wc.to_file(path.join(root_path, "wordcloud_nasdaq_titles.png"))





