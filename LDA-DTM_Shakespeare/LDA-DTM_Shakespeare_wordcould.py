#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:25:01 2018

@author: verani
"""
#please install these modules before you run the code 
#!pip install wordcloud
#!pip install Pillow

from os import path
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Please change the working directory to your path!
#os.chdir("/Users/xinwenni/LDA-DTM/Shakespeare") 
root_path = os.getcwd()

# Read the whole text.
with open(path.join(root_path, 'Output.txt'), 'r', encoding='utf-8', errors='ignore') as outout_file:
    text = outout_file.readlines()

# Mask
shakespeare_pic = np.array(Image.open(path.join(root_path, "william-shakespeare-black-silhouette.jpg")))

# Optional additional stopwords
stopwords = set(STOPWORDS)
stopwords.add("ham")

# Construct Word Cloud
# no backgroundcolor and mode = 'RGBA' create transparency
wc = WordCloud(max_words=1000, mask=shakespeare_pic,
               stopwords=stopwords, mode='RGBA', background_color=None)

# Pass Text
wc.generate(text[0])

# store to file
wc.to_file(path.join(root_path, "shakepeare.png"))

# to show the picture 
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")# 
plt.show()


