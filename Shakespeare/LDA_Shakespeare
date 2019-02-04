#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:15:06 2018

@author: verani
"""

# Further Analysis
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import os

from os import path


import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer

import pandas as pd
from nltk.tokenize import RegexpTokenizer


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
    
dictshow = SequenceSelection(dictionary = dict2, length = 100, startindex = 0)

#these 34 words are given as an example in the book from L.Borke & W. Haerdle
doc_34words="art   away  blood  day  dead  dear  death  eyes  exit  fair  father  fear god good great heart heaven ill king leave lady like life love lord make man men must night sweet think time well"
words_34=doc_34words.split()
dict_34words={}
for word in words_34:
    dict_34words[word]=dictshow[word]



# Plot most frequent words
n = range(len(dictshow))
plt.bar(n, dictshow.values(), align='center')
plt.xticks(n, dictshow.keys())
plt.title("Most frequent Words")
plt.savefig("FrequentWords.png", transparent=True)



# LDA analysis

d = os.getcwd()




text_pre = open(path.join(d, 'three_tragedy.txt'), encoding = "utf8").read()

doc_l = str.split(text_pre)

#doc_l.pop()[0]

doc_complete = doc_l

doc_out = []
for l in doc_l:
    
    cleantextprep = str(l)
    
    # Regex cleaning
    expression = "[^a-zA-Z ]" # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep) # apply regex
    cleantext = cleantextCAP.lower() # lower case 
    bound = ''.join(cleantext)
    doc_out.append(bound)

doc_complete = doc_out


import string

stop = set(stopwords.words('english'))

exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
import nltk
nltk.download('wordnet')
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


doc_clean = [clean(doc).split() for doc in doc_complete]    


# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
K=3
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=K, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=K, num_words=10))

topicWordProbMat=ldamodel.print_topics(K)




# LDA heatmap



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





