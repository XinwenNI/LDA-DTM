#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 20:39:16 2018

@author: xinwenni
"""

import os
import re
import pandas as pd
## `nltk.download('punkt')
import numpy as np
from nltk.corpus import stopwords
from os import path
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
from gensim import corpora
from collections import  defaultdict
from gensim.test.utils import common_corpus
from gensim.models import LdaSeqModel
from gensim.matutils import hellinger
from gensim.corpora import Dictionary, bleicorpus
import numpy
from gensim.models import ldaseqmodel

path= os.getcwd()

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


# load data
#df = pd.read_csv('df02.csv',encoding="ISO-8859-1")
df = pd.read_csv('dfsent02.csv',encoding="ISO-8859-1")

time=df['date']
df['year']=time.str.slice(0,4)       

df = pd.concat([df, pd.DataFrame(columns = ['year']),
                      pd.DataFrame(columns = ['month']),
                      pd.DataFrame(columns = ['clean_content'])])


#df1=np.array(df)
#df1=df1.tolist()


temp_df=df[0:20]

for i in range(len(df)):
    content=df.iat[i,3]
#    content=temp_df.iloc[i:i+1,['body']]
    content_clean=BasicCleanText(raw_text=content)
    content_clean=" ".join(content_clean)
    df.iat[i,4]=content_clean

gp=df.groupby(by=['year'])
total_yearly_list=list(gp.size())

documents=list(df['clean_content'])
stoplist=stopwords

stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# drop the words only appers once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

dictionary = corpora.Dictionary(texts)   # generate the dictionary
dictionary.compactify() #

dictionary.save(os.path.join('dictionary.dict')) # store the dictionary, for future reference
print(dictionary)

#Save vocabulary
vocFile = open(os.path.join( 'vocabulary.dat'),'w')
for word in dictionary.values():
    vocFile.write(word+'\n')
    
vocFile.close()
print(vocFile)

#Prevent storing the words of each document in the RAM
class MyCorpus(object):
     def __iter__(self):
         for document in documents:
             # assume there's one document per line, tokens separated by whitespace
             yield dictionary.doc2bow(document.lower().split())


corpus_memory_friendly = MyCorpus()

multFile = open(os.path.join( 'foo-mult.dat'),'w')

for vector in corpus_memory_friendly: # load one vector into memory at a time
    multFile.write(str(len(vector)) + ' ')
    for (wordID, weigth) in vector:
        multFile.write(str(wordID) + ':' + str(weigth) + ' ')

    multFile.write('\n')
    
multFile.close()

print(multFile)


time_slice=total_yearly_list

#LdaSeqModel(corpus=None, time_slice=None, id2word=None, alphas=0.01, num_topics=10, initialize='gensim', sstats=None, lda_model=None, obs_variance=0.5, chain_variance=0.005, passes=10, random_state=None, lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20, chunksize=100)

#use LdaSeqModel to generate DTM results
ldaseq = LdaSeqModel(corpus=corpus_memory_friendly, id2word=dictionary, time_slice=time_slice, num_topics=5)
# for given time, the distriibution of each topic 
ldaseq.print_topics(time=1)
# for given topic the word distribution over time
DTM_topic0=ldaseq.print_topic_times(topic=0, top_terms=10)

#
#for i in range(5):
#    arr=ldaseq.print_topic_times(topic=0, top_terms=10)
#    for key in arr[1].keys():
#        for j in range(len(time_slice)):
#            
#    for ​key ​in ​arr​[​1​].​keys​():
#        for year_i ​in​ range​(len(time_slice)​):
#            print(​[​conference​,​topic_i ​,​key​,​​(​year_i ​+​​2009​),​arr​[​1​][​key​][​year_i​]])
 

#

## the function doc_topics checks the topic proportions on documents already trained on. It accepts the document number in the corpus as an input.
#words = [dictionary[word_id] for word_id, count in corpus_memory_friendly[25]]
#print (words)
#
#doc = ldaseq.doc_topics(25) 
#print(doc)


## set Chain Variance
#ldaseq_chain = ldaseqmodel.LdaSeqModel(corpus=corpus_memory_friendly, id2word=dictionary, time_slice=time_slice, num_topics=5)
##ldaseq_chain = ldaseqmodel.LdaSeqModel(corpus=corpus_memory_friendly, id2word=dictionary, time_slice=time_slice, num_topics=5, chain_variance=0.05)
#ldaseq_chain.print_topic_times(2)



#Visualising Dynamic Topic Models

#path= os.getcwd()
#
#from gensim.models.wrappers.dtmmodel import DtmModel
#from gensim.corpora import Dictionary, bleicorpus
#import pyLDAvis
#
#path= os.getcwd()
#dtm_model = DtmModel(path, corpus_memory_friendly, time_slice, num_topics=5, id2word=dictionary, initialize_lda=True)
#dtm_model.save('dtm_policy_risks')
#
## if we've saved before simply load the model
#dtm_model = DtmModel.load('dtm_policy_risks')
#
#
#doc_topic, topic_term, doc_lengths, term_frequency, vocab = dtm_model.dtm_vis(time=0, corpus=corpus_memory_friendly)
#vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
#pyLDAvis.display(vis_wrapper)



            