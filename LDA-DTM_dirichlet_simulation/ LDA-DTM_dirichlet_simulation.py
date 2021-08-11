#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:19:49 2019

@author: xinwenni
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

TopicShakespeare=['love','death','king']
Topic=['Education','Economics','Transport']
TopicEdu=['University','Teacher','Course']
TopicEco=['Market','Company','Finance']
TopicTrans=['Train','Car','Airplane']

# length of the simulated document 
n = 10000;

# dirictlet parameters 
alpha =  pd.DataFrame([[1, 1, 1],[10, 10, 10],[2,5,15]], columns=['col1','col2','col3'], index=['alpha1','alpha2','alpha3'])# control document via topic
beta =  pd.DataFrame([[1, 1, 1],[10, 10, 10],[2,5,15]], columns=['col1','col2','col3'], index=['beta1','beta2','beta3'])# control document via topic



PickedTopic=pd.DataFrame(TopicShakespeare)
# assume there is only one topic, generate m different documents according
# to different dirictlet perameters 
count=np.zeros([len(beta['col1']),len(beta.T['beta1'])])
count=pd.DataFrame(count,columns=['word1','word2','word3'], index=['beta1','beta2','beta3'])


txt=np.zeros([n,len(beta['col1'])])
txt=pd.DataFrame(txt, columns=['beta1','beta2','beta3'])
index=np.zeros([n,len(beta['col1'])])
index=pd.DataFrame(index, columns=['beta1','beta2','beta3'])

# here fixed the number of words in the topic 

for m in range(len(beta.T['beta1'])):
    phi=np.random.dirichlet(beta.iloc[m,:], n).transpose()
    phi=pd.DataFrame(phi)
    for i in range(n):
        x=random.uniform(0, 1)
        if x<phi.iloc[0,i]:
            txt.iloc[i,m]=PickedTopic.iloc[0,0]
            index.iloc[i,m]=1
            count.iloc[m,0]=count.iloc[m,0]+1;
        elif x<phi.iloc[0,i]+phi.iloc[1,i]:
            txt.iloc[i,m]=PickedTopic.iloc[1,0]
            index.iloc[i,m]=2
            count.iloc[m,1]=count.iloc[m,1]+1;
        else:
            txt.iloc[i,m]=PickedTopic.iloc[2,0]
            index.iloc[i,m]=3
            count.iloc[m,2]=count.iloc[m,2]+1;
prob_matrix=count/n

print(txt.beta1)
print(txt.beta2)
print(txt.beta3)


#plot the prob distribution 

ind = np.arange(len(beta.T['beta1']))
width = 0.75
p1 = plt.bar(ind, prob_matrix.iloc[0,:], width,color=['r', 'black', 'yellow'])
plt.xticks(ind, ('Love', 'Death', 'King'))
plt.savefig("Probability_bargraph_1.png")

plt.show()



ind = np.arange(len(beta.T['beta3']))
width = 0.75
p1 = plt.bar(ind, prob_matrix.iloc[2,:], width,color=['r', 'black', 'yellow'])
plt.xticks(ind, ('Love', 'Death', 'King'))
plt.savefig("Probability_bargraph_3.png")

plt.show()