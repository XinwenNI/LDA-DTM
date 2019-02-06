# coding:utf-8
# please in stall the following module if you haven't yet.

#!pip install wordcloud
#!pip install jieba

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
import chnSegment
import plotWordcloud

# Please change the working directory to your path!
os.chdir("/Users/xinwenni/LDA-DTM/Speech_XiJinping") 

def generate_wordcloud(text):
    '''
    输入文本生成词云,如果是中文文本需要先进行分词处理
    '''
    # 设置显示方式
    d=path.dirname(__file__)
    mask = np.array(Image.open(path.join(d, "Images//Danghui.png")))
#    mask = np.array(Image.open(path.join(d, "Images//william-shakespeare-black-silhouette.jpg")))
    font_path=path.join(d,"font//msyh.ttf")
    stopwords = set(STOPWORDS)
    wc = WordCloud(
           max_words=2000, #  
           mask=mask,# background       
           stopwords=stopwords, # set the stop words 
           font_path=font_path, # 
           mode='RGBA',
           background_color= None,# 
                  )

    # wordcloud
    wc.generate(text)

    # save to local
    wc.to_file(path.join(d, "Images//wordcloud_19da.png"))

    # show the pic
    plt.imshow(wc, interpolation='bilinear')
    # interpolation='bilinear' 
    plt.axis("off")# 
    plt.show()






    # read the file 
d = os.getcwd()
text = open(path.join(d, 'doc//十九大报告全文.txt'),encoding = 'UTF8').read()

# for Chinese 
text=chnSegment.word_segment(text)
    
    # 
plotWordcloud.generate_wordcloud(text)
