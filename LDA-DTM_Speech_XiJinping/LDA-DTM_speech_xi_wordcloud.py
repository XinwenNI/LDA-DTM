# coding:utf-8
# please install the following module if you haven't yet.

#!pip install wordcloud
#!pip install jieba

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
import chnSegment


# Please change the working directory to your path!
#os.chdir("/Users/xinwenni/LDA-DTM/LDA-DTM_Speech_Xijiping") 

# read the file 
d = os.getcwd()
text = open(path.join(d, 'doc//十九大报告全文.txt'),encoding = 'UTF8').read()

# for Chinese 
text=chnSegment.word_segment(text)
    
    # 
#plotWordcloud.generate_wordcloud(text)
stopwords = set(STOPWORDS)
font_path=path.join(d,"font//msyh.ttf")
mask = np.array(Image.open(path.join(d, "Danghui.png")))
wc = WordCloud(max_words=1000, mask=mask,
               stopwords=stopwords,font_path=font_path, mode='RGBA', background_color=None)

# Pass Text
wc.generate(text)

# store to file
wc.to_file(path.join(d, "wordcloud_19da.png"))

# to show the picture 
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")# 
plt.show()
