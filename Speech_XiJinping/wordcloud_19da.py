# coding:utf-8

import os
from os import path
import chnSegment
import plotWordcloud



    # 读取文件
d = os.getcwd()
text = open(path.join(d, 'doc//十九大报告全文.txt')).read()

# 若是中文文本，则先进行分词操作
text=chnSegment.word_segment(text)
    
    # 生成词云
plotWordcloud.generate_wordcloud(text)



