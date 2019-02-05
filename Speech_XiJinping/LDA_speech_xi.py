# coding:utf-8

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
import chnSegment
import plotWordcloud



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
           max_words=2000, # 词云显示的最大词数  
           mask=mask,# 设置背景图片       
           stopwords=stopwords, # 设置停用词
           font_path=font_path, # 兼容中文字体，不然中文会显示乱码
           mode='RGBA',
           background_color= None,# 设置背景颜色
                  )

    # 生成词云 
    wc.generate(text)

    # 生成的词云图像保存到本地
    wc.to_file(path.join(d, "Images//wordcloud_19da.png"))

    # 显示图像
    plt.imshow(wc, interpolation='bilinear')
    # interpolation='bilinear' 表示插值方法为双线性插值
    plt.axis("off")# 关掉图像的坐标
    plt.show()






    # 读取文件
d = os.getcwd()
text = open(path.join(d, 'doc//十九大报告全文.txt')).read()

# 若是中文文本，则先进行分词操作
text=chnSegment.word_segment(text)
    
    # 生成词云
plotWordcloud.generate_wordcloud(text)
