[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **LDA-DTM_speech_xi_wordcloud** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: LDA-DTM_speech_xi_wordcloud  

Published in: LDA-DTM

Description: "word cloud of the president of the People's Republic of China Xi Jinping's speech in 19th National Congress of the Communist Party of China" 

Keywords: LDA, word cloud, Xi Jinping, China, 19 Da

See also: LDA-DTM_NASDAQ, LDA-DTM_Shakespeare, LDA-DTM_Regulation_Risk

Author: Xinwen Ni

Submitted:  01 OCT 2018



```

![Picture1](danghui.png)

![Picture2](wordcloud_19da.png)

### PYTHON Code
```python

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

```

automatically created on 2020-10-17