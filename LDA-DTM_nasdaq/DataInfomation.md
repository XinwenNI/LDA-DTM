# NASDAQ News Database 2016

NASDAQ News data is collected from NASDAQ News platform (https://www.nasdaq.com) that offers news related to Global Market, US Stock Market, US Fix-income Market and etc.
This dataset sample can only be used for academic research purpose.

# Download Link:

https://drive.google.com/file/d/1eAw4_3y1VhBuiYZdxkqbc8dF89yhf-uM/view?usp=sharing

## Format: 

Json format, reading example in Python is provided as follows:

Note: Uunzip the json.zip file before reading

```python
import json

path = 'your working directory path'

with open(path + 'NASDAQ_News_2016.json', 'r') as json_file:
    nasdaq_news_2016 = json.load(json_file)
```

## Time range: 

2016.01.01-2016.12.31, one year sample, 162144 articles in total.

## Features:

article_link: Original link for the article

article_title: Title of the article

article_time: Posted time of the article, string format, UTC -5 time (New York Time)

author_name: Name(s) of the author(s)

author_link: Link to the author(s)'s homepage

article_content: Main content of the article

appears_in: Tags for article's, e.g Investing, Stocks, Options and so on

symbols: Tickers that are related to the article


## Person to contact regarding the dataset

junjie.hu@hu-berlin.de

