import pandas as pd
import re
import warnings
import numpy as np
import nltk
from nltk.corpus import stopwords, words
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class dataInfo():
  
  def __init__(self, data):
    self.data = data
  
  def isNull(self):
    return self.data.isnull().sum()
  
  def isDuplicates(self):
    return self.data.duplicated().sum()
  
  def dataShape(self):
    return self.data.shape
  
  def tokenized_post(self):
    self.data['tokens'] = self.data['post'].apply(lambda x: x.split())

  def words_in_each_post(self):
    self.data['words_length'] = self.data['tokens'].apply(lambda x: len(x))
  
  def length_of_each_post(self):
    self.data['post_length'] = self.data['post'].apply(lambda x: len(x))
  
  def amountOfTweetsInEachCategory(self):
    return self.data['sentiment'].value_counts()

  def The_average_words_in_text(self):
    sum = 0
    for i in self.data['words_length']:
        sum += i
    return sum / len(self.data['post'])

  def The_average_length_of_text(self):
    sum = 0
    for i in self.data['post_length']:
        sum += i
    return sum / len(self.data['post'])

  def The_most_common_words(self):
    word_count = nltk.FreqDist([word for tokens in self.data['tokens'] for word in tokens if not word.startswith('#')])
    return word_count
  
  def The_most_common_hashtags(self):
    hashtag_count = nltk.FreqDist([re.sub(r'^#', '', word) for tokens in self.data['tokens'] for word in tokens if word.startswith('#')])
    return hashtag_count

  def The_most_common_users_tags(self):
    usertag_count = nltk.FreqDist([re.sub(r'^@', '', word) for tokens in self.data['tokens'] for word in tokens if word.startswith('@')])
    return usertag_count

  def value_information(self):
    self.tokenized_post()
    self.words_in_each_post()
    self.length_of_each_post()

  def printDataInfo(self):

    text = ""
    text += str(f"<Number of empty values:\n{self.isNull()}\n")
    text += f"\n<Number of Duplicates:\n{self.isDuplicates()}\n"

    self.value_information()

    text += f"\n<Number of empty values:\n{self.isNull()}\n"
    text += f"\n<Number of posts per category:\n{self.amountOfTweetsInEachCategory()}\n"
    text += f"\n<Average words in text:\n{self.The_average_words_in_text()}\n"
    text += f"\n<Average length of text:\n{self.The_average_length_of_text()}\n"
    
    
    word_count = self.The_most_common_words()

    text+= '\n<Most common words:\n'

    for word, count in word_count.most_common(10):
        text+= f'{word}: {count}\n'
        print(f'{word}: {count}')
    
    hashtag_count = self.The_most_common_hashtags()
    
    text+= '\n<Most common hashtags:\n'
    print('\n<Most common hashtags:')

    for hashtag, count in hashtag_count.most_common(10):
        text+= f'{hashtag}: {count}\n'
        print(f'{hashtag}: {count}')

    usertag_count = self.The_most_common_users_tags()


    text+= '\n<Most common user tags:\n'
    print('\n<Most common user tags:')
    for usertag, count in usertag_count.most_common(10):
        text+= f'{usertag}: {count}\n'
        print(f'{usertag}: {count}')

    return text