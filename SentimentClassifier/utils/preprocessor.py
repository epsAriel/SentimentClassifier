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
import pickle
from sklearn.metrics import classification_report

class PreprocessorC:
    
    ################################ Class c'tor ############################
    def __init__(self, data):
        """ c'tor of the class"""
        self.Data = data
    
    ################################ remove all the data duplicates ############################
    def Remove_Duplicates(self):
      return self.Data.drop_duplicates()
    
    ################################ switch all the letters from upper letters to lower letters ############################
    def Upper_To_Lower_Convertor(self):
        """ This function switch all upper letters to lower letters """
        self.Data['post'] = list(map(lambda x: x.lower(), self.Data['post']))
        return self.Data
        
    ################################ remove all the punctuation marks and special chars ############################
    def String_Cleaner_Splitter(self):
        """ This function clean the string from punctuation and special marks """
        self.Data['post'] = self.Data['post'].str.replace('[^\w\s]','')
        return self.Data
    
    
    ################################ helping function ############################
    def String_connector(self, sentence):
        """ This function take a list of tweets and conncet them as a one string """
        text = ' '
        for i in sentence:
            text += (' ' + i)
        return text
    
    
    ################################ remove all the stopwords ############################
    def Remove_Stopwords(self):
        """ This function remove all the stopwords in all sentences """
    
        def Remove(table):
            """ This function recieve a single sentence and returns it without stopwords"""
               
            filtered_words = []                                # create a list of the sentence words without the stopwords
            StopWords = set(stopwords.words('english'))        # set of stopwords
    
            for i in table.split():                            # run all the words in single sentences
                if not i in StopWords:
                    filtered_words.append(i)
         
            return self.String_connector(filtered_words)      # return the connected words as string.
        
        self.Data['post'] = list(map(Remove,self.Data['post']))           
        
        return self.Data
 

    ############################################# Helping function ####################################
    def Spelling(self, sentence , dic):
        text_list = sentence.split()
        text = ''
        
        for i in text_list:
            if i in dic:
                text += (i + ' ')
            else:
                text += (str(TextBlob(i).correct()) + ' ')
        
        return text
    
    ################################ fix spelling mistakes #####################################
    def Speller(self):
        dic = words.words()
        self.Data['post'] = list(map(lambda x: self.Spelling(x, dic), self.Data['post']))
        
        return self.Data
    
    def lemmatizer(self):
        def lemma(data):
        
            lemmatizer = WordNetLemmatizer()
            sent =  list(map(lambda x: lemmatizer.lemmatize(x), data.split()))
            text = ""
        
            for i in sent:
                text += (i + " ")

            return text
        self.Data['post'] = list(map(lambda x: lemma(x), self.Data['post']))
        return self.Data
    
    def Stemming_words(self):
        def Stemming(sent):
            ps = PorterStemmer()
            sent = list(map(lambda x: ps.stem(x), sent.split()))
            text = ""
        
            for i in sent:
                text += (i + " ")

            return text
        
        self.Data['post'] = list(map(lambda x: Stemming(x), self.Data['post']))   
        return  self.Data
    
    
    ################################ initialize all class function ############################
    def Proccesor(self):
        """ This funcion runs all the dataset preprocessing """
        
        self.Data = self.Remove_Duplicates()
        self.Data=self.String_Cleaner_Splitter()
        self.Data=self.Upper_To_Lower_Convertor()
       # self.Data = self.Stemming_words()
       # self.Data = self.lemmatizer()
        self.Data=self.Remove_Stopwords()
       # self.Data = self.Speller()
        
        self.Data = self.Remove_Duplicates()
        self.Data = self.Data.dropna()
        return self.Data 