import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from gensim.models import Word2Vec, KeyedVectors
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
import pickle

class textToVec:

  def __init__(self, data):
    self.data = data
  
  def bagOfWords(self):
    bow, vectorizer = [], CountVectorizer(max_features=300)
    matrix = vectorizer.fit_transform(self.data['post'])
    [bow.append(row.toarray().flatten()) for row in matrix]

    self.save_representation(vectorizer, 'bow')

    return bow

  def tfIdf(self):
    tf, vectorizer = [], TfidfVectorizer(max_features=300)
    matrix = vectorizer.fit_transform(self.data['post'])
    [tf.append(row.toarray().flatten()) for row in matrix]

    self.save_representation(vectorizer, 'tfidf')
    return tf

  def wordToVec(self):
    sentences = [post.split() for post in self.data['post']]
    model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
    word2vec_matrix = [model.wv[post.split()].mean(axis=0) if post.split() else "empty" for post in self.data['post']]

    self.save_representation(model, 'word2vec')
    return word2vec_matrix

  def gloveEmbeddings(self):
      glove_model = self.loadGloveModel('glove.6B.300d.txt')
      word_embeddings_matrix = self.calculateEmbeddings(glove_model)

      with open('glove.pkl', 'wb') as file:
        pickle.dump(glove_model, file)
      return word_embeddings_matrix

  def fasttextEmbeddings(self):
      fasttext_model = self.loadFastTextModel('wiki-news-300d-1M.vec')
      word_embeddings_matrix = self.calculateEmbeddings(fasttext_model)

      with open('fasttext.pkl', 'wb') as file:
        pickle.dump(fasttext_model, file)
      return word_embeddings_matrix

  def calculateEmbeddings(self, model):
    word_embeddings_matrix = []
    for post in self.data['post']:
        words = post.split()
        vectors = [model[word] for word in words if word in model]
        if vectors:
            average_vector = sum(vectors) / len(vectors)
        else:
            average_vector = "empty"
        word_embeddings_matrix.append(average_vector)
    return word_embeddings_matrix

  def loadGloveModel(self, glove_file):
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype='float32')
                glove_model[word] = vector
            except ValueError:
                pass
    return glove_model
    
  def loadFastTextModel(self, fasttext_file):
    fasttext_model = KeyedVectors.load_word2vec_format(fasttext_file)
    return fasttext_model

  def save_representation(self, vectorizer, name):
    filename = f"{name}.pkl"
    with open(filename, "wb") as file:
        pickle.dump(vectorizer, file)

  def allRepresentations(self):
    self.data['bow'] = self.bagOfWords()
    self.data['tfidf'] = self.tfIdf()
    self.data['word2vec'] = self.wordToVec()
    self.data['glove'] = self.gloveEmbeddings()
    self.data['fasttext'] = self.fasttextEmbeddings()

    return self.data