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

def text_representation(rep, data):
    representation_filename = f"{rep}.pkl"

    with open(representation_filename, "rb") as file:
        vectorizer = pickle.load(file)

    if rep == 'bow' or rep == 'tfidf':
        transformed_tweets = vectorizer.transform(data['post'])
        transformed_tweets = transformed_tweets.toarray()
        return transformed_tweets

    elif rep == 'word2vec':
        word2vec_matrix = [vectorizer.wv[post.split()].mean(axis=0) if post.split() else "empty" for post in data['post']]
        return word2vec_matrix

    else:
        word_embeddings = []
        for post in data['post']:
            words = post.split()
            post_embeddings = [vectorizer[word] for word in words if word in vectorizer]
            if post_embeddings:
                average_vector = np.mean(post_embeddings, axis=0)
            else:
                average_vector = np.zeros(200)
            word_embeddings.append(average_vector)

        transformed_tweets = np.array(word_embeddings)

        return transformed_tweets

def model_predict(text, model_name):
    model_filename = f"{model_name}.pkl"

    with open(model_filename, "rb") as file:
        model = pickle.load(file)

    label_number = model.predict([text])[0]
    label_text = get_label_text(label_number)

    return label_text

def get_label_text(label_number):
    label_mapping = {
        0: 'age',
        1: 'ethnicity',
        2: 'gender',
        3: 'religion',
        4: 'other_cyberbullying',
        5: 'not_cyberbullying'
    }

    return label_mapping.get(label_number, 'unknown')