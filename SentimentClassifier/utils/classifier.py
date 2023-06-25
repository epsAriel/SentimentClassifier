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



class SentimentClassifier:
    def __init__(self, data):
        self.data = data
        self.label_mapping = {
            'age': 0,
            'ethnicity': 1,
            'gender': 2,
            'religion': 3,
            'other_cyberbullying': 4,
            'not_cyberbullying': 5
        }

    def preprocess_data(self, embeddings):
        posts = list(self.data[embeddings])
        sentiments = self.data['sentiment'].values
        numeric_sentiments = np.array([self.label_mapping.get(sentiment, -1) for sentiment in sentiments])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(posts, numeric_sentiments, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train, y_train):
        # Train a logistic regression model
        logistic_regression = LogisticRegression(solver='lbfgs', max_iter=1000)
        logistic_regression.fit(X_train, y_train)
        return logistic_regression

    def train_svm(self, X_train, y_train):
        # Train an SVM model
        svm = SVC()
        svm.fit(X_train, y_train)
        return svm

    def train_random_forest(self, X_train, y_train):
        # Train a Random Forest model
        random_forest = RandomForestClassifier()
        random_forest.fit(X_train, y_train)
        return random_forest

    def evaluate_model(self, model, X_test, y_test):
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def save_model(self, model, name):
            filename = f"{name}.pkl"
            with open(filename, "wb") as file:
                pickle.dump(model, file)

    def run_classification(self, embeddings):
        X_train, X_test, y_train, y_test = self.preprocess_data(embeddings)

        # Logistic Regression
        logistic_regression = self.train_logistic_regression(X_train, y_train)
        self.save_model(logistic_regression, "logistic_regression")
        y_pred_lr = logistic_regression.predict(X_test)
        accuracy_lr = self.evaluate_model(logistic_regression, X_test, y_test)
        report_lr = classification_report(y_test, y_pred_lr)
        print("Logistic Regression Accuracy:", accuracy_lr)
        print("Logistic Regression Classification Report:")
        print(report_lr)

        # SVM
        svm = self.train_svm(X_train, y_train)
        self.save_model(svm, "svm")
        y_pred_svm = svm.predict(X_test)
        accuracy_svm = self.evaluate_model(svm, X_test, y_test)
        report_svm = classification_report(y_test, y_pred_svm)
        print("SVM Accuracy:", accuracy_svm)
        print("SVM Classification Report:")
        print(report_svm)

        # Random Forest
        random_forest = self.train_random_forest(X_train, y_train)
        self.save_model(random_forest, "random_forest")
        y_pred_rf = random_forest.predict(X_test)
        accuracy_rf = self.evaluate_model(random_forest, X_test, y_test)
        report_rf = classification_report(y_test, y_pred_rf)
        print("Random Forest Accuracy:", accuracy_rf)
        print("Random Forest Classification Report:")
        print(report_rf)
