from SentimentClassifier import app
from SentimentClassifier.models import *
from flask import render_template, redirect, request
from .utils.dataInfo import dataInfo
from .utils.preprocessor import PreprocessorC
from .utils.textRep import textToVec
from .utils.classifier import SentimentClassifier
from .utils.predict import *
import pandas as pd
from pathlib import Path
import os

############################################################### Main Nav Bar ############################################################### 
@app.route("/")
def HomePage():
    return render_template('Homepage.html')

@app.route("/Reports")
def Reports():
    reports = Report.query.all()
    return render_template('Reports.html', reports=reports)


@app.route("/report-details/<int:report_id>")
def report_details(report_id):
    report = Report.query.get(report_id)
    acc = report.acc
    preprocessor = Preprocessor.query.get(report.preprocessor_id)
    return render_template('Train2.html', acc=acc, report=report, preprocessor = preprocessor)

############################################################### About data ############################################################### 
@app.route("/About1")
def About1():
    file_path = os.path.join(os.path.abspath(os.getcwd()), "SentimentClassifier", "files", "cyberbullying_tweets.csv")
    data = pd.read_csv(file_path)
    data.rename(columns={data.columns[0]: 'post', data.columns[1]: 'sentiment'}, inplace=True)

    return render_template('About1.html',data=data[:5])


@app.route("/About2")
def About2():
    file_path = os.path.join(os.path.abspath(os.getcwd()), "SentimentClassifier", "files", "cyberbullying_tweets.csv")
    data = pd.read_csv(file_path)
    data.rename(columns={data.columns[0]: 'post', data.columns[1]: 'sentiment'}, inplace=True)

    data_info = dataInfo(data).printDataInfo()
    data_info = data_info.split("\n")

    return render_template('About2.html', data_info=data_info)


@app.route("/About3")
def About3():
    return render_template('About3.html')


@app.route("/About4")
def About4():
    return render_template('About4.html')
############################################################### Train and preprocess ############################################################### 
@app.route("/Train1",  methods=['GET', 'POST'])
def Train1():
    if request.method == 'POST':
        text_rep = request.form.get('textRep')
        model = request.form.get('model')
        cleaner = 'cleaner' in request.form
        lower = 'lower' in request.form
        stemming = 'stemming' in request.form
        lemma = 'lemma' in request.form
        stopwords = 'stopwords' in request.form
        speller = 'speller' in request.form

        acc = 0
        file_path = os.path.join(os.path.abspath(os.getcwd()), "SentimentClassifier", "files", "cyberbullying_tweets.csv")
        data = pd.read_csv(file_path)
        data.rename(columns={data.columns[0]: 'post', data.columns[1]: 'sentiment'}, inplace=True)
        
        prep = PreprocessorC(data)

        data = prep.Remove_Duplicates()
        if cleaner:
            data = prep.String_Cleaner_Splitter()
        if lower:
            data = prep.Upper_To_Lower_Convertor()
        if stemming:
            data = prep.Stemming_words()
        if lemma:
            data = prep.lemmatizer()
        if stopwords:
            data = prep.Remove_Stopwords()
        if speller:
            data = prep.Speller()
        data = prep.Remove_Duplicates()
        data = data.dropna()
        
        textRep = textToVec(data)
        if text_rep == 'bow':
            data['bow'] = textRep.bagOfWords()
        elif text_rep == 'tfidf':
            data['tfidf'] = textRep.tfIdf()
        elif text_rep == 'word2vec':
            data['word2vec'] = textRep.wordToVec()
        elif text_rep == 'glove':
            data['glove'] = textRep.gloveEmbeddings()
        elif text_rep == 'fasttext':
            data['fasttext'] = textRep.fasttextEmbeddings()


        data = data[~(data == 'empty').any(axis=1)]
        sentiment = SentimentClassifier(data)
        X_train, X_test, y_train, y_test = sentiment.preprocess_data(text_rep)
        
        if model == 'logisticRegression':
            logistic_regression = sentiment.train_logistic_regression(X_train, y_train)
            sentiment.save_model(logistic_regression, "logistic_regression")
            y_pred_lr = logistic_regression.predict(X_test)
            acc = sentiment.evaluate_model(logistic_regression, X_test, y_test)
        
        elif model == 'SVM':
            svm = sentiment.train_svm(X_train, y_train)
            sentiment.save_model(svm, "svm")
            y_pred_svm = svm.predict(X_test)
            acc = sentiment.evaluate_model(svm, X_test, y_test)
        
        elif model == 'randomForest':
            random_forest = sentiment.train_random_forest(X_train, y_train)
            sentiment.save_model(random_forest, "random_forest")
            y_pred_rf = random_forest.predict(X_test)
            acc = sentiment.evaluate_model(random_forest, X_test, y_test)
        
        
        preprocessor = Preprocessor(cleaner=cleaner, lower=lower, stemming=stemming, lemma=lemma, stopwords=stopwords, speller=speller)
        db.session.add(preprocessor)
        db.session.commit()

        report = Report(text_rep=text_rep, model=model, preprocessor_id=preprocessor.id, acc=str(acc))
        db.session.add(report)
        db.session.commit()
        
        return render_template('Train2.html', acc=acc, report=report, preprocessor = preprocessor)
    return render_template('Train1.html')
############################################################### Test ############################################################### 
@app.route("/Test1",  methods=['GET', 'POST'])
def Test1():
    if request.method == 'POST':
        sentence = request.form.get('sentence')
        data = pd.DataFrame({'post': [sentence], 'sentiment': [0]})
        res = None
        last_report = Report.query.order_by(Report.id.desc()).first()
        proc = Preprocessor.query.get(last_report.preprocessor_id)
        
        prep = PreprocessorC(data)

        data = prep.Remove_Duplicates()
        if proc.cleaner:
            data = prep.String_Cleaner_Splitter()
        if proc.lower:
            data = prep.Upper_To_Lower_Convertor()
        if proc.stemming:
            data = prep.Stemming_words()
        if proc.lemma:
            data = prep.lemmatizer()
        if proc.stopwords:
            data = prep.Remove_Stopwords()
        if proc.speller:
            data = prep.Speller()
        data = prep.Remove_Duplicates()
        data = data.dropna()

        if last_report.text_rep == 'bow':
            text = text_representation('bow', data)[0]
        elif last_report.text_rep == 'tfidf':
            text = text_representation('tfidf', data)[0]
        elif last_report.text_rep == 'word2vec':
            text = text_representation('word2vec', data)[0]
        elif last_report.text_rep == 'glove':
            text = text_representation('glove', data)[0]
        elif last_report.text_rep == 'fasttext':
            text = text_representation('fasttext', data)[0]

        if last_report.model == 'logisticRegression':
            res = model_predict(text, 'logistic_regression')
        
        elif last_report.model == 'SVM':
            res = model_predict(text, 'svm')
        
        elif last_report.model == 'randomForest':
            res = model_predict(text, 'random_forest')

        return render_template('Test2.html', res=res, text=sentence)
    return render_template('Test1.html')