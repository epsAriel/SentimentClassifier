from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
db = SQLAlchemy()
database_uri="sqlite:///site.db"
app.config["SQLALCHEMY_DATABASE_URI"] = database_uri
    
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

db.init_app(app)
with app.app_context():
    db.create_all()

app.app_context().push()
db.create_all()
from SentimentClassifier import routes