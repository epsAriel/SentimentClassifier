from SentimentClassifier import db

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text_rep = db.Column(db.String(20), nullable=False)
    model = db.Column(db.String(20), nullable=False)
    preprocessor_id = db.Column(db.Integer, db.ForeignKey('preprocessor.id'))
    acc =  db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f"Report('{self.text_rep}', '{self.model}', '{self.preprocessor_id}')"


class Preprocessor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cleaner = db.Column(db.Boolean)
    lower = db.Column(db.Boolean)
    stemming = db.Column(db.Boolean)
    lemma = db.Column(db.Boolean)
    stopwords = db.Column(db.Boolean)
    speller = db.Column(db.Boolean)

    def __repr__(self):
        return f"Preprocessor('{self.cleaner}', '{self.lower}', '{self.stemming}', '{self.lemma}', '{self.stopwords}', '{self.speller}')"
