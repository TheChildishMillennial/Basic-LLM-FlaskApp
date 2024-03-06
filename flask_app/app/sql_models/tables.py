from flask_app.app.extensions import db

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)