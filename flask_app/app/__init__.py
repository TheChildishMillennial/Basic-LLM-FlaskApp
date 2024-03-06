from flask import Flask

from flask_app.app.config import Config
from flask_app.app.extensions import db

# import all the classes from the sql_tables
from flask_app.app.sql_models.tables import Admin

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize Flask Extension Here
    db.init_app(app)

    with app.app_context():
        db.create_all()

    #Register Blueprints Here
    from flask_app.app.llm import bp as llm_bp
    app.register_blueprint(llm_bp)

    return app
