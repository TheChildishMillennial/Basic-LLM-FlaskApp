from flask import Blueprint

bp = Blueprint('llm', __name__)

from flask_app.app.llm import routes