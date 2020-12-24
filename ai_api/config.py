import os
import sys
from pathlib import Path
current_path = os.getcwd()
dti_path = str(Path(current_path).parent) + '\\ai_engine'

class BaseConfig:
  # General
  FLASK_APP = 'ai_api'
  # DB
  SQLALCHEMY_TRACK_MODIFICATIONS = True

class DevelopmentConfig(BaseConfig):
  ENV = 'development'
  FLASK_ENV = 'development'
  DEBUG = True
  TESTING = True
  SQLALCHEMY_DATABASE_URI = 'sqlite:///'+dti_path+'\\data\\emoji.db'
  SECRET_KEY = 'RPA_SECRET' # os.urandom(24)
  SERVER_NAME = 'localhost:5555'
