#"""
#The flask application package.
#"""

#from flask import Flask
#app = Flask(__name__)

#import ai_api.views

#!/usr/bin/python
# Updated as of 13th April 2020

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import environ
import sys
from flask_swagger_ui import get_swaggerui_blueprint

db = SQLAlchemy()

def create_app(test_config=None):
  # create and configure the app
  app = Flask(__name__, instance_relative_config=False)

  ### swagger specific ###
  SWAGGER_URL = '/dtiapi'
  API_URL = '/static/swagger.json'
  SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "DTI APIs"
    }
  )
  app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
  ### end swagger specific ###
 
  env = sys.argv[1] if len(sys.argv) > 1 else 'development'
  config_dict = {
    'local': 'config.LocalConfig',
    'development': 'config.DevelopmentConfig',
    'uat': 'config.UATConfig',
    'production': 'config.ProductionConfig',
    'demo' : 'config.DemoConfig'
  }
  config = config_dict[env]
  app.config.from_object(config)
  try:
    os.makedirs(app.instance_path)
  except OSError:
    pass
  db.init_app(app)


  with app.app_context():
    from ai_api.models import (User, Role)
    from ai_api.routes import (login, register, api_trigger)
    from ai_api.decorators import before_request, after_request
    
    db.create_all()
    #print('tables created')
    return app
