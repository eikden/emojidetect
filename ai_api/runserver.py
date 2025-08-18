#from os import environ
#from ai_api import app

#if __name__ == '__main__':
#    HOST = environ.get('SERVER_HOST', 'localhost')
#    try:
#        PORT = int(environ.get('SERVER_PORT', '5555'))
#    except ValueError:
#        PORT = 5555
#    app.run(HOST, PORT)

"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
from os import environ
from ai_api import create_app
import sys
import os
from pathlib import Path

current_path = os.path.dirname(os.path.abspath(__file__))
dti_path = str(Path(current_path).parent) + '\\ai_engine'
#This path will use for server environment setup
sys.path.append(dti_path)
if __name__ == '__main__':
  app = create_app()
  app.run()
  print(app.root_path)
