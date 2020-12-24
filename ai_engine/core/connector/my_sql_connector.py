import mysql.connector
from mysql.connector import Error
from core.connector.dbconfig import read_db_config
import os
from pathlib import Path

#initial path declare
current_path = os.path.dirname(os.path.abspath(__file__))
dti_path = str(Path(current_path).parent.parent)

def MySqlConnector(config=None, host=False):
  """ Connect to MySQL database """
 
  try:
    conn = None
    if not config:
      config = read_db_config(dti_path+r'\config.ini', 'mysql')
    conn = mysql.connector.connect(host=config['host'],database=config['database'],
                                       user=config['user'], password=config['password'])

    if conn.is_connected():
      #print('connection established.')
      return conn
    else:
        print('connection failed.')
  except Error as error:
    print(error)