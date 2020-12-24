import os
from pathlib import Path
import pymssql
from core.connector.dbconfig import read_db_config

#initial path declare
current_path = os.path.dirname(os.path.abspath(__file__))
dti_path = str(Path(current_path).parent.parent)
def MSSqlConnector(config=None, host=False):
  """ Connect to MSSQL database """

  try:
    conn = None
    if not config:
      config = read_db_config(dti_path+r'\config.ini', 'mssql')
      #window authentication connection
      conn = pymssql.connect(server=config['server'], database=config['database'])
    if conn.cursor():
      #print('connection established.')
      return conn
    else:
        print('connection failed.')
  except Exception as error:
    print(error)
    return conn