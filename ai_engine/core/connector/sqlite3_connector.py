import sqlite3
from sqlite3 import Error
from core.connector.dbconfig import read_db_config
import os
from pathlib import Path

#initial path declare
current_path = os.path.dirname(os.path.abspath(__file__))
dti_path = str(Path(current_path).parent.parent)

def sqlite3_connector(config=None):
    conn = None
    try:
        if not config:
            config = read_db_config(dti_path+r'\config.ini', 'sqlite3')
        #window authentication connection
        connection_path=os.path.join(config['path'],config['database'])
        print('Connection String: {0}'.format(connection_path))
        conn = sqlite3.connect(connection_path)
        print(sqlite3.version)
        return conn
    except Error as e:
        print(e)
        return conn