from initial import initial_base
initial_base()

from core.connector.sqlite3_connector import sqlite3_connector

#MSSqlConnector(config=None, host=False):
conn = sqlite3_connector()
print('sqlite connection status: {0}'.format(conn))
