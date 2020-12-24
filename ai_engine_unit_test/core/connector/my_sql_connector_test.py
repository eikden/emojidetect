from initial import initial_base
initial_base()

from core.connector.my_sql_connector import MySqlConnector

#MSSqlConnector(config=None, host=False):
conn = MySqlConnector()
print('MySqlConnector: {0}'.format(conn))
