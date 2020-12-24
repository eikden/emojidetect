from initial import initial_base
initial_base()

from core.connector.ms_sql_connector import MSSqlConnector

#MSSqlConnector(config=None, host=False):
conn = MSSqlConnector()
print('MsSqlConnector: {0}'.format(conn))
