from core.connector.ms_sql_connector import MSSqlConnector
import pandas as pd

def ms_sql_query_get(query, params):
    connector = MSSqlConnector()
    cursor = connector.cursor(as_dict=True)
    cursor.execute(query, params)
    data=cursor.fetchall()
    result=pd.DataFrame(data)
    cursor.close()
    connector.close()
    return result

#to be correct
def get_code(function_name):
    query='''SELECT [code]
             FROM dbo.test_code
             WHERE function_name=%s;
          '''
    params = function_name
    result = ms_sql_query_get(query,params)
    code = str(result.iloc[0,0])
    return code
