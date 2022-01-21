import configparser
import os
from matplotlib.pyplot import connect
import psycopg2
import pandas as pd

# Set variables
config = None
conn = None

def get_config():
    """
    get configuration. . .
    """
    global config

    if config is not None:
        return config

    config = configparser.ConfigParser()
    config.read(os.path.join(
        os.path.dirname(__file__), '../../config/config.ini')
        )

    return config

def get_db_connection():
    """
    if conn is None, return conn
    else connect to db
    """
    
    global conn

    if conn is not None:
        return conn
    else:
        config = get_config()
        conn = psycopg2.connect(
            host=config['DB']['HOST'],
            database=config['DB']['DATABASE'],
            user=config['DB']['USER'],
            password=config['DB']['PASSWORD']
            )

    return conn

def query_table(table, add_params=None):
    """
    table: name of table to query
    add_params: add custom query here
    returns: pandas dataframe with 't' as index
    """
    if add_params is None:
        query = '''
                select
                    *
                from gassmann.{}
        '''.format(table)

        df = pd.read_sql_query(
            query, get_db_connection(), index_col='t'
            )

        return df
    else:
        query = """
                {query}
        """
        query.format(query=add_params)
        df = pd.read_sql_query(query, get_db_connection())

        return df
    

