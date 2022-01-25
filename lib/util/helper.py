import configparser
import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

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

def get_db_connection(engine=False):
    """
    if conn is None, return conn
    else connect to db
    """
    
    global conn

    if conn is not None:
        return conn
    elif engine == True:
        config = get_config()
        engine = create_engine(
            'postgresql+psycopg2://{}:{}@{}/{postgres}'.format(
            config['DB']['USER'],
            config['DB']['PASSWORD'],
            config['DB']['HOST'],
            config['DB']['DATABASE']
            )
        )
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
        query = """
                select
                    *
                from gassmann.{}
        """.format(table)

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

def calculate_kilowatt():

    query = """
    select 
        table_name
    from information_schema.tables
    where table_name like 'g%'
    """

    table_names = pd.read_sql_query(query, get_db_connection())

    for table in table_names.to_numpy():

        print('Creating kW materialized view for', table[0])

        query = """
        drop materialized view if exists gassmann.{}_kw;
        create materialized view gassmann.{}_kw as
        (select
            k."t",
            avg(k.kw) as avg_kw
        from (select
                g."t",
                (g."V" * g."I" * g."PF") / 1000 as kw
            from gassmann.{} g) k
        group by k."t");
        """.format(table[0], table[0], table[0])

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()

        print('kW materialized view for', table[0], 'completed')
   

    