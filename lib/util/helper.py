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
    Get configuration of the database
    return: DB config
    """
    global config

    if config is not None:
        return config

    config = configparser.ConfigParser()
    #if local == True:
    config.read(os.path.join(
        os.path.dirname(__file__), '../../config/config_local.ini')
    )
    return config
    #else:
    #    config.read(os.path.join(
    #        os.path.dirname(__file__), '../../config/config.ini')
    #        )
    #    return config

def get_db_connection(engine=False):
    """
    if conn is None, return conn
    else connect to db
    return: DB connection
    """
    
    global conn

    if conn is not None:
        return conn
    elif engine == True:
        config = get_config()
        engine = create_engine(
            'postgresql+psycopg2://{}:{}@{}/{}'.format(
            config['DB']['USER'],
            config['DB']['PASSWORD'],
            config['DB']['HOST'],
            config['DB']['DATABASE']
            )
        )
        return engine
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

    config = get_config()

    if add_params is None:
        query = """
                select
                    *
                from {}."{}"
        """.format(config['DB']['SCHEMA'], table)

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


def get_table_names(time=None):

    config = get_config()

    if time == None:
        query = """
        select
            table_name
        from 
            information_schema.tables
        where 
            table_name not like '%H%'
        and 
            table_name not like '%T%'
        and 
            table_name like 'g_%'
        """

        all_table_names = pd.read_sql_query(query, get_db_connection())
        return all_table_names.to_numpy()

    elif time == '1H':
        query = """
        select
            table_name
        from 
            information_schema.tables
        where 
            table_name like '%H%'
        """

        hourly_table_names = pd.read_sql_query(query, get_db_connection())
        return hourly_table_names.to_numpy()

    else:
        raise ValueError('Time value is not to be recieved by this function')


def get_sensor_map():

    sensor_map = {
        '5fe33f53923d596335e69d41': 'gesamtmessung',
        '5fe3400d923d596335e69d42': 'vk_2_eg',
        '5fe34044923d596335e69d43': 'stahl_folder',
        '5fe34060923d596335e69d44': 'og_3',
        '5fe34098923d596335e69d46': 'eg',
        '5fe340b3923d596335e69d47': 'entsorgung',
        '5fe340cb923d596335e69d48': 'og',
        '5fe340e3923d596335e69d49': 'uv_eg',
        '5fe340fc923d596335e69d4a': 'r707lv_f4032',
        '5fe34116923d596335e69d4b': 'uv_sigma_line_eg',
        '5fe3412c923d596335e69d4c': 'r707lv_trockner',
        '5fe34145923d596335e69d4d': 'r707lv_vari_air',
        '5fe3415e923d596335e69d4e': 'xl106_druckmaschine',
        '5fe3417a923d596335e69d4f': 'xl106_uv_scan',
        '5fe34191923d596335e69d50': 'hauptluftung',
        '5fe341bd923d596335e69d51': 'vk_1_ug',
        '5fe341d4923d596335e69d52': 'r707lv_f4034',
        '5fe341ee923d596335e69d53': 'og_2',
        '6017e77a42d6f4614409d192': 'not_in_list'
    }

    return sensor_map