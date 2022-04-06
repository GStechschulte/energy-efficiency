import configparser
import os
import json
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
        os.path.dirname(__file__), '../../config/config.ini')
    )
    return config


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


def query_table(table=None, add_params=None):
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
        #query = """
        #        {query}
        #"""
        #query.format(query=add_params)
        df = pd.read_sql_query(add_params, get_db_connection())

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


def get_sensor_map(dataset):

    if dataset == 'gassmann':
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

    elif dataset == 'hipe':
        sensor_map = {
            'chip_saw': 'chip_saw',
            'high_temp_oven': 'high_temp_oven',
            'chip_press': 'chip_press',
            'main_terminal': 'main_terminal',
            'soldering_oven': 'soldering_oven',
            'screen_printer': 'screen_printer',
            'vacuum_pump_2': 'vacuum_pump_2',
            'vacuum_pump_1': 'vacuum_pump_1',
            'vacuum_oven': 'vacuum_oven',
            'washing_machine': 'washing_machine',
            'pick_place_unit': 'pick_place_unit'
        }
        return sensor_map


def weekday_time_series(sensor_id):
    """
    Since we have 10 days of data, the first day, October 8th, is a Friday.
    Preprocess the time series to only include Monday - Friday

    Parameters
    ----------
    df: pd.DataFrame with index datetime

    Returns
    -------
    df: preprocessed with only the working days of the week
    """ 

    df = query_table(sensor_id)
    weekday_df = df[(df.index.day >= 11) & (df.index.day <= 15)]

    return weekday_df

def update_gam_metrics(model, tau, add_regressor, trend, out_sample_rmse, 
in_sample_rmse):

    score_table = 'metrics'

    query = """
        insert into {schema}.{score_table}
        (model, tau, add_regressor, trend, out_sample_rmse, in_sample_rmse)
        values
        ('{model}', '{tau}', TRIM('{add_regressor}'), '{trend}', '{out_sample_rmse}',
        '{in_sample_rmse}')
    """

    query = query.format(
        schema=config['DB']['SCHEMA'],
        score_table=score_table,
        model=model,
        tau=tau,
        add_regressor=json.dumps(add_regressor),
        trend=trend,
        out_sample_rmse=out_sample_rmse,
        in_sample_rmse=in_sample_rmse
    )

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()


def update_gp_metrics(model, time_agg, kernel, lr, training_iter, optimizer, out_sample_mse, 
out_sample_mape, elapsed_time):

    score_table = 'gp_metrics'

    query = """
        insert into {schema}.{score_table}
        (model, time_agg, kernel, lr, training_iter, optimizer, out_sample_mse, out_sample_mape,
        elapsed_time)
        values
        ('{model}', '{time_agg}', TRIM('{kernel}'), '{lr}', '{training_iter}', '{optimizer}', 
        '{out_sample_mse}', '{out_sample_mape}', '{elapsed_time}')
    """

    query = query.format(
        schema=config['DB']['SCHEMA'],
        score_table=score_table,
        model=model,
        time_agg=time_agg,
        kernel=kernel,
        lr=lr,
        training_iter=training_iter,
        optimizer=optimizer,
        out_sample_mse=out_sample_mse,
        out_sample_mape=out_sample_mape,
        elapsed_time=elapsed_time
    )

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()



