from lib.util import helper
import pandas as pd
import os
from os.path import isfile, join
from os import listdir
from sqlalchemy import create_engine
import psycopg2
from lib.util import helper


def run_timeseries(dataset, custom=False, time_agg=list):

    config = helper.get_config()

    engine = create_engine(
    'postgresql+psycopg2://{}:{}@{}/{}'.format(
        config['DB']['USER'],
        config['DB']['PASSWORD'],
        config['DB']['HOST'],
        config['DB']['DATABASE'])
    )

    print(engine)

    if custom != True:
        time_resolution = ['1T', '5T', '10T', '15T', '30T', '1H']
    else:
        assert time_agg != list
        minutes = time_agg

    
    if dataset == 'gassmann':

        table_names = list(helper.get_sensor_map(dataset='gassmann').values())
        
        query = """
        select 
            table_name
        from information_schema.tables
        """

        all_tables = pd.read_sql_query(query, helper.get_db_connection())

        for table in all_tables.to_numpy():
            config = helper.get_config()

            if table in table_names:
                query = """
                select
                    *
                from {}.{}
                """.format(config['DB']['SCHEMA'], table[0])
                
                df = pd.read_sql_query(query, helper.get_db_connection(), index_col='t')

                for time in time_resolution:
                    feats = ['v', 'i', 's', 'p', 'q', 'pf', 'phi']
                    print('Creating', time, 'time aggregation table for', table[0])

                    df_time_agg = pd.DataFrame(
                    data=df[feats].resample(time).mean(),
                    index=df.resample(time).sum().index
                    )

                    df_time_agg['kw'] = df_time_agg['p'] / 1000
                    table_name = table[0] + '_{}'.format(time)

                    df_time_agg.to_sql(
                        "{}".format(table_name), 
                        con=engine, 
                        index=True,
                        if_exists='replace',
                        schema='{}'.format(config['DB']['SCHEMA'],))

                print('Time aggregation materialized view for', table[0], 'completed')
            
            else:
                pass
    
    elif dataset == 'hipe':

        table_names = list(helper.get_sensor_map(dataset='hipe').values())
        
        query = """
        select 
            table_name
        from information_schema.tables
        """

        all_tables = pd.read_sql_query(query, helper.get_db_connection())

        for table in all_tables.to_numpy():
            config = helper.get_config()

            if table in table_names:
                query = """
                select
                    *
                from {}.{}
                """.format(config['DB']['SCHEMA'], table[0])
                
                df = pd.read_sql_query(query, helper.get_db_connection(), index_col='t')

                for time in time_resolution:
                    feats = ['kw']
                    print('Creating', time, 'time aggregation table for', table[0])

                    df_time_agg = pd.DataFrame(
                    data=df[feats].resample(time).mean(),
                    index=df.resample(time).sum().index
                    )

                    df_time_agg['p'] = df_time_agg['kw'] * 1000
                    table_name = table[0] + '_{}'.format(time)

                    df_time_agg.to_sql(
                        "{}".format(table_name), 
                        con=engine, 
                        index=True,
                        if_exists='replace',
                        schema='{}'.format(config['DB']['SCHEMA'],))

                print('Time aggregation materialized view for', table[0], 'completed')
            
            else:
                pass

        
  









