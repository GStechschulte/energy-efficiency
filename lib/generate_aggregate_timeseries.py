from lib.util import helper
import pandas as pd
import os
from os.path import isfile, join
from os import listdir
import db_config as config

def run_timeseries(custom=False, time_agg=list):

    if custom != True:
        time_resolution = ['1T', '5T', '10T', '15T', '30T', '1H']
    else:
        assert time_agg != list
        minutes = time_agg

    query = """
    select 
        table_name
    from information_schema.tables
    where table_name like 'g%'
    """

    table_names = pd.read_sql_query(query, helper.get_db_connection())

    for table in table_names.to_numpy():

        config = helper.get_config()

        query = """
        select
            *
        from {}.{}
        """.format(config['DB']['SCHEMA'], table[0])
        
        df = pd.read_sql_query(query, helper.get_db_connection(), index_col='t')

        for time in time_resolution:

            feats = ['V', 'I', 'S', 'P', 'Q', 'PF', 'PHI']
            print('Creating', time, 'time aggregation materialized view for', table[0])

            df_time_agg = pd.DataFrame(
            data=df[feats].resample(time).mean(),
            index=df.resample(time).sum().index
            )

            df_time_agg['kw'] = df_time_agg['P'] / 1000
            table_name = table[0] + '_{}'.format(time)

            df_time_agg.to_sql(
                "{}".format(table_name), 
                con=helper.get_db_connection(engine=True), 
                index=True,
                if_exists='replace',
                schema='{}'.format(config['DB']['SCHEMA'],))

        print('Time aggregation materialized view for', table[0], 'completed')
        
  

"""
def run_files(foldername, regex):

    dirname = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(dirname)

    filepath = os.path.join(dirname, 'sql/files/' + foldername)

    filenames = get_files(filepath)

def get_files(filepath):
    
    files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
"""







