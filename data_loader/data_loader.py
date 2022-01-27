from sqlalchemy import create_engine
import sys
import pandas as pd
import os
sys.path.append('/Users/wastechs/Documents/git-repos/energy-efficiency')
import db_config as config

class data_preprocess():
    
    def time(df):
        """
        Convert df['t'] epochs to datetime
        return: df['t'] as a timestamp
        """
        missing = df.isna().sum().values
        assert missing.any() == 0

        df.t = pd.to_datetime(df.t, unit='s')
        df.set_index(df.t, inplace=True)
        del df['t']

        return df

if __name__ == '__main__':
    
    # List all machine csv files in "your" directory
    path = '/Users/wastechs/Documents/data/clemap/cat/'
    files = os.listdir(path)

    # Create DB engine from db_config file
    engine = create_engine(
        'postgresql+psycopg2://{}:{}@{}/{postgres}'.format(
            config.sql['user'],
            config.sql['password'],
            config.sql['host'],
            config.sql['db_name']
            )
        )
    
    for file in files:
        table_name = file.replace('.csv', '')
        df = pd.read_csv(path + "{}".format(file))
        df_preprocess = data_preprocess.time(df)
        df_preprocess.to_sql(
            "{}".format(table_name), 
            con=engine, 
            schema='gassmann', 
            if_exists='replace', index=True)

        print(table_name, 'has been uploaded to postgreSQL')