from sqlalchemy import create_engine
import sys
import pandas as pd
import os
sys.path.append('/Users/wastechs/Documents/git-repos/energy-efficiency')
import db_config as config
from lib.util import helper
import psycopg2

class data_loader():

    def __init__(self, local):
        self.conn = helper.get_db_connection(local=local)

        try:
            print(self.conn)
            self.cursor = self.conn.cursor()
        except:
            print('Connection not established')

    def conn(self):
        return self.conn
    
    def load(self, csv, table_name, delimiter):
        with open(csv, 'r') as f:
            next(f)
            self.cursor.copy_from(f, table_name, sep='{}'.format(delimiter))
        self.conn.commit()

if __name__ == '__main__':
    
    path = '/Users/wastechs/Documents/data/clemap/cat_final/'
    files = os.listdir(path)
    postgreSQL = data_loader(local=True)
    sensors = helper.get_sensor_map()
    
    for file in files:
        sensor_name = file.replace('.csv', '')
        for key, value in sensors.items():
            if key == sensor_name:
                table = value
        
        full_path = path + file
        print(file, 'is being uploaded to postgreSQL table', table)
        postgreSQL.load("{}".format(full_path), 'gassmann.{}'.format(table), ',')
        print(file, 'has been uploaded to postgreSQL table', table)