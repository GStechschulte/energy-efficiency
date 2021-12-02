#from typing_extensions import ParamSpec
import psycopg2
import csv
import sys
#sys.path.append('/Users/wastechs/Documents/git-repos/data-mining')
#import db_config as config

class data_loader():

    def __init__(self, host, dbname, user, password):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password

        self.conn = psycopg2.connect(
            host=self.host, 
            dbname=self.dbname, 
            user=self.user,
            password=self.password,
            port='5432')

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
    
    # Enter DB information in db_config file
    postgreSQL = data_loader(host='gstech01-hs21.enterpriselab.ch', 
                            dbname='postgres',
                            user='gabe',
                            password='clemap')

    # Machines
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_1.csv', 'public.machine_1', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_2.csv', 'public.machine_2', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_3.csv', 'public.machine_3', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_4.csv', 'public.machine_4', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_5.csv', 'public.machine_5', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_6.csv', 'public.machine_6', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_7.csv', 'public.machine_7', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_8.csv', 'public.machine_8', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_9.csv', 'public.machine_9', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_10.csv', 'public.machine_10', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_10.csv', 'public.machine_11', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_12.csv', 'public.machine_12', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_13.csv', 'public.machine_13', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_14.csv', 'public.machine_14', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_15.csv', 'public.machine_15', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_16.csv', 'public.machine_16', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_17.csv', 'public.machine_17', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_18.csv', 'public.machine_18', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/machines/machine_18.csv', 'public.machine_19', ',')