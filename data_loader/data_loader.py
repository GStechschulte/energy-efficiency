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
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe33f53923d596335e69d41.csv', 'public.machine_1', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe340b3923d596335e69d47.csv', 'public.machine_2', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe3412c923d596335e69d4c.csv.csv', 'public.machine_3', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe341bd923d596335e69d51.csv', 'public.machine_4', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe3400d923d596335e69d42.csv', 'public.machine_5', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe340cb923d596335e69d48.csv', 'public.machine_6', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe34145923d596335e69d4d.csv.csv', 'public.machine_7', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe341d4923d596335e69d52.csv', 'public.machine_8', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe34044923d596335e69d43.csv', 'public.machine_9', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe340e3923d596335e69d49.csv', 'public.machine_10', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe3415e923d596335e69d4e.csv.csv', 'public.machine_11', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe341ee923d596335e69d53.csv', 'public.machine_12', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe34060923d596335e69d44.csv', 'public.machine_13', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe340fc923d596335e69d4a.csv.csv', 'public.machine_14', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe3417a923d596335e69d4f.csv.csv', 'public.machine_15', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/6017e77a42d6f4614409d192.csv', 'public.machine_16', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe34098923d596335e69d46.csv', 'public.machine_17', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe34116923d596335e69d4b.csv.csv', 'public.machine_18', ',')
    postgreSQL.load('/Users/wastechs/Documents/data/clemap/cat/5fe34191923d596335e69d50.csv', 'public.machine_19', ',')