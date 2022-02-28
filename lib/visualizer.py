from lib.util import helper
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_daily(agg):

    config = helper.get_config()

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

    table_names = pd.read_sql_query(query, helper.get_db_connection())
    machine_kw = {}

    for table in table_names.to_numpy():

        if agg == 'avg':
            query = """
            select
                date_part('day', p."t") as day,
                avg(p.avg_phase_kw) * 24 as avg_kwh_day
            from(
                select
                    k."t",
                    avg(k.kw) as avg_phase_kw	
                from (select
                        g."t",
                        g."p" / 1000 as kw
                    from {}.{} g) k
                group by k."t") p
            group by date_part('day', p."t")
            order by day;
            """.format(config['DB']['SCHEMA'], table[0])

            df_kw = pd.read_sql_query(query, helper.get_db_connection())
            machine_kw[table[0]] = list(df_kw.avg_kwh_day)

        elif agg == 'sum':
            query = """
            select
                date_part('day', p."t") as day,
                sum(p.avg_phase_kw) as sum_kw_day
            from(
                select
                    k."t",
                    avg(k.kw) as avg_phase_kw	
                from (select
                        g."t",
                        g."p" / 1000 as kw
                    from {}.{} g) k
                group by k."t") p
            group by date_part('day', p."t")
            order by day;
            """.format(config['DB']['SCHEMA'], table[0])

            df_kw = pd.read_sql_query(query, helper.get_db_connection())
            machine_kw[table[0]] = list(df_kw.sum_kw_day)

    return machine_kw


def visualize_hourly(single_machine=False):

    config = helper.get_config()
    hourly_table_names = helper.get_table_names(time='1H')
    machine_hourly_kw = {}

    if single_machine == True:
        query = """
        select
        """
    else:
        for table in hourly_table_names:
            query = """
            select
                date_part('hour', g."t") as hour,
                avg(kw) as avg_kw
            from
                {}.{} g
            group by date_part('hour', g."t")
            order by hour
            """.format(config['DB']['SCHEMA'], table[0])

            df_hourly_kw = pd.read_sql_query(query, helper.get_db_connection())
            machine_hourly_kw[table[0]] = list(df_hourly_kw.avg_kw)
        
        return machine_hourly_kw


def hourly_profile_heatmap():
    
    config = helper.get_config()
    table_names = helper.get_table_names(time=None)

    for table in table_names:
        query = """
        select
            *
        from
            {}.{} t
        """.format(config['DB']['SCHEMA'], table[0])

        machine_df = pd.read_sql_query(query, helper.get_db_connection())
        
        cats = [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]
        cat_type = CategoricalDtype(categories=cats, ordered=True)

        machine_df['day_name'] = machine_df.index.day_name().values
        machine_df['day_name'] = machine_df['day_name'].astype(cat_type)
        machine_df['hour'] = machine_df.index.hour
        
        hourly_profile = machine_df.group(['day_name', 'hour'])['kw'].mean().unstack()

        plt.figure(figsize=(12, 8))
        sns.heatmap(hourly_profile)
        plt.title(table[0], 'Hourly Load Heatmap')



#def weekly_profile_heatmap():


