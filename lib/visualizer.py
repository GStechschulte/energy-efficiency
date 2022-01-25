from lib.util import helper
import pandas as pd
import numpy as np

def visualize_daily():

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
                    g."P" / 1000 as kw
                from gassmann.{} g) k
            group by k."t") p
        group by date_part('day', p."t")
        order by day;
        """.format(table[0])

        df_kw = pd.read_sql_query(query, helper.get_db_connection())
        machine_kw[table[0]] = list(df_kw.avg_kwh_day)

    return machine_kw

