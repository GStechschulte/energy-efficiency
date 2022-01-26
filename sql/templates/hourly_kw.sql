-- Hourly kW
select
	date_part('hour', g."t") as hour,
	avg(kw) as avg_kw
from
	gassmann."g_5fe33f53923d596335e69d41_1H" g
group by date_part('hour', g."t")
order by hour