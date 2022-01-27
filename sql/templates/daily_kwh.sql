-- Daily kWH
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
		  from gassmann.g_5fe33f53923d596335e69d41 g) k
	group by k."t") p
group by date_part('day', p."t")
order by day;