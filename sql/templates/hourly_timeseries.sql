drop materialized view if exists gassmann.g_5fe33f53923d596335e69d41_hourly;
create materialized view gassmann.g_5fe33f53923d596335e69d41_hourly as
select
	date_trunc('hour', g."t") as t,
	avg(g."V") as avg_hourly_v,
	sum(g."V") as sum_hourly_v,
	avg(g."I") as avg_hourly_i,
	sum(g."I") as sum_hourly_i,
	avg(g."S") as avg_hourly_s,
	sum(g."S") as sum_hourly_s,
	avg(g."P") as avg_hourly_p,
	sum(g."P") as sum_hourly_p,
	avg(g."Q") as avg_hourly_q,
	sum(g."Q") as sum_hourly_q,
	avg(g."PF") as avg_hourly_pf,
	sum(g."PF") as sum_hourly_pf,
	avg(g."PHI") as avg_hourly_phi,
	sum(g."PHI") as sum_hourly_phi
from gassmann.g_5fe33f53923d596335e69d41 g
group by
	date_trunc('hour', g."t")
order by t