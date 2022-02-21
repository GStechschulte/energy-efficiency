create table if not exists sensors.r707lv_f4032
(
	t double precision,
	l numeric,
	v double precision,
	i double precision,
	s double precision,
	p double precision,
	q double precision,
	pf double precision,
	phi double precision
);

alter table sensors.vk_1_ug
alter column t type timestamp using to_timestamp(t::double precision);