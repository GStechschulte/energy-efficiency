drop table if exists sensors.metrics;
create table if not exists sensors.metrics
(
	model character varying,
	tau double precision,
	add_regressor character varying,
	trend character varying,
	out_sample_rmse double precision,
	in_sample_rmse double precision
);