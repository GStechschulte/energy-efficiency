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

drop table if exists sensors.gp_metrics;
create table if not exists sensors.gp_metrics
(
	model character varying,
	time_agg character varying,
	kernel character varying,
	lr double precision,
	training_iter double precision,
	optimizer character varying,
	out_sample_mse double precision,
	out_sample_mape double precision
);

