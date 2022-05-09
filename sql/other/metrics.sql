select 
	*,
	sqrt(out_sample_mse) as rmse
from 
	sensors.gp_metrics
where
	model like '%gp%'
and 
	out_sample_ace is not null
order by 
	out_sample_mse asc,
	out_sample_mape asc, out_sample_pinball asc, out_sample_ace desc 
	
	
