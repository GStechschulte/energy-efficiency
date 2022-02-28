from audioop import mul
from cgi import test
from black import out
import pandas as pd
from numpy import array
from pyparsing import col
from sklearn import utils
from sklearn.preprocessing import OrdinalEncoder
from lib.util import helper
from lib.util.data_preprocessing import preprocess
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from lib.util import helper



update = False

# 1.) Retrieve preprocessed data
def gam_model(min_train_date, end_train_date, end_test_date, tau_prior,
multiplicative_seasonality='additive', add_seasonality=None, trend_model='linear',
tables=list, update_score=False): 
    """
    Preprocess each machine's data in the table list

    Parameters
    ----------
    tables: list of machines to query data from

    Returns
    -------
    dfs: preprocessed machine data with only the 
    working days of the week and rounded kW / Watt 
    values
    """
    global update
    update = update_score

    assert type(tables) == list

    for machine in tables:
        
        print('Preprocessing data for {}'.format(machine))

        training_data, test_data = preprocess(
            table=machine, min_train_date=min_train_date,
            end_train_date=end_train_date, end_test_date=end_test_date
            )

        print('Preprocessed training and test data for {} is complete'
        .format(machine))
        print('Starting GAM validation for {}'.format(machine))

        validate_gam(
            training_data, test_data, tau_prior, multiplicative_seasonality,
            add_seasonality, trend_model
            )

        print('GAM validation complete for {}'.format(machine))


def validate_gam(training_data, test_data, tau_prior, 
multiplicative_seasonality, add_seasonality, trend_model):
    """
    Preprocess each machine's data in the table list

    Parameters
    ----------
    tables: list of machines to query data from

    Returns
    -------
    
    """
    #assert type(tau_prior) == list or type(tau_prior) == array

    model_name = 'Prophet'

    if type(tau_prior) == list or type(tau_prior) == array:
        for tau in tau_prior:
            model_prophet = Prophet(
                changepoint_prior_scale=tau,
                growth=trend_model,
                seasonality_mode=multiplicative_seasonality,
            )

            if add_seasonality is not None:
                assert type(add_seasonality) == dict

                name = add_seasonality['name']
                period = add_seasonality['period']
                fourier_order = add_seasonality['fourier_order']

                model_prophet.add_seasonality(
                    name=name,
                    period=period,
                    fourier_order=fourier_order
                )
            else:
                pass

            # Fit a model to each tau value
            model_prophet.fit(df=training_data)

            # Make predictions according to this value
            min_train_date = pd.Timestamp(training_data['ds'].min())
            end_test_date = pd.Timestamp(test_data['ds'].max())
            frequency = pd.infer_freq(training_data['ds'])

            forecast_time_period = pd.DataFrame(
                data=pd.date_range(
                    min_train_date, end_test_date, freq=frequency
                ),
                columns=['ds']
            )

            # Compute test metrics
            predictions = model_prophet.predict(forecast_time_period)
            
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                predictions[col] = predictions[col].clip(lower=0.0)
            
            train_test = pd.concat([training_data, test_data])
            train_test.reset_index(inplace=True)

            predictions['actual'] = train_test['y']
        
            out_of_sample = predictions[predictions['ds'] > training_data['ds'].max()]
            ground_truth_out = out_of_sample['actual']
            predictions_out = out_of_sample['yhat']

            in_sample = predictions[predictions['ds'] <= training_data['ds'].max()]
            ground_truth_in = in_sample['actual']
            predictions_in = in_sample['yhat']

            out_sample_metrics = compute_metrics(ground_truth_out, predictions_out)
            in_sample_metrics = compute_metrics(ground_truth_in, predictions_in)

            print('Tau Prior = {} - Forecast RMSE = {} - In sample RMSE = {}'
            .format(tau, out_sample_metrics, in_sample_metrics))

            # Send test metrics to postgreSQL
            if update == True:
                helper.update_gam_metrics(
                    model=model_name,
                    tau=tau,
                    add_regressor=add_seasonality,
                    trend=trend_model,
                    out_sample_rmse=out_sample_metrics,
                    in_sample_rmse=in_sample_metrics
                )

    else:
        pass ## for right now, just pass

    # Optional: plot results

#def gp_model()


def compute_metrics(ground_truth, predictions):

    ## RMSE
    rmse = mean_squared_error(ground_truth, predictions)

    ## MAPE

    return rmse
