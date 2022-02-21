import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot

class Prophet():

    def __init__(
        self, tau_prior_scale=0.05, multiplicative_seasonality=False, add_seasonality=None,
        trend_model='linear'):

        self.tau_prior_scale = tau_prior_scale
        self.multiplicative_seasonality = multiplicative_seasonality
        self.add_seasonality = add_seasonality
        self.trend_model = trend_model
    

    def build_gam(self):

        if self.multiplicative_seasonality == False:
            self.model = Prophet(
                changepoint_prior_scale=self.tau_prior_scale,
                growth=self.trend_model
            )
        else:
            self.model = Prophet(
                changepoint_prior_scale=self.tau_prior_scale,
                growth=self.trend_model,
                multiplicative_seasonality=self.multiplicative_seasonality
            )
        
        if self.add_seasonality is not None:
            assert type(self.add_seasonality) == dict

            name = self.add_seasonality['name']
            period = self.add_seasonality['period']
            fourier_order = self.add_seasonality['fourier_order']

            self.model.add_seasonality(
                name=name,
                period=period,
                fourier_order=fourier_order)
        else:
            pass

        return self
    

    def fit(self, training_data):
        self.model.fit(training_data)
    

    def predict(self, test_data):
        self.forecast = self.model.predict(test_data)

        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            self.forecast[col] = self.forecast[col].clip(lower=0.0)

        return self.forecast