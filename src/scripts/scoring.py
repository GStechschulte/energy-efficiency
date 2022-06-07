import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, \
    mean_squared_error, mean_pinball_loss


def scoring_metrics(ground_truth, test_preds, lower_preds, 
upper_preds) -> np.array:
    """
    Compute the evaluation metrics on the inverse 
    transformed test data
    """

    mean_pb_loss = mean_pinball_loss(ground_truth, test_preds)

    indicator = []
    for x, low, up in zip(ground_truth, lower_preds, upper_preds):
            if x <= up and x >= low:
                indicator.append(1)
            else:
                indicator.append(0)

    ace = sum(indicator) / len(ground_truth)

    mse = mean_squared_error(ground_truth, test_preds)
    mape = mean_absolute_percentage_error(ground_truth, test_preds)

    print('\n', 'Scoring Metrics')
    print('-'*20)
    print('MSE       = ', round(mse, 4))
    print('RMSE      = ', round(np.sqrt(mse), 4))
    print('MAPE      = ', round(mape, 4))
    print('ACE       = ', round(ace, 4))
    print('Pinball   = ', round(mean_pb_loss, 4))