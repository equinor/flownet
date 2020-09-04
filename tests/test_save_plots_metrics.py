import os.path

import pandas as pd

from flownet.ahm._ahm_iteration_analytics import save_plots_metrics


def test_save_plots_metrics() -> None:
    d = {'quantity': ['WOPR', 'WWPR'], 'iteration': [0, 1], 'RMSE': [0.6, 0.3], 'MAE': [0.5, 0.3]}
    df_metrics = pd.DataFrame(data=d)
    metrics = ['RMSE', 'MAE']
    str_key = 'WOPR'
    save_plots_metrics(df_metrics, metrics, str_key)
    assert os.path.isfile("metric_RMSE_WOPR.png") and os.path.isfile("metric_MAE_WOPR.png")
