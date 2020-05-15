import numpy as np

from flownet.ahm._ahm_iteration_analytics import accuracy_metric


def test_calculation_accuracy_metric() -> None:
    assert (
        accuracy_metric(
            np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            np.array([[0.4, 0.4, 0.4], [0.6, 0.6, 0.6]]),
            "RMSE",
        )
        == 0.09999999999999998
    )
