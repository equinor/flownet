import math
import numpy as np
import pandas as pd

from flownet.ert.forward_models._iteration_analytics import (
    prepare_opm_reference_data,
    prepare_flownet_data,
    normalize_data,
    accuracy_metric,
)


def test_prepare_opm_reference_data() -> None:
    data = {"key_1": [1, 2], "key_2": [3, 4]}
    assert np.allclose(
        prepare_opm_reference_data(
            pd.DataFrame(data, columns=["key_1", "key_2"]), "key_", 2
        ),
        np.array([[1, 1], [3, 3], [2, 2], [4, 4]]),
    )


def test_prepare_flownet_data() -> None:
    data = {
        "realization_id": [1, 1, 2, 2],
        "key_1": [11, 12, 21, 22],
        "key_2": [13, 14, 23, 24],
    }
    assert np.allclose(
        prepare_flownet_data(pd.DataFrame(data, columns=["key_1", "key_2"]), "key_", 2),
        np.array([[11, 21], [13, 23], [12, 22], [14, 24]]),
    )


def test_normalize_data() -> None:
    data_1 = np.array([[1.0, 1.0], [2.0, 2.0]])
    data_2 = [np.array([[1.1, 0.9], [2.1, 1.9]]), np.array([[1.2, 0.8], [2.2, 1.8]])]
    data_3 = np.array([[4.0, 4.0], [4.0, 4.0]])
    data_4 = [np.array([[4.1, 3.9], [4.1, 3.9]]), np.array([[4.2, 3.8], [4.2, 3.8]])]
    data_5 = np.array([[0.0, 0.0], [0.0, 0.0]])
    data_6 = [np.array([[0.1, 0.0], [0.1, 0.0]]), np.array([[0.2, 0.0], [0.2, 0.0]])]
    tmp_1 = normalize_data(data_1, data_2)
    tmp_2 = normalize_data(data_3, data_4)
    tmp_3 = normalize_data(data_5, data_6)
    res_1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    res_2 = [np.array([[0.1, -0.1], [1.1, 0.9]]), np.array([[0.2, -0.2], [1.2, 0.8]])]
    res_3 = np.array([[0.0, 0.0], [0.0, 0.0]])
    res_4 = [
        np.array([[0.025, -0.025], [0.025, -0.025]]),
        np.array([[0.05, -0.05], [0.05, -0.05]]),
    ]
    res_5 = np.array([[0.0, 0.0], [0.0, 0.0]])
    res_6 = [np.array([[0.1, 0.0], [0.1, 0.0]]), np.array([[0.2, 0.0], [0.2, 0.0]])]
    assert (
        np.allclose(tmp_1[0], res_1)
        and all(np.allclose(x, y) for x, y in zip(tmp_1[1], res_2))
        and np.allclose(tmp_2[0], res_3)
        and all(np.allclose(x, y) for x, y in zip(tmp_2[1], res_4))
        and np.allclose(tmp_3[0], res_5)
        and all(np.allclose(x, y) for x, y in zip(tmp_3[1], res_6))
    )


def test_calculation_accuracy_metric() -> None:
    assert math.isclose(
        accuracy_metric(
            np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            np.array([[0.4, 0.4, 0.4], [0.6, 0.6, 0.6]]),
            "RMSE",
        ),
        0.1,
    )
