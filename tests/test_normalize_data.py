import numpy as np

from flownet.ahm._ahm_iteration_analytics import normalize_data


def test_normalize_data() -> None:
    data_1 = np.array([[1.0, 1.0], [2.0, 2.0]])
    data_2 = [np.array([[1.1, 0.9], [2.1, 1.9]]), np.array([[1.2, 0.8], [2.2, 1.8]])]
    tmp = normalize_data(data_1, data_2)
    res_1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    res_2 = [np.array([[0.1, -0.1], [1.1, 0.9]]), np.array([[0.2, -0.2], [1.2, 0.8]])]
    assert np.allclose(tmp[0], res_1) and all(
        [np.allclose(x, y) for x, y in zip(tmp[1], res_2)]
    )
