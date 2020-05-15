import numpy as np
import pandas as pd

from flownet.ahm._ahm_iteration_analytics import prepare_flownet_data


def test_prepare_flownet_data() -> None:
    data = {
        "realization_id": [1, 1, 2, 2],
        "key_1": [11, 12, 21, 22],
        "key_2": [13, 14, 23, 24],
    }
    assert np.allclose(
        prepare_flownet_data(pd.DataFrame(data, columns=["key_1", "key_2"]), "key_", 2),
        np.array([[11, 21], [12, 22], [13, 23], [14, 24]]),
    )
