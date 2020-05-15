import numpy as np
import pandas as pd

from flownet.ahm._ahm_iteration_analytics import prepare_opm_reference_data


def test_prepare_opm_reference_data() -> None:
    data = {"key_1": [1, 2], "key_2": [3, 4]}
    assert np.allclose(
        prepare_opm_reference_data(
            pd.DataFrame(data, columns=["key_1", "key_2"]), "key_", 2
        ),
        np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
    )
