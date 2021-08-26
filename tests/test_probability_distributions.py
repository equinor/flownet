from typing import List

import numpy as np
import pandas as pd


from flownet.parameters._base_parameter import parameter_probability_distribution_class
from flownet.parameters.probability_distributions import ProbabilityDistribution


DATA = {
    "parameter": [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "P10",
        "P11",
        "P12",
        "P13",
        "P14",
    ],
    "minimum": [
        1,
        1,
        None,
        1,
        1,
        None,
        None,
        None,
        0,
        1,
        1,
        1,
        None,
        None,
    ],
    "maximum": [
        2,
        None,
        3,
        2,
        None,
        3,
        None,
        None,
        5,
        3,
        3,
        None,
        3,
        None,
    ],
    "mean": [
        None,
        2,
        2,
        None,
        3,
        2,
        3,
        3,
        3,
        None,
        2,
        2,
        2,
        None,
    ],
    "base": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        2,
        None,
        2,
        2,
        1,
    ],
    "stddev": [
        None,
        None,
        None,
        None,
        None,
        None,
        1,
        1,
        1,
        None,
        None,
        None,
        None,
        None,
    ],
    "distribution": [
        "uniform",
        "uniform",
        "uniform",
        "logunif",
        "logunif",
        "logunif",
        "normal",
        "lognormal",
        "truncated_normal",
        "triangular",
        "triangular",
        "triangular",
        "triangular",
        "const",
    ],
}

PD_LOOKUP = {
    "minimum": "minimum",
    "maximum": "maximum",
    "mean": "mean",
    "base": "mode",
    "stddev": "stddev",
}

DISTRIBUTION_DF = pd.DataFrame(DATA)
# NaNs to None
DISTRIBUTION_DF = DISTRIBUTION_DF.replace({np.nan: None})


def test_probability_distributions() -> None:
    probdist: List[ProbabilityDistribution] = [
        parameter_probability_distribution_class(row)
        for _, row in DISTRIBUTION_DF.iterrows()
    ]
    for i in range(len(DATA.get("parameter"))):
        assert probdist[i].name.lower() == DATA["distribution"][i]
        for var in ["minimum", "maximum", "mean", "base", "stddev"]:
            if DATA[var][i] is not None:
                assert getattr(probdist[i], PD_LOOKUP.get(var)) == DATA[var][i]
