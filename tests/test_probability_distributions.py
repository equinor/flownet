import pandas as pd

from ...flownet.parameters._base_parameter import parameter_probability_distribution_class
from ...flownet.parameters.probability_distributions import (
    ProbabilityDistribution,
    UniformDistribution,
    LogUniformDistribution,
    NormalDistribution,
    LogNormalDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    Constant,
)

DATA = {
    "parameter": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
    "minimum": [1, 1, None, 1, 1, None, None, None, 0],
    "maximum": [2, None, 3, 2, None, 3, None, None, 5],
    "mean": [None, 2, 2, 3, 2, 3, 3, 3],
    "base": [None, None, None, None, None, None, None, None, None],
    "stddev": [None, None, None, None, None, None, 1, 1, 1],
    "distribution": ["uniform", "uniform", "uniform", "logunif", "logunif", "logunif", "normal", "lognormal", "truncated_normal"]
}

DISTRIBUTION_DF = pd.DataFrame(DATA)

def test_probability_distributions() -> None:
    probdist = List[ProbabilityDistribution] = [
        parameter_probability_distribution_class(row)
        for _, row in DISTRIBUTION_DF.iterrows()]
        