from typing import List

import jinja2
import pandas as pd

from ..network_model import NetworkModel
from .probability_distributions import (
    UniformDistribution,
    LogUniformDistribution,
    NormalDistribution,
    LogNormalDistribution,
    TruncatedNormalDistribution,
    TriangularDistribution,
    Constant,
    ProbabilityDistribution,
)
from ._base_parameter import Parameter

_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)


def _probability_distribution(row) -> ProbabilityDistribution:
    """

    Args:
        row:

    Returns:

    """
    if row[f"distribution"] == "uniform":
        return UniformDistribution(row[f"minimum"], row[f"maximum"])
    if row[f"distribution"] == "logunif":
        return LogUniformDistribution(row[f"minimum"], row[f"maximum"])
    if row[f"distribution"] == "normal":
        return NormalDistribution(row[f"mean"], row[f"stddev"])
    if row[f"distribution"] == "lognormal":
        return LogNormalDistribution(row[f"mean"], row[f"stddev"])
    if row[f"distribution"] == "truncated_normal":
        return TruncatedNormalDistribution(
            row[f"mean"],
            row[f"stddev"],
            row[f"minimum"],
            row[f"maximum"],
        )
    if row[f"distribution"] == "triangular":
        return TriangularDistribution(row[f"minimum"], row[f"base"], row[f"maximum"])
    if row[f"distribution"] == "constant":
        return Constant(row[f"constant"])


class Aquifer(Parameter):
    """Parameter type which takes care of stochastically drawn aquifer parameters.

    Args:
        distribution_values:
            A dataframe with five columns ("parameter", "minimum", "maximum",
            "loguniform", "aquid") which state:
                * The name of the parameter,
                * The minimum value of the parameter,
                * The maximum value of the parameter,
                * Whether the distribution is uniform of loguniform,
                * To which aquifer this applies.
        network: FlowNet network instance.
        scheme: If the scheme is to be global or not.

    """

    def __init__(
        self, distribution_values: pd.DataFrame, network: NetworkModel, scheme: str
    ):
        self._random_variables: List[ProbabilityDistribution] = [
            _probability_distribution(row) for _, row in distribution_values.iterrows()
        ]

        self._network: NetworkModel = network
        self._unique_aquids: List[int] = list(distribution_values["aquid"].unique())
        self._parameters: List[Parameter] = list(
            distribution_values["parameter"].unique()
        )
        self._scheme: str = scheme
        self._check_parameters()

    def _check_parameters(self):
        """Helper function to check the user-defined parameters.
        It will raise an error if something is wrong.

        Returns:
            Nothing

        """
        if not all(elem in self._parameters for elem in ["size_in_bulkvolumes"]):
            raise AssertionError(
                "Please specify all required parameters for the numerical aquifer:\n"
                '"size_in_bulkvolumes"'
            )

    def render_output(self) -> dict:
        """Creates aquifer include content - which are given to the PROPS and GRID section.

        Returns:
            Include content for aquifers

        """
        samples_per_aquid = len(self.random_samples) // len(self._unique_aquids)

        parameters = [
            dict(
                zip(
                    self._parameters,
                    self.random_samples[
                        i * samples_per_aquid : (i + 1) * samples_per_aquid
                    ],
                )
            )
            for i, _ in enumerate(self._unique_aquids)
        ]

        return {
            "EDIT": _TEMPLATE_ENVIRONMENT.get_template(
                "NUMERICAL_AQUIFER.jinja2"
            ).render(
                {
                    "aquifer_i": self._network.aquifers_i,
                    "total_bulkvolume": self._network.total_bulkvolume,
                    "unique_aquids": self._unique_aquids,
                    "parameters": parameters,
                }
            )
        }
