from typing import List

import jinja2
import pandas as pd

from ..network_model import NetworkModel
from .probability_distributions import (
    UniformDistribution,
    LogUniformDistribution,
    NormalDistribution,
    LogNormalDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    Constant,
    ProbabilityDistribution,
)
from ._base_parameter import Parameter


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)


def _probability_distribution(row, param: str) -> ProbabilityDistribution:
    """

    Args:
        row:
        param:

    Returns:

    """
    if row[f"distribution_{param}"] == "uniform":
        return UniformDistribution(row[f"minimum_{param}"], row[f"maximum_{param}"])
    if row[f"distribution_{param}"] == "logunif":
        return LogUniformDistribution(row[f"minimum_{param}"], row[f"maximum_{param}"])
    if row[f"distribution_{param}"] == "normal":
        return NormalDistribution(row[f"mean_{param}"], row[f"stddev_{param}"])
    if row[f"distribution_{param}"] == "lognormal":
        return LogNormalDistribution(row[f"mean_{param}"], row[f"stddev_{param}"])
    if row[f"distribution_{param}"] == "truncated_normal":
        return TruncatedNormalDistribution(
            row[f"mean_{param}"],
            row[f"stddev_{param}"],
            row[f"minimum_{param}"],
            row[f"maximum_{param}"],
        )
    if row[f"distribution_{param}"] == "triangular":
        return TriangularDistribution(
            row[f"minimum_{param}"], row[f"base_{param}"], row[f"maximum_{param}"]
        )
    if row[f"distribution_{param}"] == "constant":
        return Constant(row[f"constant_{param}"])


class FaultTransmissibility(Parameter):
    """
    Parameter type which takes care of stochastically drawn fault transmissibility multiplier values.

    Args:
        distribution_values:
            A dataframe with five columns ("parameter", "minimum", "maximum",
            "loguniform") which state:
                * The name of the parameter,
                * The minimum value of the parameter,
                * The maximum value of the parameter,
                * Whether the distribution is uniform of loguniform
        network: FlowNet network instance.

    """

    def __init__(self, distribution_values: pd.DataFrame, network: NetworkModel):
        self._random_variables: List[ProbabilityDistribution] = [
            _probability_distribution(row, "fault_mult")
            for _, row in distribution_values.iterrows()
        ]

        self._network: NetworkModel = network
        self._unique_faults: int = len(distribution_values)

    def render_output(self) -> dict:
        """Creates FAULTS include content - which are given to the RUNSPEC and GRID section.

        Returns:
            Include content for aquifers

        """
        output_runspec = ""
        output_grid = ""
        output_edit = ""

        if self._network.faults:

            fault_segments = sum(
                {
                    key: len(value) for [key, value] in self._network.faults.items()
                }.values()
            )
            output_runspec = f"FAULTDIM\n{fault_segments} /\n"

            output_grid += _TEMPLATE_ENVIRONMENT.get_template(
                "FAULTS.inc.jinja2"
            ).render({"faults": self._network.faults})

            output_edit += _TEMPLATE_ENVIRONMENT.get_template(
                "MULTFLT.inc.jinja2"
            ).render({"faults": self._network.faults, "samples": self.random_samples})

        return {
            "RUNSPEC": output_runspec,
            "GRID": output_grid,
            "EDIT": output_edit,
        }
