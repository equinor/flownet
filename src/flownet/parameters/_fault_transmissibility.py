from typing import List

import jinja2
import pandas as pd

from ..network_model import NetworkModel
from .probability_distributions import ProbabilityDistribution
from ._base_parameter import Parameter, parameter_probability_distribution_class


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)


class FaultTransmissibility(Parameter):
    """
    Parameter type which takes care of stochastically drawn fault transmissibility multiplier values.

    Args:
        distribution_values:
            A dataframe with seven columns ("parameter", "minimum", "maximum", "mean", "base", "stddev",
            "distribution") which state:
                * The name of the parameter,
                * The minimum value of the parameter (set to None if not applicable),
                * The maximum value of the parameter (set to None if not applicable),
                * The mean value of the parameter,
                * The mode of the parameter distribution (set to None if not applicable),
                * The standard deviation of the parameter,
                * The type of probability distribution,
        network: FlowNet network instance.

    """

    def __init__(self, distribution_values: pd.DataFrame, network: NetworkModel):
        self._random_variables: List[ProbabilityDistribution] = [
            parameter_probability_distribution_class(row, "fault_mult")
            for _, row in distribution_values.iterrows()
        ]

        self._network: NetworkModel = network
        self._unique_faults: int = len(distribution_values)

    def render_output(self) -> dict:
        """Creates FAULTS include content - which are given to the RUNSPEC and GRID section.

        Returns:
            Include content for faults

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
