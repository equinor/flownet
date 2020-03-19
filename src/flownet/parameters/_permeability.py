from typing import Dict, List

import jinja2
import pandas as pd

from ..network_model import NetworkModel
from ..utils import write_grdecl_file
from .probability_distributions import (
    UniformDistribution,
    LogUniformDistribution,
    ProbabilityDistribution,
)
from ._base_parameter import Parameter


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)


class Permeability(Parameter):
    """
    Parameter type which takes care of stochastically drawn permeability values.

    Args:
        distribution_values:
            A dataframe with index equal to tube model index, and with three columns
            ('minimum', 'maximum', 'loguniform') which states the minimum allowed
            permeability value, maximum allowed permeability value, and one boolean
            value if it should be loguniform (False implies uniform).
        ti2ci: A dataframe with index equal to tube model index, and one column which equals cell indices.
        network: The FlowNet NetworkModel instance.

    """

    def __init__(
        self,
        distribution_values: pd.DataFrame,
        ti2ci: pd.DataFrame,
        network: NetworkModel,
    ):
        self._ti2ci: pd.DataFrame = ti2ci
        self._network: NetworkModel = network

        self._random_variables: List[ProbabilityDistribution] = [
            LogUniformDistribution(row["minimum"], row["maximum"])
            if row["loguniform"]
            else UniformDistribution(row["minimum"], row["maximum"])
            for _, row in distribution_values.iterrows()
        ]

    def render_output(self) -> Dict:
        """
        Creates PERMX, PERMY and PERMZ include content - which are given to the
        GRID section.

        Returns:
             PERMX, PERMY and PERMZ include content.

        """
        perm_per_tube = pd.DataFrame(
            self.random_samples, index=self._ti2ci.index.unique(), columns=["PERMX"]
        )
        perm_per_tube["PERMY"] = perm_per_tube["PERMX"]
        perm_per_tube["PERMZ"] = perm_per_tube["PERMX"]

        merged_df = self._ti2ci.merge(perm_per_tube, left_index=True, right_index=True)

        output = ""
        output += write_grdecl_file(merged_df, "PERMX")  # type: ignore[operator]
        output += write_grdecl_file(merged_df, "PERMY")  # type: ignore[operator]
        output += write_grdecl_file(merged_df, "PERMZ")  # type: ignore[operator]

        if self._network.nncs:
            output += _TEMPLATE_ENVIRONMENT.get_template("NNC.inc.jinja2").render(
                {"nncs": self._network.nncs}
            )

        return {"GRID": output}
