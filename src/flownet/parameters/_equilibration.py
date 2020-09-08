from typing import Dict, Union, List, Optional
from itertools import combinations
import pathlib

import jinja2
import pandas as pd

from ..network_model import NetworkModel
from ..utils import write_grdecl_file
from .probability_distributions import UniformDistribution, LogUniformDistribution
from ._base_parameter import Parameter


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)


class Equilibration(Parameter):
    """
    Parameter type which takes care of stochastically drawn Equilibration parameters.

    Args
        distribution_values:
            A dataframe with five columns ("parameter", "minimum", "maximum",
            "loguniform", "eqlnum") which state:
                * The name of the parameter,
                * The minimum value of the parameter,
                * The maximum value of the parameter,
                * Whether the distribution is uniform of loguniform,
                * To which EQLNUM this applies.
        network: FlowNet network instance.
        ti2ci: A dataframe with index equal to tube model index, and one column which equals cell indices.
        eqlnum: A dataframe defining the EQLNUM for each flow tube.
        datum_depth: Depth of the datum(s) (m).
        rsvd: Pandas dataframe with a single rsvd table.

    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        distribution_values: pd.DataFrame,
        network: NetworkModel,
        ti2ci: pd.DataFrame,
        eqlnum: pd.DataFrame,
        datum_depth: Optional[List[float]] = None,
        rsvd: Optional[pd.DataFrame] = None,
    ):
        self._datum_depth: Union[List[float], None] = datum_depth
        self._rsvd: Union[pathlib.Path, None] = rsvd
        self._network: NetworkModel = network

        self._ti2ci: pd.DataFrame = ti2ci

        self._random_variables = [
            LogUniformDistribution(row["minimum"], row["maximum"])
            if row["loguniform"]
            else UniformDistribution(row["minimum"], row["maximum"])
            for _, row in distribution_values.iterrows()
        ]

        self._unique_eqlnums: List[int] = list(distribution_values["eqlnum"].unique())
        self._parameters: List[Parameter] = list(
            distribution_values["parameter"].unique()
        )
        self._eqlnum: pd.DataFrame = eqlnum

    def get_dims(self) -> Dict:
        """
        Function to export the table dimensions used for memory allocation in Flow.

        Returns:
            Dictionary containing all dimensions to set.

        """
        dims_dict = {"NTEQUL": len(self._unique_eqlnums)}

        return dims_dict

    def render_output(self) -> Dict:
        """
        Creates EQUIL and EQLNUM include content - which are given to the PROPS and GRID section.

        Returns:
            Dictionary with EQUIL and EQLNUM include content.

        """
        merged_df_eqlnum = self._ti2ci.merge(
            self._eqlnum, left_index=True, right_index=True
        )

        parameters = []
        samples_per_eqlnum = len(self.random_samples) // len(self._unique_eqlnums)
        for i, _ in enumerate(self._unique_eqlnums):
            param_value_dict = dict(
                zip(
                    self._parameters,
                    self.random_samples[
                        i * samples_per_eqlnum : (i + 1) * samples_per_eqlnum
                    ],
                )
            )
            parameters.append(param_value_dict)

        thpres = ""
        rsvd = ""
        eqlnum_combinations = []

        if len(self._unique_eqlnums) > 1:
            for connections_at_node in self._network.connection_at_nodes:
                eqlnum_combinations.extend(list(combinations(connections_at_node, 2)))

            eqlnum_combinations = list(set(eqlnum_combinations))
            eqlnum1 = list(list(zip(*eqlnum_combinations))[0])
            eqlnum2 = list(list(zip(*eqlnum_combinations))[1])

            thpres = _TEMPLATE_ENVIRONMENT.get_template("THPRES.jinja2").render(
                {
                    "eqlnum1": eqlnum1,
                    "eqlnum2": eqlnum2,
                }
            )

            if self._rsvd is not None:
                rsvd = _TEMPLATE_ENVIRONMENT.get_template("RSVD.jinja2").render(
                    {
                        "nr_eqlnum": len(self._unique_eqlnums),
                        "rsvd": pd.read_csv(self._rsvd, names=["depth", "rs"]),
                    }
                )

        return {
            "RUNSPEC": f"EQLDIMS\n{len(self._unique_eqlnums)} /\n",
            "REGIONS": write_grdecl_file(merged_df_eqlnum, "EQLNUM", int_type=True),
            "SOLUTION": _TEMPLATE_ENVIRONMENT.get_template("EQUIL.jinja2").render(
                {
                    "nr_eqlnum": len(self._unique_eqlnums),
                    "datum_depth": self._datum_depth,
                    "parameters": parameters,
                }
            )
            + rsvd
            + f"\n{thpres}",
        }
