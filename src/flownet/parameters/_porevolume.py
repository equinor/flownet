from typing import Dict, List, Optional

import jinja2
import pandas as pd

from ..network_model import NetworkModel
from ..utils.constants import C_DARCY
from ..utils import write_grdecl_file
from .probability_distributions import ProbabilityDistribution
from ._base_parameter import Parameter, parameter_probability_distribution_class


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)

class PoreVolume(Parameter):    
    def __init__(
        self,
        distribution_values: pd.DataFrame,
        ti2ci: pd.DataFrame,
        network: NetworkModel
       ):
        self._ti2ci: pd.DataFrame = ti2ci

        self._network: NetworkModel = network

        self._random_variables: List[ProbabilityDistribution] = (
            [  # Add random variables for bulk volume multipliers
                parameter_probability_distribution_class(row, "bulkvolume_mult")
                for _, row in distribution_values.iterrows()
            ]
            + [  # Add random variables for porosity
                parameter_probability_distribution_class(row, "porosity")
                for _, row in distribution_values.iterrows()
            ]
        )
        self._number_tubes: int = len(self._ti2ci.index.unique())


    def render_output(self) -> Dict:
        """
        Creates  PORV, which are given to the EDIT section.

        Returns:
             PORV

        """

        # Calculate  PORV

        properties_per_cell = pd.DataFrame(index=self._ti2ci.index)
        properties_per_cell["INITIAL_BULKVOLUME"] = self._network.initial_cell_volumes

        properties_per_cell["PORO"] = self._ti2ci.merge(
            pd.DataFrame(
                self.random_samples[self._number_tubes : 2 * self._number_tubes],
                index=self._ti2ci.index.unique(),
                columns=["PORO"],
            ),
            left_index=True,
            right_index=True,
        )["PORO"]

        properties_per_cell["BULKVOLUME_MULT"] = self._ti2ci.merge(
            pd.DataFrame(
                self.random_samples[: self._number_tubes],
                index=self._ti2ci.index.unique(),
                columns=["BULKVOLUME_MULT"],
            ),
            left_index=True,
            right_index=True,
        )["BULKVOLUME_MULT"]

        properties_per_cell["BULKVOLUME"] = (
            properties_per_cell["INITIAL_BULKVOLUME"]
            * properties_per_cell["BULKVOLUME_MULT"]
        )
        # TODO this part should be equal to the PoreVolume 
        # Obtained in flow diagnostics
        properties_per_cell["PORV"] = (
            properties_per_cell["BULKVOLUME"] * properties_per_cell["PORO"]
        )


        return {
            "GRID": write_grdecl_file(  # type: ignore[operator]
                properties_per_cell, "PORO"
            ),
            "EDIT": write_grdecl_file(  # type: ignore[operator]
                properties_per_cell, "PORV" 
            ),           
        }


        

