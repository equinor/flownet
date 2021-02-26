from typing import Dict, List

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


def _transmissibility(  # pylint: disable=too-many-arguments
    area: float,
    dx_i: float,
    dx_j: float,
    z_i: float,
    z_j: float,
    k_i: float,
    k_j: float,
    multx_i: float,
) -> float:
    """The equations here for calculating transmissibilities are taken from
    the simulation manual. The simulator will calculate transmissibilities automatically
    for everything except for NNC definitions.

    Args:
        area: The (constant) Flownet tube cross section area
        dx_i: Length cell i
        dx_j: Length cell j
        z_i: Depth cell i
        z_j: Depth cell j
        k_i: Permeability cell i
        k_j: Permeability cell j
        multx_i: Transmissibility multiplier for cell i

    """
    # pylint: disable=invalid-name
    TMLTX_i = multx_i

    A = (area * dx_i + area * dx_i) / (dx_i + dx_j)
    B = (dx_i / k_i + dx_j / k_j) / 2

    DHS = ((dx_i + dx_j) / 2) ** 2
    DVS = (z_i - z_j) ** 2

    DIPC = DHS / (DHS + DVS)

    return C_DARCY * TMLTX_i * A * DIPC / B


class PorvPoroTrans(Parameter):
    """
    Parameter type which takes care of stochastically drawn pore volume, porosity and
    permeability. Also makes consistent NNC transmissibilities.

    This is done by the following process:
      1) Porosity is drawn as random variables and inserted into the GRID section.
      2) For each cell in the network, there is assigned an "initial bulk volume" which
         is calculated by distributing the total convex hull bulk volume to each cell
         based on cell lengths.
      2) Random variables corresponding to volume multiplieres are drawn for each tube.
         The length-distributed bulk-volumes are then multiplied by this multiplier, and finally
         the random porosity in order to get the modified pore volume
         (which is added to the EDIT section).

    To account for the cross sectional area that should change when the pore volume changes,
    the transmissibility TRANX' between cells will be multiplied with a correction factor f,

    TRANX = (BULKVOLUME / (A * L)) * TRANX' = f * TRANX'

    where,
        A is the cross-sectional area between two cells in a flow tube,
        L is the the length of a single grid cell in a flow tube,
        TRANX' is the simulator pre-calculated transmissibility,
        BULKVOLUME is the, via the history matching process, assigned bulk volume of a single cell.

    After BULKVOLUME and PORO(SITY) have been drawn, PORV is calculated using PORV = BULKVOLUME * PORO.
    Both the PORV and the new transmissibility mutiplier f will be written to the EDIT section, while PORO
    is written to the grid section.

    Args:
        distribution_values:
            A dataframe with index equal to tube model index, and with eighteen columns
            specifiying min, max, mean, mode, stddev and distribution for three different parameters
            (bulkvolume_mult, porosity and permeability).
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

        self._random_variables: List[ProbabilityDistribution] = (
            [  # Add random variables for bulk volume multipliers
                parameter_probability_distribution_class(row, "bulkvolume_mult")
                for _, row in distribution_values.iterrows()
            ]
            + [  # Add random variables for porosity
                parameter_probability_distribution_class(row, "porosity")
                for _, row in distribution_values.iterrows()
            ]
            + [  # Add random variables for permeability
                parameter_probability_distribution_class(row, "permeability")
                for _, row in distribution_values.iterrows()
            ]
        )

        self._network: NetworkModel = network
        self._number_tubes: int = len(self._ti2ci.index.unique())

    def render_output(self) -> Dict:
        """
        Creates PORO, PORV, PERMX, PERMY, PERMZ and NNC include content - which are given to
        the GRID and EDIT section respectively.

        Returns:
             PORO, PORV, PERMX, PERMY, PERMZ and NNC include content.

        """

        # Calculate PORO, PORV AND MULTX

        properties_per_cell = pd.DataFrame(index=self._ti2ci.index)

        properties_per_cell["INITIAL_BULKVOLUME"] = (
            self._network.total_bulkvolume
            * self._network.grid["cell_length"].values
            / self._network.grid["cell_length"].sum()
        )

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

        properties_per_cell["MULTX"] = properties_per_cell["BULKVOLUME"].values / (
            self._network.area * self._network.grid["cell_length"].values
        )

        properties_per_cell["PORV"] = (
            properties_per_cell["BULKVOLUME"] * properties_per_cell["PORO"]
        )

        # Calculate updated base permeability (and NNC transmissibility)

        perm_per_tube = pd.DataFrame(
            self.random_samples[2 * self._number_tubes :],
            index=self._ti2ci.index.unique(),
            columns=["PERMX"],
        )
        perm_per_tube["PERMY"] = perm_per_tube["PERMX"]
        perm_per_tube["PERMZ"] = perm_per_tube["PERMX"]

        merged_df = self._ti2ci.merge(perm_per_tube, left_index=True, right_index=True)

        properties_per_cell.loc[merged_df["PERMX"] < 50, ["PORV"]] = 0

        output = ""
        output += write_grdecl_file(merged_df, "PERMX")  # type: ignore[operator]
        output += write_grdecl_file(merged_df, "PERMY")  # type: ignore[operator]
        output += write_grdecl_file(merged_df, "PERMZ")  # type: ignore[operator]

        if self._network.nncs:
            nnc_trans = []
            for nnc in self._network.nncs:
                dx_i = self._network.grid.cell_length[nnc[0]]
                dx_j = self._network.grid.cell_length[nnc[1]]

                z_i = self._network.grid.z_mean[nnc[0]]
                z_j = self._network.grid.z_mean[nnc[1]]

                k_i = merged_df.PERMX.iloc[nnc[0]]
                k_j = merged_df.PERMX.iloc[nnc[1]]

                multx_i = properties_per_cell.MULTX.iloc[nnc[0]]

                nnc_trans.append(
                    _transmissibility(
                        self._network.area, dx_i, dx_j, z_i, z_j, k_i, k_j, multx_i
                    )
                )

            output += _TEMPLATE_ENVIRONMENT.get_template("NNC.inc.jinja2").render(
                {"nncs": self._network.nncs, "nnc_trans": nnc_trans}
            )

        return {
            "GRID": write_grdecl_file(  # type: ignore[operator]
                properties_per_cell, "PORO"
            )
            + write_grdecl_file(properties_per_cell, "MULTX")  # type: ignore[operator]
            + output,
            "EDIT": write_grdecl_file(  # type: ignore[operator]
                properties_per_cell, "PORV"
            ),
        }
