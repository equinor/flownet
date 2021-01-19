from typing import Dict, List, Tuple, Union, Optional
import concurrent.futures
import functools

import pandas as pd
from pyscal import WaterOilGas, WaterOil, GasOil, PyscalFactory, PyscalList

from configsuite import ConfigSuite
from ..utils import write_grdecl_file
from ..utils.constants import H_CONSTANT

from .probability_distributions import ProbabilityDistribution
from ._base_parameter import Parameter, parameter_probability_distribution_class


def gen_wog(parameters: pd.DataFrame, fast_pyscal: bool = False) -> WaterOilGas:
    """
    Creates a PyScal WaterOilGas object based on the input parameters supplied.

    Args:
        parameters: A dataframe consisting of all specified parameters.
        fast_pyscal: Run pyscal in fast-mode skipping checks. Useful for large models/ensemble.

    Returns:
        A PyScal WaterOilGas object

    """
    wog_relperm = WaterOilGas(
        swirr=parameters["swirr"],
        swl=parameters["swl"],
        swcr=parameters["swcr"],
        sorw=parameters["sorw"],
        sorg=parameters["sorg"],
        sgcr=parameters["sgcr"],
        h=H_CONSTANT,
        fast=fast_pyscal,
    )

    wog_relperm.wateroil.add_corey_water(
        nw=parameters["nw"], krwend=parameters["krwend"]
    )
    wog_relperm.wateroil.add_corey_oil(
        now=parameters["now"], kroend=parameters["kroend"]
    )
    wog_relperm.gasoil.add_corey_gas(ng=parameters["ng"], krgend=parameters["krgend"])
    wog_relperm.gasoil.add_corey_oil(nog=parameters["nog"], kroend=parameters["kroend"])

    return wog_relperm


def gen_wo(parameters: pd.DataFrame, fast_pyscal: bool = False) -> WaterOil:
    """
    Creates a PyScal WaterOil object based on the input parameters supplied.

    Args:
        parameters: A dataframe consisting of all specified parameters.
        fast_pyscal: Run pyscal in fast-mode skipping checks. Useful for large models/ensembles.

    Returns:
        A PyScal WaterOil object

    """
    wo_relperm = WaterOil(
        swirr=parameters["swirr"],
        swl=parameters["swl"],
        swcr=parameters["swcr"],
        sorw=parameters["sorw"],
        h=H_CONSTANT,
        fast=fast_pyscal,
    )

    wo_relperm.add_corey_water(nw=parameters["nw"], krwend=parameters["krwend"])
    wo_relperm.add_corey_oil(now=parameters["now"], kroend=parameters["kroend"])

    return wo_relperm


def gen_og(parameters: pd.DataFrame, fast_pyscal: bool = False) -> GasOil:
    """
    Creates a PyScal GasOil object based on the input parameters supplied.

    Args:
        parameters: A dataframe consisting of all specified parameters.
        fast_pyscal: Run pyscal in fast-mode skipping checks. Useful for large models/ensembles.

    Returns:
        A PyScal GasOil object

    """
    og_relperm = GasOil(
        swirr=parameters["swirr"],
        swl=parameters["swl"],
        sorg=parameters["sorg"],
        sgcr=parameters["sgcr"],
        h=H_CONSTANT,
        fast=fast_pyscal,
    )

    og_relperm.add_corey_gas(ng=parameters["ng"], krgend=parameters["krgend"])
    og_relperm.add_corey_oil(nog=parameters["nog"], kroend=parameters["kroend"])

    return og_relperm


class RelativePermeability(Parameter):
    """
    Parameter type which takes care of stochastically drawn Relative Permeability parameters.

    Required parameters for SWOF generation: "swirr", "swl", "swcr", "sorw", "nw", "now", "krwend", "kroend"
    Required parameters for SGOF generation: "swirr", "swl", "sgcr", "sorg", "ng", "nog", "krgend", "kroend"

    Args:
        distribution_values:
            A dataframe with eight columns ("parameter", "minimum", "maximum", "mean", "base", "stddev",
            "distribution", "satnum") which state:
                * The name of the parameter,
                * The minimum value of the parameter (set to None if not applicable),
                * The maximum value of the parameter (set to None if not applicable),
                * The mean value of the parameter,
                * The mode of the parameter distribution (set to None if not applicable),
                * The standard deviation of the parameter,
                * The type of probability distribution,
                * To which SATNUM this applies.
        ti2ci: A dataframe with index equal to tube model index, and one column which equals cell indices.
        satnum: A dataframe defining the SATNUM for each flow tube.
        config: Information from the FlowNet config yaml
        interpolation_values:
            A dataframe with information about the relative permeability models used for interpolation.
            One row corresponds to one model, the column names should be the names of the parameters
            needed to establish the model. In addition there should be a column "CASE", which should be
            set to "low", "base" or "high", and a column "SATNUM" defining which SATNUM region the model applies to.

    """

    def __init__(
        self,
        distribution_values: pd.DataFrame,
        ti2ci: pd.DataFrame,
        satnum: pd.DataFrame,
        config: ConfigSuite.snapshot,
        interpolation_values: Optional[pd.DataFrame] = None,
    ):
        self._ti2ci: pd.DataFrame = ti2ci
        self._random_variables: List[ProbabilityDistribution] = [
            parameter_probability_distribution_class(row)
            for _, row in distribution_values.iterrows()
        ]

        self._unique_satnums: List[int] = list(distribution_values["satnum"].unique())
        self._parameters: List[Parameter] = list(
            distribution_values["parameter"].unique()
        )
        self._satnum: pd.DataFrame = satnum
        self._phases = config.flownet.phases
        self._interpolation_values: Optional[pd.DataFrame] = None
        self._scal_for_interp: Optional[PyscalList] = None
        if isinstance(interpolation_values, pd.DataFrame):
            self._interpolation_values = interpolation_values
            # self._scal_for_interp will be indexed by SATNUM region, starting at 1
            self._scal_for_interp = PyscalFactory.create_scal_recommendation_list(
                interpolation_values
            )

        self._swof, self._sgof = self._check_parameters()
        self._fast_pyscal: bool = config.flownet.fast_pyscal
        self._independent_interpolation = (
            config.model_parameters.relative_permeability.independent_interpolation
        )

    def _check_parameters(self) -> Tuple[bool, bool]:
        """
        Helper function to check the user-defined parameters and determination
        of what to generate: SWOF/SGOF.
        It will raise an error if something is wrong.

        Returns:
            A tuple of booleans defining whether to generate the SWOF and or SGOF tables.

        """
        if isinstance(self._interpolation_values, pd.DataFrame):
            check_parameters = list(self._interpolation_values.columns[:-2])
        else:
            check_parameters = self._parameters

        if len(check_parameters) != 8 and len(check_parameters) != 13:
            raise AssertionError(
                "Please specify the correct number of relative permeability "
                "parameters to model SWOF and/or SGOF. (8 or 13 parameters)\n"
                "Required parameters for SWOF generation: swirr, swl, swcr, sorw, nw, now, krwend, kroend\n"
                "Required parameters for SGOF generation: swirr, swl, sgcr, sorg, ng, nog, krgend, kroend\n"
                "See the documentation for more information."
            )

        # Check if SWOF should be generated
        swof = all(
            elem in check_parameters
            for elem in [
                "swirr",
                "swl",
                "swcr",
                "sorw",
                "nw",
                "now",
                "krwend",
                "kroend",
            ]
        )

        # Check if SGOF should be generated
        sgof = all(
            elem in check_parameters
            for elem in [
                "swirr",
                "swl",
                "sorg",
                "sgcr",
                "ng",
                "nog",
                "krgend",
                "kroend",
            ]
        )

        if not swof and not sgof:
            raise AssertionError(
                "One or more of the defined relative permeability parameters is not recognized.\n"
                "Please only specify allowed parameters. See the documentation for more information."
            )

        return swof, sgof

    def get_dims(self) -> Union[None, Dict[str, int]]:
        """
        Function to export the table dimensions used for memory allocation in Flow.

        Returns:
            Dictionary containing all dimensions to set.

        """
        dims_dict = {
            "NTSFUN": len(self._unique_satnums),
            "NSSFUN": 100,  # TODO: do not use hard-coded NSSFUN # pylint: disable=fixme
        }

        return dims_dict

    def render_output(self) -> Dict:
        """
        Creates SWOF/SGOF and SATNUM include content - which are given to the PROPS and GRID section.

        Returns:
            SWOF/SGOF and SATNUM include content

        """
        merged_df_satnum = self._ti2ci.merge(
            self._satnum, left_index=True, right_index=True
        )

        samples_per_satnum = len(self.random_samples) // len(self._unique_satnums)

        str_swofs = ""
        str_sgofs = ""
        parameters = []
        for i, _ in enumerate(self._unique_satnums):
            param_value_dict: Dict = dict(
                zip(
                    self._parameters,
                    self.random_samples[
                        i * samples_per_satnum : (i + 1) * samples_per_satnum
                    ],
                )
            )
            parameters.append(param_value_dict)

        partial_gen_wog = functools.partial(gen_wog, fast_pyscal=True)
        partial_gen_wo = functools.partial(gen_wo, fast_pyscal=True)
        partial_gen_og = functools.partial(gen_og, fast_pyscal=True)

        print(len(parameters))
        print(len(self._scal_for_interp))

        if isinstance(self._interpolation_values, pd.DataFrame):
            wog_list = self._scal_for_interp.interpolate(
                [param.get("interpolate") for param in parameters],
                [param.get("interpolate gas") for param in parameters]
                if self._independent_interpolation
                else None,
                h=0.05,
            )
            str_props_section = wog_list.SWOF()
            str_props_section += wog_list.SGOF()
        else:
            if self._swof and self._sgof:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for _, relperm in zip(  # type: ignore[assignment]
                        parameters, executor.map(partial_gen_wog, parameters)
                    ):
                        str_swofs += relperm.SWOF(header=False)
                        str_sgofs += relperm.SGOF(header=False)
            elif self._swof:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for _, relperm in zip(  # type: ignore[assignment]
                        parameters, executor.map(partial_gen_wo, parameters)
                    ):
                        str_swofs += relperm.SWOF(header=False)
            elif self._sgof:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for _, relperm in zip(  # type: ignore[assignment]
                        parameters, executor.map(partial_gen_og, parameters)
                    ):
                        str_sgofs += relperm.SGOF(header=False)
            else:
                raise ValueError(
                    "It seems like both SWOF and SGOF should not be generated."
                    "Either one of the two should be generated. Can't continue..."
                )

            str_props_section = f"SWOF\n{str_swofs}\n" if self._swof else ""
            str_props_section += f"SGOF\n{str_sgofs}\n" if self._sgof else ""

        str_runspec_section = "\n".join(self._phases).upper() + "\n"

        return {
            "RUNSPEC": str_runspec_section,
            "REGIONS": write_grdecl_file(merged_df_satnum, "SATNUM", int_type=True),
            "PROPS": str_props_section,
        }
