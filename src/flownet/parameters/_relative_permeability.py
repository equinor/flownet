from typing import Dict, List, Tuple, Union, Optional
import warnings

import pandas as pd
import numpy as np

from configsuite import ConfigSuite
from ..utils import write_grdecl_file
from ..utils.constants import H_CONSTANT

from .probability_distributions import ProbabilityDistribution
from ._base_parameter import Parameter, parameter_probability_distribution_class


def swof_from_parameters(parameters: Dict) -> str:
    """
    Creates a SWOF table based on a dictionary of input parameters/values

    Args:
        parameters: Dictionary of saturation and relative permeability endpoints

    Returns:
        A string with the resulting SWOF table
    """
    swl = parameters["swl"]
    swcr = parameters["swcr"]
    sorw = parameters["sorw"]
    kroend = parameters["kroend"]
    krwend = parameters["krwend"]
    krwmax = parameters["krwmax"]
    now = parameters["now"]
    nw = parameters["nw"]

    # array of water saturations to calculate krow and krw for
    sw = np.sort(np.append(np.arange(swl, 1, H_CONSTANT), [swcr, 1 - sorw, 1]))
    # remove potential duplicate values
    sw = sw[~(np.triu(np.abs(sw[:, None] - sw) <= 1e-5, 1)).any(0)]
    # normalized saturations
    swn = ((sw - swcr) / (1 - swcr - sorw)).clip(min=0)
    son = ((1 - sw - sorw) / (1 - sorw - swl)).clip(min=0)
    # calculate relative permeabilities
    krow = kroend * son ** now
    krw = krwend * swn ** nw
    # interpolate between krwend and krwmax
    krw_interp = (krwmax - krwend) / sorw * (sw - 1) + krwmax
    krw[krw > krwend] = krw_interp[krw > krwend]
    # only zero capillary pressure implemented
    pc = np.zeros(len(sw))

    swof = np.transpose(np.stack((sw, krw, krow, pc)))
    swof_string = np.array2string(swof, formatter={"float_kind": lambda x: f"{x:.7f}"})
    swof_string = swof_string.replace(" [", "").replace("[", "").replace("]", "")

    return swof_string


def sgof_from_parameters(parameters: Dict) -> str:
    """
    Creates a SGOF table based on a dictionary of input parameters/values

    Args:
        parameters: Dictionary of saturation and relative permeability endpoints

    Returns:
        A string with the resulting SGOF table
    """
    swl = parameters["swl"]
    sorg = parameters["sorg"]
    sgcr = parameters["sgcr"]
    krgend = parameters["krgend"]
    kroend = parameters["kroend"]
    ng = parameters["ng"]
    nog = parameters["nog"]

    # array of gas saturations to calculate krog and krg for
    sg = np.sort(
        np.append(
            np.arange(sgcr, 1 - swl - sorg, H_CONSTANT), [1 - sorg - swl, 1 - swl, 0]
        )
    )
    # remove potential duplicate values
    sg = sg[~(np.triu(np.abs(sg[:, None] - sg) <= 1e-5, 1)).any(0)]
    # normalized saturations
    son = (((1 - sg) - sorg - swl) / (1 - sorg - swl)).clip(min=0)
    sgn = ((sg - sgcr) / (1 - swl - sgcr - sorg)).clip(min=0)
    # calculate relative permeabilities
    krog = kroend * son ** nog
    krg = krgend * sgn ** ng
    # interpolate between krgend and krgmax (=1)
    krg_interp = (1 - krgend) / sorg * (sg - 1) + 1 + swl / sorg * (1 - krgend)
    krg[krg > krgend] = krg_interp[krg > krgend]
    # only zero capillary pressure implemented
    pc = np.zeros(len(sg))

    sgof = np.transpose(np.stack((sg, krg, krog, pc)))
    sgof_string = np.array2string(sgof, formatter={"float_kind": lambda x: f"{x:.7f}"})
    sgof_string = sgof_string.replace(" [", "").replace("[", "").replace("]", "")

    return sgof_string


def interpolate_wo(parameter: float, scalrec: Dict) -> Dict:
    """
    Creates interpolated saturation endpoints and relative
    permeability endpoints for water/oil based on an interpolation parameter
    and three separate input cases (low/base/high).

    Args:
        parameter: A value on the interval -1 to 1 used for interpolation
        scalrec: A dictionary containing the relative permeability
            and saturation endpoints for the low/base/high cases

    Returns:
        A dictionary with the interpolated water/oil saturation and relative
        permeability endpoints
    """
    # interpolate swirr, swl, swcr, sorw, nw, now, krwend, kroend
    if parameter < 0:
        (i, j) = (1, 0)
    else:
        (i, j) = (1, 2)
    parameter = abs(parameter)

    parameter_dict = {}
    for elem in ["swirr", "swl", "swcr", "sorw", "nw", "now", "krwend", "kroend"]:
        parameter_dict[elem] = (
            scalrec[elem][i] * (1 - parameter) + scalrec[elem][j] * parameter
        )

    return parameter_dict


def interpolate_go(parameter: float, scalrec: Dict) -> Dict:
    """
    Creates interpolated saturation endpoints and relative
    permeability endpoints for gas/oil based on an interpolation parameter
    and three separate input cases (low/base/high).

    Args:
        parameter: A value on the interval -1 to 1 used for interpolation
        scalrec: A dataframe containing the relative permeability
            and saturation endpoints for the low/base/high cases

    Returns:
        A dictionary with the interpolated gas/oil saturation and relative
        permeability endpoints
    """
    # interpolate swirr, swl, sgcr, sorg, ng, nog, krgend, kroend
    if parameter < 0:
        (i, j) = (1, 0)
    else:
        (i, j) = (1, 2)
    parameter = abs(parameter)

    parameter_dict = {}
    for elem in ["swirr", "swl", "sgcr", "sorg", "ng", "nog", "krgend", "kroend"]:
        parameter_dict[elem] = (
            scalrec[elem][i] * (1 - parameter) + scalrec[elem][j] * parameter
        )

    return parameter_dict


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
        self._krwmax_add_to_krwend = (
            config.model_parameters.relative_permeability.krwmax_add_to_krwend
        )
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
        self._fast_pyscal: bool = config.flownet.fast_pyscal
        self._interpolation_values: Optional[pd.DataFrame] = None

        if isinstance(interpolation_values, pd.DataFrame):
            self._interpolation_values = interpolation_values

        self._swof, self._sgof = self._check_parameters()
        self._independent_interpolation = (
            config.model_parameters.relative_permeability.independent_interpolation
        )
        self._swcr_add_to_swl = (
            config.model_parameters.relative_permeability.swcr_add_to_swl
        )

    def _check_krwmax(self, params: Dict) -> Dict:
        """
        Helper function to check if krwmax is defined in config file or
        if it should be defaulted to 1.

        Args:
            params: Dictionary with parameter names and
                parameter values for one realization in ERT
        Returns:
            Updated dictionary with parameters
        """
        if "krwmax" not in params.keys():
            params["krwmax"] = 1
        if self._krwmax_add_to_krwend:
            params["krwmax"] = min(1, params["krwend"] + params["krwmax"])
        if params["krwmax"] < params["krwend"]:
            params["krwmax"] = params["krwend"]
            warnings.warn(
                "KRWMAX < KRWEND in one or more realizations. KRWMAX set equal to KRWEND."
                "Consider revising the prior distribution input for KRWMAX and/or KRWEND, "
                "or use the krwmax_add_to_krwend option."
            )
        return params

    def _check_parameters(self) -> Tuple[bool, bool]:
        """
        Helper function to check the user-defined parameters and determination
        of what to generate: SWOF/SGOF.
        It will raise an error if something is wrong.

        Returns:
            A tuple of booleans defining whether to generate the SWOF and or SGOF tables.

        """
        if isinstance(self._interpolation_values, pd.DataFrame):
            check_parameters = list(self._interpolation_values.columns[:-2]).copy()
        else:
            check_parameters = self._parameters.copy()

        # krwmax will be defaulted to one if not defined, so removed from required parameters
        if "krwmax" in check_parameters:
            check_parameters.remove("krwmax")
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

        if self._interpolation_values is not None:
            for satnum in self._unique_satnums:
                # all endpoints and corey exponents are interpolated
                # linearly between two of the input case - between
                # 'low' and  'base' if parameter < 0
                # and between 'base' and 'high' if parameter >= 0.
                # pyscal interpolates endpoints and then interpolates
                # between the non-linear parts of the relative permeability
                # curves wrt normalised saturation (not interpolating corey
                # exponents - using scipy interp1d). Resulting SWOFs and
                # SGOFs are seemingly pretty similar, but not equal.
                parameter_dict = (
                    self._interpolation_values.loc[
                        self._interpolation_values["SATNUM"] == satnum
                    ]
                    .reset_index(drop=True)
                    .to_dict()
                )
                interp_wo = parameters[satnum - 1].get("interpolate")
                interp_go = (
                    parameters[satnum - 1].get("interpolate gas")
                    if self._independent_interpolation
                    else interp_wo
                )

                if isinstance(interp_wo, float) and isinstance(interp_go, float):
                    param_wo = interpolate_wo(interp_wo, parameter_dict)
                    param_go = interpolate_go(interp_go, parameter_dict)
                else:
                    raise ValueError(
                        "Interpolation parameter is not a float."
                        "Something wrong with parameters generated by ERT"
                    )
                param_wo = self._check_krwmax(param_wo)
                str_swofs += swof_from_parameters(param_wo) + "\n/\n"
                str_sgofs += sgof_from_parameters(param_go) + "\n/\n"
        else:
            for param in parameters:
                if self._swof:
                    if self._swcr_add_to_swl:
                        param["swcr"] = param["swl"] + param["swcr"]
                    param = self._check_krwmax(param)
                    str_swofs += swof_from_parameters(param) + "\n/\n"
                if self._sgof:
                    str_sgofs += sgof_from_parameters(param) + "\n/\n"
        str_props_section = f"SWOF\n{str_swofs}\n" if self._swof else ""
        str_props_section += f"SGOF\n{str_sgofs}\n" if self._sgof else ""
        str_runspec_section = "\n".join(self._phases).upper() + "\n"

        return {
            "RUNSPEC": str_runspec_section,
            "REGIONS": write_grdecl_file(merged_df_satnum, "SATNUM", int_type=True),
            "PROPS": str_props_section,
        }
