from pyscal import WaterOil, GasOil

from flownet.parameters._relative_permeability import (
    swof_from_parameters,
    sgof_from_parameters,
    interpolate_wo,
    interpolate_go,
)
from flownet.utils.constants import H_CONSTANT


def test_swof_generation() -> None:
    """
    Testing if the FlowNet code and pyscal generate
    the same SWOF table - test tolerance set to 4 decimals
    """
    parameter_dict = {}
    parameter_dict["swirr"] = 0.01
    parameter_dict["swl"] = 0.05
    parameter_dict["swcr"] = 0.2
    parameter_dict["sorw"] = 0.25
    parameter_dict["krwend"] = 0.5
    parameter_dict["krwmax"] = 0.6
    parameter_dict["kroend"] = 0.95
    parameter_dict["nw"] = 2.25
    parameter_dict["now"] = 2.25

    wateroil = WaterOil(
        swirr=parameter_dict["swirr"],
        swl=parameter_dict["swl"],
        swcr=parameter_dict["swcr"],
        sorw=parameter_dict["sorw"],
        h=H_CONSTANT,
    )

    wateroil.add_corey_oil(now=parameter_dict["now"], kroend=parameter_dict["kroend"])
    wateroil.add_corey_water(
        nw=parameter_dict["nw"],
        krwend=parameter_dict["krwend"],
        krwmax=parameter_dict["krwmax"],
    )

    pyscal_swof_string = wateroil.SWOF(
        header=False, dataincommentrow=False
    ).splitlines()[3:-1]
    numpy_swof_string = swof_from_parameters(parameter_dict).splitlines()

    for i, line in enumerate(pyscal_swof_string):
        assert [round(float(elem), 4) for elem in line.split()] == [
            round(float(elem), 4) for elem in numpy_swof_string[i].split()
        ]


def test_sgof_generation() -> None:
    """
    Testing if the FlowNet code and pyscal generate
    the same SGOF table - test tolerance set to 4 decimals
    """
    parameter_dict = {}
    parameter_dict["swirr"] = 0.01
    parameter_dict["swl"] = 0.05
    parameter_dict["sgcr"] = 0.055
    parameter_dict["sorg"] = 0.15
    parameter_dict["krgend"] = 0.95
    parameter_dict["kroend"] = 0.95
    parameter_dict["ng"] = 2.25
    parameter_dict["nog"] = 2.25

    gasoil = GasOil(
        swirr=parameter_dict["swirr"],
        swl=parameter_dict["swl"],
        sgcr=parameter_dict["sgcr"],
        sorg=parameter_dict["sorg"],
        h=H_CONSTANT,
    )

    gasoil.add_corey_oil(nog=parameter_dict["nog"], kroend=parameter_dict["kroend"])
    gasoil.add_corey_gas(ng=parameter_dict["ng"], krgend=parameter_dict["krgend"])

    pyscal_sgof_string = gasoil.SGOF(header=False, dataincommentrow=False).splitlines()[
        3:-1
    ]
    numpy_sgof_string = sgof_from_parameters(parameter_dict).splitlines()

    for i, line in enumerate(pyscal_sgof_string):
        assert [round(float(elem), 4) for elem in line.split()] == [
            round(float(elem), 4) for elem in numpy_sgof_string[i].split()
        ]


def test_scalrec_extremes_wo() -> None:
    # assert that interpolation reproduces the low/base/high values
    parameter_dict = {
        "swirr": {0: 0.01, 1: 0.01, 2: 0.01},
        "swl": {0: 0.05, 1: 0.05, 2: 0.05},
        "swcr": {0: 0.1, 1: 0.2, 2: 0.3},
        "sorw": {0: 0.2, 1: 0.25, 2: 0.3},
        "krwend": {0: 0.4, 1: 0.5, 2: 0.6},
        "kroend": {0: 0.9, 1: 0.95, 2: 1},
        "nw": {0: 1.5, 1: 2.25, 2: 3},
        "now": {0: 1.5, 1: 2.25, 2: 3},
    }
    interpolated_parameter_dict_low = interpolate_wo(-1, parameter_dict)
    interpolated_parameter_dict_base = interpolate_wo(0, parameter_dict)
    interpolated_parameter_dict_high = interpolate_wo(1, parameter_dict)
    for key in parameter_dict.items():
        assert parameter_dict[key][0] == interpolated_parameter_dict_low[key]
        assert parameter_dict[key][1] == interpolated_parameter_dict_base[key]
        assert parameter_dict[key][2] == interpolated_parameter_dict_high[key]


def test_scalrec_extremes_go() -> None:
    # assert that interpolation reproduces the low/base/high values
    parameter_dict = {
        "swirr": {0: 0.01, 1: 0.01, 2: 0.01},
        "swl": {0: 0.05, 1: 0.05, 2: 0.05},
        "sgcr": {0: 0.1, 1: 0.15, 2: 0.2},
        "sorg": {0: 0.3, 1: 0.55, 2: 0.8},
        "krgend": {0: 0.9, 1: 0.95, 2: 1},
        "kroend": {0: 0.9, 1: 0.95, 2: 1},
        "ng": {0: 1.5, 1: 2.25, 2: 3},
        "nog": {0: 1.5, 1: 2.25, 2: 3},
    }
    interpolated_parameter_dict_low = interpolate_go(-1, parameter_dict)
    interpolated_parameter_dict_base = interpolate_go(0, parameter_dict)
    interpolated_parameter_dict_high = interpolate_go(1, parameter_dict)
    for key in parameter_dict.items():
        assert parameter_dict[key][0] == interpolated_parameter_dict_low[key]
        assert parameter_dict[key][1] == interpolated_parameter_dict_base[key]
        assert parameter_dict[key][2] == interpolated_parameter_dict_high[key]
