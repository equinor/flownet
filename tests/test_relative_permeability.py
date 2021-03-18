from pyscal import WaterOil, GasOil
from flownet.parameters._relative_permeability import (
    swof_from_parameters,
    sgof_from_parameters,
)
from flownet.utils.constants import H_CONSTANT


def test_SWOF() -> None:
    parameter_dict = {}
    parameter_dict["swirr"] = 0.01
    parameter_dict["swl"] = 0.05
    parameter_dict["swcr"] = 0.2
    parameter_dict["sorw"] = 0.25
    parameter_dict["krwend"] = 0.5
    parameter_dict["kroend"] = 0.95
    parameter_dict["nw"] = 2.25
    parameter_dict["now"] = 2.25

    wo = WaterOil(
        swirr=parameter_dict["swirr"],
        swl=parameter_dict["swl"],
        swcr=parameter_dict["swcr"],
        sorw=parameter_dict["sorw"],
        h=H_CONSTANT,
    )

    wo.add_corey_oil(now=parameter_dict["now"], kroend=parameter_dict["kroend"])
    wo.add_corey_water(nw=parameter_dict["nw"], krwend=parameter_dict["krwend"])

    pyscal_swof_string = wo.SWOF(header=False, dataincommentrow=False).splitlines()[
        3:-1
    ]
    numpy_swof_string = swof_from_parameters(parameter_dict).splitlines()

    for i, line in enumerate(pyscal_swof_string):
        assert [round(float(elem), 4) for elem in line.split()] == [
            round(float(elem), 4) for elem in numpy_swof_string[i].split()
        ]


def test_SGOF() -> None:
    parameter_dict = {}
    parameter_dict["swirr"] = 0.01
    parameter_dict["swl"] = 0.05
    parameter_dict["sgcr"] = 0.055
    parameter_dict["sorg"] = 0.15
    parameter_dict["krgend"] = 0.95
    parameter_dict["kroend"] = 0.95
    parameter_dict["ng"] = 2.25
    parameter_dict["nog"] = 2.25

    go = GasOil(
        swirr=parameter_dict["swirr"],
        swl=parameter_dict["swl"],
        sgcr=parameter_dict["sgcr"],
        sorg=parameter_dict["sorg"],
        h=H_CONSTANT,
    )

    go.add_corey_oil(nog=parameter_dict["nog"], kroend=parameter_dict["kroend"])
    go.add_corey_gas(ng=parameter_dict["ng"], krgend=parameter_dict["krgend"])

    pyscal_sgof_string = go.SGOF(header=False, dataincommentrow=False).splitlines()[
        3:-1
    ]
    numpy_sgof_string = sgof_from_parameters(parameter_dict).splitlines()

    for i, line in enumerate(pyscal_sgof_string):
        assert [round(float(elem), 4) for elem in line.split()] == [
            round(float(elem), 4) for elem in numpy_sgof_string[i].split()
        ]
