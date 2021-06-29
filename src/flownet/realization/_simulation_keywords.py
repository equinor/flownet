from abc import ABC
import datetime

import numpy as np


class Keyword(ABC):
    """
    Each simulation Keyword is a subclass of the Keyword-class - which enforces that a keyword is always linked
    to a particular date.

    """

    name: str
    well_name: str
    oil_rate: float
    gas_rate: float
    bhp: float
    thp: float

    def __init__(self, date):
        self._date = date

    @property
    def date(self) -> datetime.date:
        return self._date


class DATES(Keyword):
    """
    This keyword advances the simulation to a given report date after which additional keywords may be entered
    to instruct OPM Flow to perform additional functions via the SCHEDULE section keywords, or further
    DATES data sets or keywords may be entered to advance the simulator to the next report date.

    See the OPM Flow manual for further details.

    """


class COMPDAT(Keyword):
    """
    The COMPDAT keyword defines how a well is connected to the reservoir by defining or modifying existing
    well connections. Ideally the connections should be declared in the correct sequence, starting with the
    connection nearest the well head and then working along the wellbore towards the bottom or toe of the
    well.

    See the OPM Flow manual for further details.

    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments

    def __init__(
        self,
        date: datetime.date,
        well_name: str,
        i: int,
        j: int,
        k1: int,
        k2: int,
        rw: float,
        status: str = "1*",
        satnum: str = "1*",
        confact: str = "1*",
        kh: str = "1*",
        skin: str = "1*",
        dfact: str = "1*",
        direct: str = "1*",
    ):
        super().__init__(date)
        self.name = "COMPDAT"
        self.well_name: str = well_name
        self.i: int = i
        self.j: int = j
        self.k1: int = k1
        self.k2: int = k2
        self.status: str = status
        self.satnum: str = satnum
        self.confact: str = confact
        self.rw: float = rw
        self.kh: str = kh
        self.skin: str = skin
        self.dfact: str = dfact
        self.direct: str = direct


class WCONHIST(Keyword):
    """
    The WCONHIST keyword defines production rates and pressures for wells that have been declared history
    matching wells by the use of this keyword. History matching wells are handled differently than ordinary wells
    that use the WCONPROD keyword for controlling their production targets and constraints. However, the
    wells still need to be defined like ordinary production wells using the WELSPECS keyword in the SCHEDULE
    section.

    See the OPM Flow manual for further details.

    """

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        date: datetime.date,
        well_name: str,
        prod_control_mode: str,
        status: str = "1*",
        oil_rate: float = np.nan,
        water_rate: float = np.nan,
        gas_rate: float = np.nan,
        salt_rate: float = np.nan,
        salt_total: float = np.nan,
        oil_total: float = np.nan,
        water_total: float = np.nan,
        gas_total: float = np.nan,
        vfp_table: str = "1*",
        artificial_lift: str = "1*",
        thp: float = np.nan,
        bhp: float = np.nan,
    ):
        super().__init__(date)
        self.name = "WCONHIST"
        self.well_name: str = well_name
        self.status: str = status
        self.prod_control_mode: str = prod_control_mode
        self.oil_rate: float = oil_rate
        self.water_rate: float = water_rate
        self.gas_rate: float = gas_rate
        self.salt_rate: float = salt_rate
        self.salt_total: float = salt_total
        self.oil_total: float = oil_total
        self.water_total: float = water_total
        self.gas_total: float = gas_total
        self.vfp_table: str = vfp_table
        self.artificial_lift: str = artificial_lift
        self.thp: float = thp
        self.bhp: float = bhp


class WCONINJH(Keyword):
    """
    The WCONINJH keyword defines injection rates and pressures for wells that have been declared history
    matching wells by the use of this keyword. History matching wells are handled differently then ordinary wells
    that use the WCONINJE keyword for controlling their injection targets and constraints. However, the wells
    still need to be defined like ordinary injection wells using the WELSPECS keyword in the SCHEDULE section.

    See the OPM Flow manual for further details.

    """

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        date: datetime.date,
        well_name: str,
        inj_type: str,
        status: str = "1*",
        rate: float = np.nan,
        total: float = np.nan,
        bhp: float = np.nan,
        thp: float = np.nan,
        vfp_table: str = "1*",
        inj_control_mode: str = "1*",
    ):
        super().__init__(date)
        self.name = "WCONINJH"
        self.well_name: str = well_name
        self.inj_type: str = inj_type
        self.status: str = status
        self.rate: float = rate
        self.total: float = total
        self.bhp: float = bhp
        self.thp: float = thp
        self.vfp_table: str = vfp_table
        self.inj_control_mode: str = inj_control_mode


class WELSPECS(Keyword):
    """
    The WELSPECS keyword defines the general well specification data for all well types, and must be used for all
    wells before any other well specification keywords are used in the input file. The keyword declares the name
    of well, the wellhead location and other key parameters.

    See the OPM Flow manual for further details.

    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments

    def __init__(
        self,
        date: datetime.date,
        well_name: str,
        group_name: str,
        i: int,
        j: int,
        phase: str,
        ref_depth_bhp: str = "1*",
        drainage_radius: str = "1*",
        inflow_equation: str = "1*",
        shutin_instruction: str = "1*",
        crossflow: str = "1*",
        pvt_table: str = "1*",
        density_calc: str = "1*",
        fip: str = "1*",
    ):
        super().__init__(date)
        self.name = "WELSPECS"
        self.well_name: str = well_name
        self.group_name: str = group_name
        self.i: int = i
        self.j: int = j
        self.ref_depth_bhp: str = ref_depth_bhp
        self.phase: str = phase
        self.drainge_radius: str = drainage_radius
        self.inflow_equation: str = inflow_equation
        self.shutin_instruction: str = shutin_instruction
        self.crossflow: str = crossflow
        self.pvt_table: str = pvt_table
        self.density_calc: str = density_calc
        self.fip: str = fip


class WSALT(Keyword):
    """
    The WSALT keyword defines the salt concentration of the injected water.

    See the OPM Flow manual for further details.

    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments

    def __init__(
        self,
        date: datetime.date,
        well_name: str,
        salt_concentration: float = np.nan,
    ):
        super().__init__(date)
        self.name = "WSALT"
        self.well_name: str = well_name
        self.salt_concentration: float = salt_concentration
