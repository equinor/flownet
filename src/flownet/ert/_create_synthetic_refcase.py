import pathlib

from ecl.summary import EclSum
from ecl.util.util import CTime
from matplotlib.dates import date2num

from ..realization import Schedule


def create_synthetic_refcase(
    case_name: pathlib.Path, schedule: Schedule, nx: int = 1, ny: int = 1, nz: int = 1
):
    """
    This function creates a synthetic simulation output in order to please ERT, which
    uses it for mapping the dates to report step.

    Args:
        case_name: Case name for synthetic case
        schedule: FlowNet Schedule instance
        nx: Number of grid blocks in the x-direction
        ny: Number of grid blocks in the y-direction
        nz: Number of grid blocks in the z-direction

    Returns:
        Nothing

    """
    datetimes = schedule.get_dates()
    numdates = [date2num(date) for date in datetimes]

    start_time = CTime._timegm(  # pylint: disable=protected-access
        0, 0, 0, datetimes[0].day, datetimes[0].month, datetimes[0].year
    )

    eclsum = EclSum.writer(str(case_name), start_time, nx, ny, nz)

    vectors = []
    for well in schedule.get_wells():
        vectors.append(["WOPR", well, 0, "Sm3/day"])
        vectors.append(["WWPR", well, 0, "Sm3/day"])
        vectors.append(["WGOR", well, 0, "Sm3/day"])
        vectors.append(["WBHP", well, 0, "Sm3/day"])

    for vector in vectors:
        # pylint: disable=no-member
        EclSum.addVariable(eclsum, vector[0], vector[1], vector[2], vector[3])

    for report_step, _ in enumerate(numdates):
        # pylint: disable=no-member
        if report_step == 0:
            tstep = EclSum.addTStep(eclsum, 1, numdates[report_step] - numdates[0])
        else:
            tstep = EclSum.addTStep(
                eclsum, report_step, numdates[report_step] - numdates[0]
            )

        for vector in vectors:
            tstep[f"{vector[0]}:{vector[1]}"] = 0

    EclSum.fwrite(eclsum)
