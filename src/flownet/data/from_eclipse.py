import warnings
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from ecl.well import WellInfo
from ecl.grid import EclGrid
from ecl.eclfile import EclFile
from ecl.summary import EclSum
from ecl2df import faults
from ecl2df.eclfiles import EclFiles

from .from_source import FromSource


class EclipseData(FromSource):
    """
    Eclipse data source class

    Args:
         eclipse_case: Full path to eclipse case to load data from
         perforation_handling_strategy: How to deal with perforations per well.
                                                 ('bottom_point', 'top_point', 'multiple')

    """

    def __init__(
        self,
        eclipse_case: Union[Path, str],
        perforation_handling_strategy: str = "bottom_point",
    ):
        super().__init__()

        self._eclipse_case: Path = Path(eclipse_case)
        self._eclsum = EclSum(str(self._eclipse_case))
        self._grid = EclGrid(str(self._eclipse_case.with_suffix(".EGRID")))
        self._restart = EclFile(str(self._eclipse_case.with_suffix(".UNRST")))
        self._wells = WellInfo(
            self._grid, rst_file=self._restart, load_segment_information=True
        )

        self._perforation_handling_strategy: str = perforation_handling_strategy

    def _coordinates(self) -> pd.DataFrame:
        """
        Function to extract well coordinates from an Flow/Eclipse simulation.

        Args:
            filename: Entire path to the simulated simulation case. This
                            case must have both and EGRID and UNRST file.
            perforation_handling_strategy: How to deal with perforations per well.
                                                ('bottom_point', 'top_point', 'multiple')

        Returns:
            columns: WELL_NAME, X, Y, Z

        """

        def multi_xyz_append(append_obj_list):
            for global_conn in append_obj_list[1]:
                coords.append(
                    [append_obj_list[0], *self._grid.get_xyz(ijk=global_conn.ijk())]
                )

        coords: List = []

        for well_name in self._wells.allWellNames():
            global_conns = self._wells[well_name][0].globalConnections()
            coord_append = coords.append
            if self._perforation_handling_strategy == "bottom_point":
                xyz = self._grid.get_xyz(ijk=global_conns[-1].ijk())
            elif self._perforation_handling_strategy == "top_point":
                xyz = self._grid.get_xyz(ijk=global_conns[0].ijk())
            elif self._perforation_handling_strategy == "multiple":
                xyz = [global_conns]
                coord_append = multi_xyz_append
            else:
                raise Exception(
                    f"perforation strategy {self._perforation_handling_strategy} unknown"
                )

            coord_append([well_name, *xyz])

        return pd.DataFrame(coords, columns=["WELL_NAME", "X", "Y", "Z"])

    def _production_data(self) -> pd.DataFrame:
        """
        Function to read production data for all producers and injectors from an
        Flow/Eclipse simulation. The simulation is required to write out the
        following vectors to the summary file: WOPR, WGPR, WWPR, WBHP, WTHP, WGIR, WWIR

        Returns:
            A DataFrame with a DateTimeIndex and the following columns:
                - date          equal to index
                - WELL_NAME     Well name as used in Eclipse
                - WOPR          Well Oil Production Rate
                - WGPR          Well Gas Production Rate
                - WWPR          Well Water Production Rate
                - WBHP          Well Bottom Hole Pressure
                - WTHP          Well Tubing Head Pressure
                - WGIR          Well Gas Injection Rate
                - WWIR          Well Water Injection Rate
                - WSTAT         Well status (OPEN, SHUT, STOP)
                - TYPE          Well Type: "OP", "GP", "WI", "GI"
                - PHASE         Main producing/injecting phase fluid: "OIL", "GAS", "WATER"

        Todo:
            * Remove depreciation warning suppression when solved in LibEcl.
            * Improve robustness pf setting of Phase and Type.

        """
        keys = ["WOPR", "WGPR", "WWPR", "WBHP", "WTHP", "WGIR", "WWIR", "WSTAT"]

        df_production_data = pd.DataFrame()

        start_date = self._get_start_date()

        # Suppress a depreciation warning inside LibEcl
        warnings.simplefilter("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():

            for well_name in self._eclsum.wells():
                df = pd.DataFrame()

                df["date"] = self._eclsum.report_dates
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

                for prod_key in keys:
                    try:
                        df[f"{prod_key}"] = self._eclsum.get_values(
                            f"{prod_key}:{well_name}", report_only=True
                        )
                    except KeyError:
                        df[f"{prod_key}"] = np.nan

                # Find number of leading empty rows (with only nan or 0 values)
                zero = df.fillna(0).eq(0).all(1).sum()

                if zero < df.shape[0]:
                    # If there are no empty rows, prepend one for the start date
                    if zero == 0:
                        df1 = df.head(1)
                        as_list = df1.index.tolist()
                        idx = as_list.index(df1.index)
                        as_list[idx] = pd.to_datetime(start_date)
                        df1.index = as_list
                        df = pd.concat([df1, df])
                        for col in df.columns:
                            df[col].values[0] = 0
                        zero = 1

                    # Keep only the last empty row (well activation date)
                    df = df.iloc[max(zero - 1, 0) :]

                    # Assign well targets to the correct schedule dates
                    df = df.shift(-1)
                    # Make sure the row for the final date is not empty
                    df.iloc[-1] = df.iloc[-2]

                # Set columns that have only exact zero values to np.nan
                df.loc[:, (df == 0).all(axis=0)] = np.nan

                df["WELL_NAME"] = well_name
                df_production_data = df_production_data.append(df)

        df_production_data["PHASE"] = None
        df_production_data.loc[df_production_data["WOPR"] > 0, "PHASE"] = "OIL"
        df_production_data.loc[df_production_data["WWIR"] > 0, "PHASE"] = "WATER"
        df_production_data.loc[df_production_data["WGIR"] > 0, "PHASE"] = "GAS"

        df_production_data["WSTAT"] = df_production_data["WSTAT"].map(
            {
                1: "OPEN",  # Producer OPEN
                2: "OPEN",  # Injector OPEN
                3: "SHUT",
                4: "STOP",
                5: "SHUT",  # PSHUT
                6: "STOP",  # PSTOP
                np.nan: "STOP",
            }
        )

        df_production_data["TYPE"] = None
        df_production_data.loc[df_production_data["WOPR"] > 0, "TYPE"] = "OP"
        df_production_data.loc[df_production_data["WWIR"] > 0, "TYPE"] = "WI"
        df_production_data.loc[df_production_data["WGIR"] > 0, "TYPE"] = "GI"

        df_production_data[["PHASE", "TYPE"]] = df_production_data[
            ["PHASE", "TYPE"]
        ].fillna(method="backfill")

        df_production_data["date"] = df_production_data.index
        df_production_data["date"] = pd.to_datetime(df_production_data["date"]).dt.date

        return df_production_data

    def _faults(self) -> pd.DataFrame:
        """
        Function to read fault plane data using ecl2df.

        Returns:
            A dataframe with columns NAME, X, Y, Z with data for fault planes

        """
        eclfile = EclFiles(self._eclipse_case)
        df_fault_keyword = faults.df(eclfile)

        points = []
        for _, row in df_fault_keyword.iterrows():

            i = row["I"] - 1
            j = row["J"] - 1
            k = row["K"] - 1

            points.append((row["NAME"], i, j, k))

            if row["FACE"] == "X" or row["FACE"] == "X+":
                points.append((row["NAME"], i + 1, j, k))
            elif row["FACE"] == "Y" or row["FACE"] == "Y+":
                points.append((row["NAME"], i, j + 1, k))
            elif row["FACE"] == "Z" or row["FACE"] == "Z+":
                points.append((row["NAME"], i, j, k + 1))
            elif row["FACE"] == "X-":
                points.append((row["NAME"], i - 1, j, k))
            elif row["FACE"] == "Y-":
                points.append((row["NAME"], i, j - 1, k))
            elif row["FACE"] == "Z-":
                points.append((row["NAME"], i, j, k - 1))
            else:
                raise ValueError(
                    f"Could not interpret '{row['FACE']}' while reading the FAULTS keyword."
                )

        df_faults = pd.DataFrame.from_records(points, columns=["NAME", "I", "J", "K"])

        if not df_faults.empty:
            df_faults[["X", "Y", "Z"]] = pd.DataFrame(
                df_faults.apply(
                    lambda row: list(
                        self._grid.get_xyz(ijk=(row["I"], row["J"], row["K"]))
                    ),
                    axis=1,
                ).values.tolist()
            )

        return df_faults.drop(["I", "J", "K"], axis=1)

    def _get_start_date(self):
        return self._eclsum.start_date

    @property
    def faults(self) -> pd.DataFrame:
        """dataframe with all fault data"""
        return self._faults()

    @property
    def production(self) -> pd.DataFrame:
        """dataframe with all production data"""
        return self._production_data()

    @property
    def coordinates(self) -> pd.DataFrame:
        """dataframe with all coordinates"""
        return self._coordinates()
