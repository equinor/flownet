import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
from ecl.grid import EclGrid
from ecl.eclfile import EclFile, EclInitFile
from ecl.summary import EclSum
from ecl2df import compdat, faults
from ecl2df.eclfiles import EclFiles

from ..data import perforation_strategy
from .from_source import FromSource


class FlowData(FromSource):
    """
    Flow data source class

    Args:
         input_case: Full path to eclipse case to load data from
         perforation_handling_strategy: How to deal with perforations per well.
                                                 ('bottom_point', 'top_point', 'multiple')

    """

    def __init__(
        self,
        input_case: Union[Path, str],
        perforation_handling_strategy: str = "bottom_point",
    ):
        super().__init__()

        self._input_case: Path = Path(input_case)
        self._eclsum = EclSum(str(self._input_case))
        self._init = EclFile(str(self._input_case.with_suffix(".INIT")))
        self._grid = EclGrid(str(self._input_case.with_suffix(".EGRID")))
        self._restart = EclFile(str(self._input_case.with_suffix(".UNRST")))
        self._init = EclInitFile(self._grid, str(self._input_case.with_suffix(".INIT")))
        self._wells = compdat.df(EclFiles(str(self._input_case)))
        self._layers = [(1, 2), (3, 4)]

        self._perforation_handling_strategy: str = perforation_handling_strategy

    # pylint: disable=too-many-branches
    def _well_connections(self) -> pd.DataFrame:
        """
        Function to extract well connection coordinates from an Flow simulation including their
        opening and closure time. The output of this function will be filtered based on the
        configured perforation strategy.

        Returns:
            columns: WELL_NAME, X, Y, Z, DATE, OPEN

        """
        new_items = []
        for _, row in self._wells.iterrows():
            X, Y, Z = self._grid.get_xyz(
                ijk=(row["I"] - 1, row["J"] - 1, row["K1"] - 1)
            )
            new_row = {
                "WELL_NAME": row["WELL"],
                "IJK": (
                    row["I"] - 1,
                    row["J"] - 1,
                    row["K1"] - 1,
                ),
                "X": X,
                "Y": Y,
                "Z": Z,
                "DATE": row["DATE"],
                "OPEN": bool(row["OP/SH"] == "OPEN"),
            }
            new_items.append(new_row)

        df = pd.DataFrame(
            new_items, columns=["WELL_NAME", "IJK", "X", "Y", "Z", "DATE", "OPEN"]
        )
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d").dt.date

        try:
            perforation_strategy_method = getattr(
                perforation_strategy, self._perforation_handling_strategy
            )
        except AttributeError as attribute_error:
            raise NotImplementedError(
                f"The perforation handling strategy {self._perforation_handling_strategy} is unknown."
            ) from attribute_error

        return perforation_strategy_method(df).sort_values(["DATE"])

    def _well_logs(self) -> pd.DataFrame:
        """
        Function to extract well log information from a Flow simulation.

        Returns:
            columns: WELL_NAME, X, Y, Z, PERM (mD), PORO (-)

        """
        coords: List = []

        for well_name in self._wells["WELL"].unique():
            unique_connections = self._wells[
                self._wells["WELL"] == well_name
            ].drop_duplicates(subset=["I", "J", "K1", "K2"])
            for _, connection in unique_connections.iterrows():
                ijk = (connection["I"] - 1, connection["J"] - 1, connection["K1"] - 1)
                xyz = self._grid.get_xyz(ijk=ijk)

                perm_kw = self._init.iget_named_kw("PERMX", 0)
                poro_kw = self._init.iget_named_kw("PORO", 0)

                coords.append(
                    [
                        well_name,
                        *xyz,
                        perm_kw[
                            self._grid.cell(i=ijk[0], j=ijk[1], k=ijk[2]).active_index
                        ],
                        poro_kw[
                            self._grid.cell(i=ijk[0], j=ijk[1], k=ijk[2]).active_index
                        ],
                    ]
                )

        return pd.DataFrame(
            coords, columns=["WELL_NAME", "X", "Y", "Z", "PERM", "PORO"]
        )

    def _production_data(self) -> pd.DataFrame:
        """
        Function to read production data for all producers and injectors from an
        Flow simulation. The simulation is required to write out the
        following vectors to the summary file: WOPR, WGPR, WWPR, WBHP, WTHP, WGIR, WWIR

        Returns:
            A DataFrame with a DateTimeIndex and the following columns:
                - date          equal to index
                - WELL_NAME     Well name as used in Flow
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

                # Set columns that have only exact zero values to np.nan
                df.loc[:, (df == 0).all(axis=0)] = np.nan

                df["WELL_NAME"] = well_name

                df["PHASE"] = None
                df.loc[df["WOPR"] > 0, "PHASE"] = "OIL"
                df.loc[df["WWIR"] > 0, "PHASE"] = "WATER"
                df.loc[df["WGIR"] > 0, "PHASE"] = "GAS"
                df["TYPE"] = None
                df.loc[df["WOPR"] > 0, "TYPE"] = "OP"
                df.loc[df["WWIR"] > 0, "TYPE"] = "WI"
                df.loc[df["WGIR"] > 0, "TYPE"] = "GI"
                # make sure the correct well type is set also when the well is shut in
                df[["PHASE", "TYPE"]] = df[["PHASE", "TYPE"]].fillna(method="backfill")
                df[["PHASE", "TYPE"]] = df[["PHASE", "TYPE"]].fillna(method="ffill")

                df_production_data = df_production_data.append(df)

        if df_production_data["WSTAT"].isna().all():
            warnings.warn(
                "No WSTAT:* summary vectors in input case - setting default well status to OPEN."
            )
            wstat_default = "OPEN"
        else:
            wstat_default = "STOP"

        df_production_data["WSTAT"] = df_production_data["WSTAT"].map(
            {
                0: wstat_default,
                1: "OPEN",  # Producer OPEN
                2: "OPEN",  # Injector OPEN
                3: "SHUT",
                4: "STOP",
                5: "SHUT",  # PSHUT
                6: "STOP",  # PSTOP
                np.nan: wstat_default,
            }
        )

        # ensure that a type is assigned also if a well is never activated
        df_production_data[["PHASE", "TYPE"]] = df_production_data[
            ["PHASE", "TYPE"]
        ].fillna(method="backfill")
        df_production_data[["PHASE", "TYPE"]] = df_production_data[
            ["PHASE", "TYPE"]
        ].fillna(method="ffill")

        df_production_data["date"] = df_production_data.index
        df_production_data["date"] = pd.to_datetime(df_production_data["date"]).dt.date

        return df_production_data

    def _faults(self) -> pd.DataFrame:
        """
        Function to read fault plane data using ecl2df.

        Returns:
            A dataframe with columns NAME, X, Y, Z with data for fault planes

        """
        eclfile = EclFiles(self._input_case)
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

    def _grid_cell_bounding_boxes(self, layer_id: Optional[int] = None) -> np.ndarray:
        """
        Function to get the bounding box (x, y and z min + max) for all grid cells

        Args:
            layer_id: The FlowNet layer id to be used to create the bounding box.

        Returns:
            A (active grid cells x 6) numpy array with columns [ xmin, xmax, ymin, ymax, zmin, zmax ]
            filtered on layer_id if not None.
        """
        if layer_id:
            # TODO: Make sure the k-range is correct (zero offset?!)
            (k_min, k_max) = self._layers[layer_id]
        else:
            (k_min, k_max) = (0, self._grid.nz)

        cells = [
            cell for cell in self._grid.cells(active=True) if (k_min <= cell.k <= k_max)
        ]
        xyz = np.empty((8 * len(cells), 3))

        for n_cell, cell in enumerate(cells):
            if k_min <= cell.k <= k_max:
                corners = cell.corner
                for n_corner, corner in enumerate(corners):
                    xyz[n_cell * 8 + n_corner, :] = corner
                    n_cell += 1

        xmin = xyz[:, 0].reshape(-1, 8).min(axis=1)
        xmax = xyz[:, 0].reshape(-1, 8).max(axis=1)
        ymin = xyz[:, 1].reshape(-1, 8).min(axis=1)
        ymax = xyz[:, 1].reshape(-1, 8).max(axis=1)
        zmin = xyz[:, 2].reshape(-1, 8).min(axis=1)
        zmax = xyz[:, 2].reshape(-1, 8).max(axis=1)

        return np.vstack([xmin, xmax, ymin, ymax, zmin, zmax]).T

    def _get_start_date(self):
        return self._eclsum.start_date

    def init(self, name: str) -> np.ndarray:
        """array with 'name' regions"""
        return self._init[name][0]

    def get_unique_regions(self, name: str) -> np.ndarray:
        """array with unique 'name' regions"""
        return np.unique(self._init[name][0])

    @property
    def grid_cell_bounding_boxes(self) -> np.ndarray:
        """Boundingboxes for all gridcells"""
        return self._grid_cell_bounding_boxes()

    @property
    def faults(self) -> pd.DataFrame:
        """dataframe with all fault data"""
        return self._faults()

    @property
    def production(self) -> pd.DataFrame:
        """dataframe with all production data"""
        return self._production_data()

    @property
    def well_connections(self) -> pd.DataFrame:
        """dataframe with all well connection coordinates"""
        return self._well_connections()

    @property
    def well_logs(self) -> pd.DataFrame:
        """dataframe with all well log"""
        return self._well_logs()

    @property
    def grid(self) -> EclGrid:
        """the simulation grid with properties"""
        return self._grid

    @property.getter
    def layers(self) -> List[Tuple(int, int)]:
        """Get the list of top and bottom k-indeces of a the orignal model that represents a FlowNet layer"""
        return self._layers