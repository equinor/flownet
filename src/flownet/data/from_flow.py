import warnings
import operator
from pathlib import Path
from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from ecl.grid import EclGrid
from ecl.eclfile import EclFile, EclInitFile
from ecl.summary import EclSum
from ecl2df import compdat, faults
from ecl2df.eclfiles import EclFiles

from ..data import perforation_strategy
from .from_source import FromSource
from ..network_model import NetworkModel


class FlowData(FromSource):
    """
    Flow data source class

    Args:
         input_case: Full path to eclipse case to load data from
         layers: List with definition of isolated layers, if present.

    """

    def __init__(
        self,
        input_case: Union[Path, str],
        layers: Tuple = (),
    ):
        super().__init__()

        self._input_case: Path = Path(input_case)
        self._eclsum = EclSum(str(self._input_case))
        self._init = EclFile(str(self._input_case.with_suffix(".INIT")))
        self._grid = EclGrid(str(self._input_case.with_suffix(".EGRID")))
        self._restart = EclFile(str(self._input_case.with_suffix(".UNRST")))
        self._init = EclInitFile(self._grid, str(self._input_case.with_suffix(".INIT")))
        self._wells = compdat.df(EclFiles(str(self._input_case)))
        self._layers = layers

    # pylint: disable=too-many-branches
    def _well_connections(self, perforation_handling_strategy: str) -> pd.DataFrame:
        """
        Function to extract well connection coordinates from a Flow simulation including their
        opening and closure time. The output of this function will be filtered based on the
        configured perforation strategy.

        Args:
            perforation_handling_strategy: Strategy to be used when creating perforations.
            Valid options are bottom_point, top_point, multiple, time_avg_open_location and
            multiple_based_on_workovers.

        Returns:
            columns: WELL_NAME, X, Y, Z, DATE, OPEN, LAYER_ID

        """
        if len(self._layers) > 0 and self._grid.nz is not self._layers[-1][-1]:
            raise ValueError(
                f"Number of layers from config ({self._layers[-1][-1]}) is not equal to "
                f"number of layers from flow simulation ({self._grid.nz})."
            )

        new_items = []
        for _, row in self._wells.iterrows():
            X, Y, Z = self._grid.get_xyz(
                ijk=(row["I"] - 1, row["J"] - 1, row["K1"] - 1)
            )
            if len(self._layers) > 0:
                for count, (i, j) in enumerate(self._layers):
                    if row["K1"] in range(i, j + 1):
                        layer_id = count
                        break
            else:
                layer_id = 0

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
                "LAYER_ID": layer_id,
            }
            new_items.append(new_row)

        df = pd.DataFrame(
            new_items,
            columns=["WELL_NAME", "IJK", "X", "Y", "Z", "DATE", "OPEN", "LAYER_ID"],
        )
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d").dt.date

        try:
            perforation_strategy_method = getattr(
                perforation_strategy, perforation_handling_strategy
            )
        except AttributeError as attribute_error:
            raise NotImplementedError(
                f"The perforation handling strategy {perforation_handling_strategy} is unknown."
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
                - WOPT          Well Cumulative Oil Production
                - WGPT          Well Cumulative Gas Production Rate
                - WWPT          Well Cumulative Water Production Rate
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
        keys = [
            "WOPR",
            "WGPR",
            "WWPR",
            "WOPT",
            "WGPT",
            "WWPT",
            "WBHP",
            "WTHP",
            "WGIR",
            "WWIR",
            "WGIT",
            "WWIT",
            "WSTAT",
        ]

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

    def grid_cell_bounding_boxes(self, layer_id: int) -> np.ndarray:
        """
        Function to get the bounding box (x, y and z min + max) for all grid cells

        Args:
            layer_id: The FlowNet layer id to be used to create the bounding box.

        Returns:
            A (active grid cells x 6) numpy array with columns [ xmin, xmax, ymin, ymax, zmin, zmax ]
            filtered on layer_id if not None.
        """
        if self._layers:
            (k_min, k_max) = tuple(map(operator.sub, self._layers[layer_id], (1, 1)))
        else:
            (k_min, k_max) = (0, self._grid.nz)

        cells = [
            cell for cell in self._grid.cells(active=True) if (k_min <= cell.k <= k_max)
        ]
        xyz = np.empty((8 * len(cells), 3))

        for n_cell, cell in enumerate(cells):
            for n_corner, corner in enumerate(cell.corners):
                xyz[n_cell * 8 + n_corner, :] = corner

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

    def get_well_connections(self, perforation_handling_strategy: str) -> pd.DataFrame:
        """
        Function to get dataframe with all well connection coordinates,
        filtered based on the perforation_handling_strategy.

        Args:
            perforation_handling_strategy: Strategy to be used when creating perforations.
            Valid options are bottom_point, top_point, multiple,
            time_avg_open_location and multiple_based_on_workovers.

        Returns:
            Dataframe with all well connection coordinates,
            filtered based on the perforation_handling_strategy.
            Columns: WELL_NAME, X, Y, Z, DATE, OPEN, LAYER_ID
        """

        return self._well_connections(
            perforation_handling_strategy=perforation_handling_strategy
        )

    @staticmethod
    def _bulk_volume_per_flownet_cell_based_on_tube_length(
        network: NetworkModel,
    ) -> np.ndarray:
        """Generate bulk volume per flownet grid cell based on the total length
        of each tube in the FlowNet and the convex hull on the FlowNet network.

        Args:
            network: FlowNet network instance.

        Returns:
            A list with multipliers for the total bulk volume for each flownet tube cell.
        """
        return (
            network.total_bulkvolume
            * network.grid["cell_length"].values
            / network.grid["cell_length"].sum()
        )

    def _bulk_volume_per_flownet_cell_based_on_voronoi_of_input_model(
        self, network: NetworkModel
    ) -> np.ndarray:
        """Generate bulk volume distribution per grid cell in the FlowNet model based on the geometrical
        distribution of the volume in the original (full field) simulation model. I.e., the original model's
        volume will be distributed over the FlowNet's tubes by assigning original model grid cell
        volumes to the nearest FlowNet tube midpoint.

        Args:
            network: FlowNet network instance.

        Returns:
            A list with multipliers for the total bulk volume for each flownet tube cell.
        """
        flownet_tube_midpoints = np.array(network.get_connection_midpoints())
        model_cell_mid_points = np.array(
            [cell.coordinate for cell in self._grid.cells()]
        )
        model_cell_active_state = [cell.active for cell in self._grid.cells()]
        model_cell_volume = [cell.volume for cell in self._grid.cells()]

        # Determine nearest flow tube for each cell in the original model
        tree = KDTree(flownet_tube_midpoints)
        _, matched_indices = tree.query(model_cell_mid_points, k=[1])

        # Distribute the original model's volume per cell, that lies within the FlowNet model's
        # convex hull, to flownet tubes that are nearest.
        tube_volumes = np.zeros(len(flownet_tube_midpoints))

        for cell_i, tube_id in enumerate(matched_indices):
            if model_cell_active_state[cell_i]:  # and model_cell_in_hull[cell_i]
                tube_volumes[tube_id[0]] += model_cell_volume[cell_i]

        # Distribute tube volume over individual cells
        properties_per_cell = pd.DataFrame(
            pd.DataFrame(data=network.grid.index, index=network.grid.model).index
        )
        cells_per_tube = properties_per_cell.reset_index().groupby(["model"]).size()
        cell_volumes = np.array(
            [
                tube_volumes[i] / cells_per_tube[i]
                for i in properties_per_cell["model"].values
            ]
        )

        return cell_volumes  # / sum(cell_volumes)

    def get_bulk_volume_per_flownet_cell(
        self,
        method: str = "tube_length",
        network: Optional[NetworkModel] = None,
    ) -> np.ndarray:
        """Function that generates initial bulk volume estimates for the FlowNet model. The
        following methods are available:
            * **tube_length**: distrubutes the volume of the convex hull of the FlowNet model,
                based on the length of a tube. I.e., if all tubes have equeal lenght, they
                will have equal volume.
            * **voronoi_per_tube**: distributes the input models bulk volume of active cells
                to the nearest FlowNet tube of a cell. The total volume of the tube is then
                devided equally over the cells of the tube. I.e., in areas with a higher
                FlowNet tube density, the volume per cell is lower. Mind that if the FlowNet
                model, i.e., the convex hull of the well connections, is much smaller than the
                original model volume outside of the well connection convex hull might be
                collapsed at the borders of the model. I.e., the borders of your model could
                het unrealisticly large volumes. This can be mitigated by increasing the hull
                factor of the FlowNet model generation process.

        Args:
            initial_volume_distribution_method: Method to be used to make an initial distribution of the
                in-place volume of the FlowNet.
            network: FlowNet network instance.

        Returns:
            A list with multipliers for the total bulk volume for each flownet tube cell, that
            distributes the bulk volume according to the spatial distribution of the volume
            in the original model.

        Raise:
            NotImplementedError in case a users asks for this type of distribution but is
            not build a FlowNet up a FlowNet from a simulation file.

        """
        if not isinstance(network, NetworkModel):
            raise TypeError(
                "To be able to distribute volume you need "
                "to supply the target FlowNet network model that you want to distribute towards."
            )

        if method == "voronoi_per_tube":
            distribution_multipliers = (
                self._bulk_volume_per_flownet_cell_based_on_voronoi_of_input_model(
                    network
                )
            )
        elif method == "tube_length":
            distribution_multipliers = (
                self._bulk_volume_per_flownet_cell_based_on_tube_length(network)
            )
        else:
            raise ValueError(
                f"'{method}' is not a valid prior volume " "distribution method."
            )

        return distribution_multipliers

    @property
    def faults(self) -> pd.DataFrame:
        """dataframe with all fault data"""
        return self._faults()

    @property
    def production(self) -> pd.DataFrame:
        """dataframe with all production data"""
        return self._production_data()

    @property
    def well_logs(self) -> pd.DataFrame:
        """dataframe with all well log"""
        return self._well_logs()

    @property
    def grid(self) -> EclGrid:
        """the simulation grid with properties"""
        return self._grid

    @property
    def layers(self) -> Union[Tuple[Tuple[int, int]], Tuple]:
        """Get the list of top and bottom k-indeces of a the orignal model that represents a FlowNet layer"""
        return self._layers
