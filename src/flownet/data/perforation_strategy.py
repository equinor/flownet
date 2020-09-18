import numpy as np
import pandas as pd


def bottom_point(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df:

    Returns:

    """
    raise NotImplementedError()


def top_point(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df:

    Returns:

    """
    raise NotImplementedError()


def multiple(df: pd.DataFrame) -> pd.DataFrame:
    """
    This strategy creates multiple connections per well, as many as there is data available.

    NB. This may lead to a lot of connections in the FlowNet with potentially numerical issues as a result. When
        generating a FlowNet that is not aware of geological layering, it is questionable whether having many
        connections per well will lead to useful results.

    Args:
        df: Dataframe with all well connections, through time, including state.

    Returns:
        DataFrame will all connections

    Todo: This should be time and state aware!

    """
    return df.drop_duplicates("IJK", keep="first")[["WELL_NAME", "X", "Y", "Z"]]


def time_avg_open_location_multiple_based_on_workovers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """


    Args:
        df:

    Returns:

    """
    return time_avg_open_location(multiple_based_on_workovers(df))


def multiple_based_on_workovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    This strategy creates multiple connections per well when the well during the historic production period has been
    straddled or plugged (i.e., individual connections have been shut).

    The following steps are performed:

        1. Split connections into groups of connections per well, based on their open/closing history. That is,
           connections that have seen opening or closure at the same moment in time are considered a group. This is
           done by generating a hash value based on opening state booleans through time.
        2. For each group a bounding box will be created and it will be verified that no foreign connections (i.e.,
           connections from other groups) are inside of the bounding box.
        3. If connections of other groups are found inside of the bounding box a line will be fitted through the
           connections of the group being checked and a perpendicular splitting plane will be created at the center of
           foreign connections. Two new groups now exist that both will be checked via step 2.
        4. When all groups have no foreign connections in their bounding boxes the average location of the groups
           are returned, including their respective open/closing times.

    Todo: opening and closing times need to be passed out of the function as well. But this requires that the receiving
          end also understands what it needs to do with times... Aka: open and close connections.

    Args:
        df: Dataframe with all well connections, through time, including state.

    Returns:
        Dataframe with 1 or more connections per will depending on the historic straddles / plugs.

    """

    df_groups = pd.DataFrame([], columns=["X", "Y", "Z", "GROUPID"])
    groupid = 0

    # Step 1
    for well_name in df["WELL_NAME"].unique():
        df_well = df.loc[df["WELL_NAME"] == well_name]
        df_well = df_well.pivot_table("OPEN", ["X", "Y", "Z", "WELL_NAME"], "DATE")
        df_well = df_well.apply(lambda x: hash(tuple(x)), axis=1)

        for group in df_well.unique():
            df_group = (
                df_well.loc[df_well == group].index.to_frame().reset_index(drop=True)
            )
            df_group["GROUPID"] = groupid
            groupid += 1
            df_groups = df_groups.append(df_group)

    # Step 2
    for groupid in df_groups["GROUPID"].unique():
        df_group = df_groups.loc[df_groups["GROUPID"] == groupid]

        xmin, ymin, zmin = df_group[["X", "Y", "Z"]].min()
        xmax, ymax, zmax = df_group[["X", "Y", "Z"]].max()

        df_forein = df_groups.loc[
            (df_groups["X"] >= xmin)
            & (df_groups["X"] <= xmax)
            & (df_groups["Y"] >= ymin)
            & (df_groups["Y"] <= ymax)
            & (df_groups["Z"] >= zmin)
            & (df_groups["Z"] <= zmax)
            & (df_groups["GROUPID"] != groupid)
        ]

        # Step 3
        if df_forein.shape[0]:
            # Todo: write splitting function
            pass

    return (
        df_groups.groupby(["GROUPID", "WELL_NAME"])
        .mean()
        .reset_index()[["WELL_NAME", "X", "Y", "Z"]]
    )


def time_avg_open_location(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df:

    Returns:

    """

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

        elif self._perforation_handling_strategy == "multiple_based_on_workovers":

            pass

        elif self._perforation_handling_strategy == "time_avg_open_location":
            connection_open_time = {}

            for i, conn_status in enumerate(self._wells[well_name]):
                time = datetime.datetime.strptime(
                    str(conn_status.simulationTime()), "%Y-%m-%d %H:%M:%S"
                )
                if i == 0:
                    prev_time = time

                for connection in conn_status.globalConnections():
                    if connection.ijk() not in connection_open_time:
                        connection_open_time[connection.ijk()] = 0.0
                    elif connection.isOpen():
                        connection_open_time[connection.ijk()] += (
                            time - prev_time
                        ).total_seconds()
                    else:
                        connection_open_time[connection.ijk()] += 0.0

                prev_time = time

            xyz_values = np.zeros((1, 3), dtype=np.float64)
            total_open_time = sum(connection_open_time.values())

            if total_open_time > 0:
                for connection, open_time in connection_open_time.items():
                    xyz_values += np.multiply(
                        np.array(self._grid.get_xyz(ijk=connection)),
                        open_time / total_open_time,
                    )
            else:
                for connection, open_time in connection_open_time.items():
                    xyz_values += np.divide(
                        np.array(self._grid.get_xyz(ijk=connection)),
                        len(connection_open_time.items()),
                    )

            xyz = tuple(*xyz_values)

        else:
            raise Exception(
                f"perforation strategy {self._perforation_handling_strategy} unknown"
            )

        coord_append([well_name, *xyz])

    return pd.DataFrame(
        coords, columns=["WELL_NAME", "X", "Y", "Z", "START_DATE", "END_DATE"]
    )
    """

    return df
