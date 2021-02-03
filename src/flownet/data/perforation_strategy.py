from typing import List

import numpy as np
import pandas as pd


def bottom_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function returns the bottom point of the well (assuming it is the last open connection specified, anywhere
    in time).

    Args:
        df: connection DataFrame

    Returns:
        bottom point connections

    """
    df_multiple = multiple(df)
    df_multiple_ever_true = (
        df_multiple.groupby(["X", "Y", "Z", "WELL_NAME", "LAYER_ID"])
        .sum()
        .reset_index()
    )
    df_first_dates = (
        df[["WELL_NAME", "DATE"]]
        .sort_values(["WELL_NAME", "DATE"])
        .groupby(["WELL_NAME"])
        .first()
    )
    result = (
        df_multiple_ever_true.groupby("WELL_NAME")
        .last()
        .reset_index()
        .assign(OPEN=True)
        .merge(df_first_dates, on="WELL_NAME")
    )
    return result


def top_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function returns the top point of the well (assuming it is the first open connection specified, anywhere
    in time).

    Args:
        df: connection DataFrame

    Returns:
        bottom point connections

    """
    df_multiple = multiple(df)
    df_multiple_ever_true = (
        df_multiple.groupby(["X", "Y", "Z", "WELL_NAME", "LAYER_ID"])
        .sum()
        .reset_index()
    )
    df_first_dates = (
        df[["WELL_NAME", "DATE"]]
        .sort_values(["WELL_NAME", "DATE"])
        .groupby(["WELL_NAME"])
        .first()
    )
    result = (
        df_multiple_ever_true.groupby("WELL_NAME")
        .first()
        .reset_index()
        .assign(OPEN=True)
        .merge(df_first_dates, on="WELL_NAME")
    )
    return result


def multiple(df: pd.DataFrame) -> pd.DataFrame:
    """
    This strategy creates multiple connections per well, as many as there is data available. Connections that
    repeatedly have the same state through time are reduced to only having records for state changes.

    NB. This may lead to a lot of connections in the FlowNet with potentially numerical issues as a result. When
        generating a FlowNet that is not aware of geological layering, it is questionable whether having many
        connections per well will lead to useful results.

    Args:
        df: Dataframe with all well connections, through time, including state.

    Returns:
        DataFrame with all connections

    """
    df = df[["WELL_NAME", "X", "Y", "Z", "DATE", "OPEN", "LAYER_ID"]].sort_values(
        ["WELL_NAME", "X", "Y", "Z", "DATE", "LAYER_ID"]
    )
    df["SHIFT"] = df.groupby(["WELL_NAME", "X", "Y", "Z", "LAYER_ID"])["OPEN"].shift(1)

    return df[(df["OPEN"] != df["SHIFT"])].drop("SHIFT", axis=1)


# pylint: disable=too-many-locals
def multiple_based_on_workovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    This strategy creates multiple connections per well when the well during the historic production period has been
    straddled or plugged (i.e., individual connections have been shut).

    The following steps are performed per layer:

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

    Args:
        df: Dataframe with all well connections, through time, including state.

    Returns:
        Dataframe with 1 or more connections per well depending on the historic straddles / plugs.

    """
    df = multiple(df)

    df_w_groups = pd.DataFrame(
        [], columns=["WELL_NAME", "X", "Y", "Z", "DATE", "OPEN", "GROUPID", "LAYER_ID"]
    )
    df_groups = pd.DataFrame([], columns=["X", "Y", "Z", "GROUPID"])
    groupid = 0

    # Step 1
    for layer in df["LAYER_ID"].unique():
        for well_name in df[df["LAYER_ID"] == layer]["WELL_NAME"].unique():
            df_well = df.loc[
                (df["WELL_NAME"] == well_name) & (df["LAYER_ID"] == layer)
            ][["X", "Y", "Z", "WELL_NAME", "LAYER_ID", "OPEN", "DATE"]]
            df_well_piv = df_well.pivot_table(
                "OPEN", ["X", "Y", "Z", "WELL_NAME", "LAYER_ID"], "DATE"
            )
            df_well_piv.fillna(method="ffill", axis=1, inplace=True)
            df_well_piv.fillna(False, inplace=True)
            df_well_piv = df_well_piv.apply(lambda x: hash(tuple(x)), axis=1)

            for group in df_well_piv.unique():
                df_group = (
                    df_well_piv.loc[df_well_piv == group]
                    .index.to_frame()
                    .reset_index(drop=True)[["X", "Y", "Z"]]
                )
                df_group["GROUPID"] = groupid
                groupid += 1
                df_groups = df_groups.append(df_group)

            df_w_groups = df_w_groups.append(
                df_well.merge(df_groups, how="left", on=["X", "Y", "Z"])
            )

    # Step 2
    for groupid in df_w_groups["GROUPID"].unique():
        df_group = df_w_groups.loc[df_w_groups["GROUPID"] == groupid]

        xmin, ymin, zmin = df_group[["X", "Y", "Z"]].min()
        xmax, ymax, zmax = df_group[["X", "Y", "Z"]].max()

        df_foreign = df_w_groups.loc[
            (
                ((df_w_groups["X"] >= xmin) & (df_w_groups["X"] <= xmax))
                & ((df_w_groups["Y"] >= ymin) & (df_w_groups["Y"] <= ymax))
                & ((df_w_groups["Z"] >= zmin) & (df_w_groups["Z"] <= zmax))
            )
            & (df_w_groups["GROUPID"] != groupid)
            & (
                df_w_groups["WELL_NAME"]
                == df_w_groups[df_w_groups["GROUPID"] == groupid]["WELL_NAME"].values[0]
            )
        ]

        # Step 3
        if df_foreign.shape[0]:
            xmin_foreign, ymin_foreign, zmin_foreign = df_foreign[["X", "Y", "Z"]].min()
            xmax_foreign, ymax_foreign, zmax_foreign = df_foreign[["X", "Y", "Z"]].max()

            df_w_groups.loc[
                (df_w_groups["GROUPID"] == groupid)
                & (
                    (df_w_groups["X"] < xmin_foreign)
                    | (df_w_groups["Y"] < ymin_foreign)
                    | (df_w_groups["Z"] < zmin_foreign)
                ),
                ["GROUPID"],
            ] = (
                df_w_groups["GROUPID"].max() + 1
            )

            df_w_groups.loc[
                (df_w_groups["GROUPID"] == groupid)
                & (
                    (df_w_groups["X"] > xmax_foreign)
                    | (df_w_groups["Y"] > ymax_foreign)
                    | (df_w_groups["Z"] > zmax_foreign)
                ),
                ["GROUPID"],
            ] = (
                df_w_groups["GROUPID"].max() + 1
            )

    df_w_groups["OPEN"] = df_w_groups["OPEN"].astype(int)
    result = (
        df_w_groups.groupby(["WELL_NAME", "DATE", "GROUPID", "LAYER_ID"])
        .mean()
        .reset_index()
    )
    result["OPEN"] = result["OPEN"].astype(bool)
    result.drop("GROUPID", axis=1, inplace=True)

    return result


def time_avg_open_location(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df:

    Returns:
    """

    coords: List = []
    for well_name in df["WELL_NAME"].unique():
        coord_append = coords.append
        connection_open_time = {}
        for index, row in (
            df[df["WELL_NAME"] == well_name]
            .sort_values("DATE", ascending=True)
            .iterrows()
        ):
            time = row["DATE"]
            if index == 0:
                prev_time = time
            xyz = (row["X"], row["Y"], row["Z"])

            if xyz not in connection_open_time:
                connection_open_time[xyz] = 0.0
            elif row["OPEN"]:
                connection_open_time[xyz] += (time - prev_time).total_seconds()
            else:
                connection_open_time[xyz] += 0.0

            prev_time = time

        xyz_values = np.zeros((1, 3), dtype=np.float64)
        total_open_time = sum(connection_open_time.values())

        if total_open_time > 0:
            for connection, open_time in connection_open_time.items():
                xyz_values += np.multiply(
                    np.array(connection),
                    open_time / total_open_time,
                )
        else:
            for connection, open_time in connection_open_time.items():
                xyz_values += np.divide(
                    np.array(connection),
                    len(connection_open_time.items()),
                )

        xyz_coordinates = tuple(*xyz_values)

        coord_append([well_name, *xyz_coordinates])

    result = pd.DataFrame(coords, columns=["WELL_NAME", "X", "Y", "Z"])
    result["OPEN"] = True
    result["DATE"] = None

    df_first_dates = (
        df[["WELL_NAME", "DATE"]]
        .sort_values(["WELL_NAME", "DATE"])
        .groupby(["WELL_NAME"])
        .first()
    )

    return result[["WELL_NAME", "X", "Y", "Z", "OPEN"]].merge(
        df_first_dates, on="WELL_NAME"
    )
