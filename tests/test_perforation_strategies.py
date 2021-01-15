from datetime import timedelta

import pandas as pd

from flownet.data.perforation_strategy import (
    bottom_point,
    top_point,
    multiple,
    multiple_based_on_workovers,
)

DF = pd.read_csv("./tests/data/well_perforations_2layers.csv")
D2 = pd.Timestamp("2021-01-13 09:53:52.832254").to_pydatetime()
D1 = D2 - timedelta(days=1)
D0 = D1 - timedelta(days=1)


def test_bottom_point() -> None:

    result = bottom_point(DF)

    assert result.shape[0] == len(DF["WELL_NAME"].unique())
    assert all(result["OPEN"].values)

    assert result.loc[result["WELL_NAME"] == "A"]["X"].values[0] == 2
    assert result.loc[result["WELL_NAME"] == "B"]["X"].values[0] == 4
    assert result.loc[result["WELL_NAME"] == "C"]["X"].values[0] == 6
    assert result.loc[result["WELL_NAME"] == "D"]["X"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["X"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["X"].values[0] == 11
    assert result.loc[result["WELL_NAME"] == "H"]["X"].values[0] == 13
    assert result.loc[result["WELL_NAME"] == "I"]["X"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "J"]["X"].values[0] == 17
    assert result.loc[result["WELL_NAME"] == "A"]["Y"].values[0] == 2
    assert result.loc[result["WELL_NAME"] == "B"]["Y"].values[0] == 4
    assert result.loc[result["WELL_NAME"] == "C"]["Y"].values[0] == 6
    assert result.loc[result["WELL_NAME"] == "D"]["Y"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["Y"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["Y"].values[0] == 11
    assert result.loc[result["WELL_NAME"] == "H"]["Y"].values[0] == 13
    assert result.loc[result["WELL_NAME"] == "I"]["Y"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "J"]["Y"].values[0] == 17
    assert result.loc[result["WELL_NAME"] == "A"]["Z"].values[0] == 2
    assert result.loc[result["WELL_NAME"] == "B"]["Z"].values[0] == 4
    assert result.loc[result["WELL_NAME"] == "C"]["Z"].values[0] == 6
    assert result.loc[result["WELL_NAME"] == "D"]["Z"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["Z"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["Z"].values[0] == 11
    assert result.loc[result["WELL_NAME"] == "H"]["Z"].values[0] == 13
    assert result.loc[result["WELL_NAME"] == "I"]["Z"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "J"]["Z"].values[0] == 17

    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "A"]["DATE"].values[0]
        ).to_pydatetime()
        == D1
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "B"]["DATE"].values[0]
        ).to_pydatetime()
        == D1
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "C"]["DATE"].values[0]
        ).to_pydatetime()
        == D2
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "D"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "E"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "G"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "H"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "I"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "J"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )

    assert all(result["OPEN"].values)
    assert all(result.WELL_NAME.isin(DF.WELL_NAME))


def test_top_point() -> None:
    result = top_point(DF)

    assert result.shape[0] == len(DF["WELL_NAME"].unique())
    assert all(result["OPEN"].values)

    assert result.loc[result["WELL_NAME"] == "A"]["X"].values[0] == 1
    assert result.loc[result["WELL_NAME"] == "B"]["X"].values[0] == 3
    assert result.loc[result["WELL_NAME"] == "C"]["X"].values[0] == 5
    assert result.loc[result["WELL_NAME"] == "D"]["X"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["X"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["X"].values[0] == 10
    assert result.loc[result["WELL_NAME"] == "H"]["X"].values[0] == 12
    assert result.loc[result["WELL_NAME"] == "I"]["X"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "J"]["X"].values[0] == 15
    assert result.loc[result["WELL_NAME"] == "A"]["Y"].values[0] == 1
    assert result.loc[result["WELL_NAME"] == "B"]["Y"].values[0] == 3
    assert result.loc[result["WELL_NAME"] == "C"]["Y"].values[0] == 5
    assert result.loc[result["WELL_NAME"] == "D"]["Y"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["Y"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["Y"].values[0] == 10
    assert result.loc[result["WELL_NAME"] == "H"]["Y"].values[0] == 12
    assert result.loc[result["WELL_NAME"] == "I"]["Y"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "J"]["Y"].values[0] == 15
    assert result.loc[result["WELL_NAME"] == "A"]["Z"].values[0] == 1
    assert result.loc[result["WELL_NAME"] == "B"]["Z"].values[0] == 3
    assert result.loc[result["WELL_NAME"] == "C"]["Z"].values[0] == 5
    assert result.loc[result["WELL_NAME"] == "D"]["Z"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["Z"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["Z"].values[0] == 10
    assert result.loc[result["WELL_NAME"] == "H"]["Z"].values[0] == 12
    assert result.loc[result["WELL_NAME"] == "I"]["Z"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "J"]["Z"].values[0] == 15

    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "A"]["DATE"].values[0]
        ).to_pydatetime()
        == D1
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "B"]["DATE"].values[0]
        ).to_pydatetime()
        == D1
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "C"]["DATE"].values[0]
        ).to_pydatetime()
        == D2
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "D"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "E"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "G"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "H"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "I"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "J"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )

    assert all(result["OPEN"].values)
    assert all(result.WELL_NAME.isin(DF.WELL_NAME))


def test_multiple() -> None:
    result = multiple(DF)

    assert not all(result["OPEN"].values)
    assert all(result.X.isin(DF.X).astype(float))
    assert all(result.Y.isin(DF.Y).astype(float))
    assert all(result.Z.isin(DF.Z).astype(float))
    assert all(result.WELL_NAME.isin(DF.WELL_NAME))
    assert all(DF.WELL_NAME.isin(result.WELL_NAME))

    assert len(result[result["WELL_NAME"] == "K"]) == 4
    assert (
        len(
            result[
                (
                    result["OPEN"]
                    == result.groupby(["WELL_NAME", "X", "Y", "Z", "LAYER_ID"])[
                        "OPEN"
                    ].shift(1)
                )
            ]
        )
        == 0
    )

    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "A"]["DATE"].values[0]
        ).to_pydatetime()
        == D1
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "B"]["DATE"].values[0]
        ).to_pydatetime()
        == D1
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "C"]["DATE"].values[0]
        ).to_pydatetime()
        == D2
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "D"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "E"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "G"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "H"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "I"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "J"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )


def test_multiple_based_on_workovers() -> None:
    result = multiple_based_on_workovers(DF)

    assert multiple(DF).shape[1] is result.shape[1]
    assert multiple(DF).shape[0] > result.shape[0]
    assert not all(result["OPEN"].values)
    assert not all(result.X.isin(DF.X).astype(float))
    assert not all(result.Y.isin(DF.Y).astype(float))
    assert not all(result.Z.isin(DF.Z).astype(float))
    assert all(result.WELL_NAME.isin(DF.WELL_NAME))
    assert all(DF.WELL_NAME.isin(result.WELL_NAME))

    assert len(result.loc[result["WELL_NAME"] == "J"]) == 4
    assert len(result.loc[result["WELL_NAME"] == "J"]["X"].unique()) == 3
    assert all(
        z < DF.loc[(DF["WELL_NAME"] == "K") & (DF["LAYER_ID"] == 0)]["Z"].max()
        for z in result.loc[(result["WELL_NAME"] == "K") & (result["LAYER_ID"] == 0)][
            "Z"
        ].values
    )
    assert all(
        z > DF.loc[(DF["WELL_NAME"] == "K") & (DF["LAYER_ID"] == 1)]["Z"].min()
        for z in result.loc[(result["WELL_NAME"] == "K") & (result["LAYER_ID"] == 1)][
            "Z"
        ].values
    )

    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "A"]["DATE"].values[0]
        ).to_pydatetime()
        == D1
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "B"]["DATE"].values[0]
        ).to_pydatetime()
        == D1
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "C"]["DATE"].values[0]
        ).to_pydatetime()
        == D2
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "D"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "E"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "G"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "H"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "I"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
    assert (
        pd.Timestamp(
            result.loc[result["WELL_NAME"] == "J"]["DATE"].values[0]
        ).to_pydatetime()
        == D0
    )
