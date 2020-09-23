from datetime import datetime, timedelta

import pandas as pd

from flownet.data.perforation_strategy import bottom_point, top_point, multiple

D2 = datetime.today()
D1 = D2 - timedelta(days=1)
D0 = D1 - timedelta(days=1)
DATA = {
    "WELL_NAME": [
        "A",
        "A",
        "A",
        "A",
        "B",
        "B",
        "B",
        "B",
        "C",
        "C",
        "D",
        "D",
        "D",
        "E",
        "E",
        "F",
        "G",
        "G",
        "G",
        "G",
        "H",
        "H",
        "H",
        "H",
        "I",
        "I",
        "I",
    ],
    "IJK": [
        (1, 1, 1),
        (2, 2, 2),
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (3, 3, 3),
        (4, 4, 4),
        (5, 5, 5),
        (6, 6, 6),
        (7, 7, 7),
        (7, 7, 7),
        (7, 7, 7),
        (8, 8, 8),
        (8, 8, 8),
        (9, 9, 9),
        (10, 10, 10),
        (11, 11, 11),
        (10, 10, 10),
        (11, 11, 11),
        (12, 12, 12),
        (13, 13, 13),
        (12, 12, 12),
        (13, 13, 13),
        (14, 14, 14),
        (14, 14, 14),
        (14, 14, 14),
    ],
    "X": [
        1,
        2,
        1,
        2,
        3,
        4,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
        8,
        8,
        9,
        10,
        11,
        10,
        11,
        12,
        13,
        12,
        13,
        14,
        14,
        14,
    ],
    "Y": [
        1,
        2,
        1,
        2,
        3,
        4,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
        8,
        8,
        9,
        10,
        11,
        10,
        11,
        12,
        13,
        12,
        13,
        14,
        14,
        14,
    ],
    "Z": [
        1,
        2,
        1,
        2,
        3,
        4,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
        8,
        8,
        9,
        10,
        11,
        10,
        11,
        12,
        13,
        12,
        13,
        14,
        14,
        14,
    ],
    "DATE": [
        D1,
        D1,
        D2,
        D2,
        D1,
        D1,
        D2,
        D2,
        D2,
        D2,
        D0,
        D1,
        D2,
        D0,
        D2,
        D2,
        D0,
        D0,
        D1,
        D1,
        D0,
        D0,
        D1,
        D1,
        D0,
        D1,
        D2,
    ],
    "OPEN": [
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ],
}
DF = pd.DataFrame(DATA)


def test_bottom_point() -> None:

    result = bottom_point(DF)

    assert result.shape[0] == len(DF[DF["OPEN"]]["WELL_NAME"].unique())
    assert all(result["OPEN"].values)

    assert result.loc[result["WELL_NAME"] == "A"]["X"].values[0] == 2
    assert result.loc[result["WELL_NAME"] == "B"]["X"].values[0] == 4
    assert result.loc[result["WELL_NAME"] == "C"]["X"].values[0] == 6
    assert result.loc[result["WELL_NAME"] == "D"]["X"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["X"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["X"].values[0] == 11
    assert result.loc[result["WELL_NAME"] == "H"]["X"].values[0] == 13
    assert result.loc[result["WELL_NAME"] == "I"]["X"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "A"]["Y"].values[0] == 2
    assert result.loc[result["WELL_NAME"] == "B"]["Y"].values[0] == 4
    assert result.loc[result["WELL_NAME"] == "C"]["Y"].values[0] == 6
    assert result.loc[result["WELL_NAME"] == "D"]["Y"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["Y"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["Y"].values[0] == 11
    assert result.loc[result["WELL_NAME"] == "H"]["Y"].values[0] == 13
    assert result.loc[result["WELL_NAME"] == "I"]["Y"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "A"]["Z"].values[0] == 2
    assert result.loc[result["WELL_NAME"] == "B"]["Z"].values[0] == 4
    assert result.loc[result["WELL_NAME"] == "C"]["Z"].values[0] == 6
    assert result.loc[result["WELL_NAME"] == "D"]["Z"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["Z"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["Z"].values[0] == 11
    assert result.loc[result["WELL_NAME"] == "H"]["Z"].values[0] == 13
    assert result.loc[result["WELL_NAME"] == "I"]["Z"].values[0] == 14

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
        == D2
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

    assert all(result["OPEN"].values)
    assert all(result.WELL_NAME.isin(DF.WELL_NAME))
    assert not all(DF.WELL_NAME.isin(result.WELL_NAME))


def test_top_point() -> None:
    result = top_point(DF)

    assert result.shape[0] == len(DF[DF["OPEN"] == True]["WELL_NAME"].unique())
    assert all(result["OPEN"].values)

    assert result.loc[result["WELL_NAME"] == "A"]["X"].values[0] == 1
    assert result.loc[result["WELL_NAME"] == "B"]["X"].values[0] == 3
    assert result.loc[result["WELL_NAME"] == "C"]["X"].values[0] == 5
    assert result.loc[result["WELL_NAME"] == "D"]["X"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["X"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["X"].values[0] == 10
    assert result.loc[result["WELL_NAME"] == "H"]["X"].values[0] == 12
    assert result.loc[result["WELL_NAME"] == "I"]["X"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "A"]["Y"].values[0] == 1
    assert result.loc[result["WELL_NAME"] == "B"]["Y"].values[0] == 3
    assert result.loc[result["WELL_NAME"] == "C"]["Y"].values[0] == 5
    assert result.loc[result["WELL_NAME"] == "D"]["Y"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["Y"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["Y"].values[0] == 10
    assert result.loc[result["WELL_NAME"] == "H"]["Y"].values[0] == 12
    assert result.loc[result["WELL_NAME"] == "I"]["Y"].values[0] == 14
    assert result.loc[result["WELL_NAME"] == "A"]["Z"].values[0] == 1
    assert result.loc[result["WELL_NAME"] == "B"]["Z"].values[0] == 3
    assert result.loc[result["WELL_NAME"] == "C"]["Z"].values[0] == 5
    assert result.loc[result["WELL_NAME"] == "D"]["Z"].values[0] == 7
    assert result.loc[result["WELL_NAME"] == "E"]["Z"].values[0] == 8
    assert result.loc[result["WELL_NAME"] == "G"]["Z"].values[0] == 10
    assert result.loc[result["WELL_NAME"] == "H"]["Z"].values[0] == 12
    assert result.loc[result["WELL_NAME"] == "I"]["Y"].values[0] == 14

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
        == D2
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

    assert all(result["OPEN"].values)
    assert all(result.WELL_NAME.isin(DF.WELL_NAME))
    assert not all(DF.WELL_NAME.isin(result.WELL_NAME))


def test_multiple() -> None:
    result = multiple(DF)

    assert not all(result["OPEN"].values)
    assert all(result.X.isin(DF.X).astype(float))
    assert all(result.Y.isin(DF.Y).astype(float))
    assert all(result.Z.isin(DF.Z).astype(float))
    assert all(result.WELL_NAME.isin(DF.WELL_NAME))
    assert all(DF.WELL_NAME.isin(result.WELL_NAME))

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
