"""
Extract COMPDAT, WELSEGS and COMPSEGS from an Eclipse deck

"""

import argparse
import datetime
import logging
from typing import Dict, List, Optional, Union

import pandas as pd
from ecl2df.common import (
    merge_zones,
    parse_opmio_date_rec,
    parse_opmio_deckrecord,
    parse_opmio_tstep_rec,
    write_dframe_stdout_file,
)
from ecl2df.eclfiles import EclFiles
from ecl2df.grid import merge_initvectors

try:
    import opm.io.deck  # pylint: disable=unused-import
except ImportError:
    # Allow parts of ecl2df to work without OPM:
    pass


logger = logging.getLogger(__name__)

"""OPM authors and Roxar RMS authors have interpreted the Eclipse
documentation ever so slightly different when naming the data.

For COMPDAT dataframe columnnames, we prefer the RMS terms due to the
one very long one, and mixed-case in opm
"""
COMPDAT_RENAMER: Dict[str, str] = {
    "WELL": "WELL",
    "I": "I",
    "J": "J",
    "K1": "K1",
    "K2": "K2",
    "STATE": "OP/SH",
    "SAT_TABLE": "SATN",
    "CONNECTION_TRANSMISSIBILITY_FACTOR": "TRAN",
    "DIAMETER": "WBDIA",
    "Kh": "KH",
    "SKIN": "SKIN",
    "D_FACTOR": "DFACT",
    "DIR": "DIR",
    "PR": "PEQVR",
}

# Workaround an inconsistency in JSON-files for OPM-common < 2021.04:
WSEG_RENAMER: Dict[str, str] = {
    "SEG1": "SEGMENT1",
    "SEG2": "SEGMENT2",
}

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def deck2dfs(
    deck: "opm.io.Deck",
    start_date: Optional[Union[str, datetime.date]] = None,
    unroll: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Loop through the deck and pick up information found

    The loop over the deck is a state machine, as it has to pick up dates and
    potential information from the WELSPECS keyword.

    Args:
        deck: A deck representing the schedule
            Does not have to be a full Eclipse deck, an include file is sufficient
        start_date: The default date to use for
            events where the DATE or START keyword is not found in advance.
            Default: None
        unroll: Whether to unroll rows that cover a range,
            like K1 and K2 in COMPDAT and in WELSEGS. Defaults to True.

    Returns:
        Dictionary with dataframes, at least for COMPDAT, COMPSEGS and WELSEGS.
    """
    compdatrecords = []  # List of dicts of every line in input file
    compsegsrecords = []
    welopenrecords = []
    welsegsrecords = []
    wsegsicdrecords = []
    wsegaicdrecords = []
    wsegvalvrecords = []
    welspecs = {}
    date = start_date  # DATE column will always be there, but can contain NaN/None
    for idx, kword in enumerate(deck):  # pylint: disable=too-many-nested-blocks
        if kword.name == ("DATES", "START"):
            for rec in kword:
                date = parse_opmio_date_rec(rec)
                logger.info("Parsing at date %s", str(date))
        elif kword.name == "TSTEP":
            if not date:
                logger.critical("Can't use TSTEP when there is no start_date")
                return {}
            for rec in kword:
                steplist = parse_opmio_tstep_rec(rec)
                # Assuming not LAB units, then the unit is days.
                days = sum(steplist)
                assert isinstance(date, datetime.date)
                date += datetime.timedelta(days=days)
                logger.info(
                    "Advancing %s days to %s through TSTEP", str(days), str(date)
                )
        elif kword.name == "WELSPECS":
            # Information from WELSPECS are to be used in case
            # 0 or 1* is used in the I or J column in COMPDAT
            # Only the latest information pr. well is stored.
            for wellrec in kword:
                welspecs_rec_dict = parse_opmio_deckrecord(wellrec, "WELSPECS")
                welspecs[welspecs_rec_dict["WELL"]] = {
                    "I": welspecs_rec_dict["HEAD_I"],
                    "J": welspecs_rec_dict["HEAD_J"],
                }
        elif kword.name == "COMPDAT":
            for rec in kword:  # Loop over the lines inside COMPDAT record
                rec_data = parse_opmio_deckrecord(
                    rec, "COMPDAT", renamer=COMPDAT_RENAMER
                )
                rec_data["DATE"] = date
                rec_data["KEYWORD_IDX"] = idx
                # start of code changes
                if rec_data["WELL"] != "*":
                    if rec_data["I"] == 0:
                        if rec_data["WELL"] not in welspecs:
                            raise ValueError(
                                "WELSPECS must be provided when I is defaulted in COMPDAT"
                            )
                        rec_data["I"] = welspecs[rec_data["WELL"]]["I"]
                    if rec_data["J"] == 0:
                        if rec_data["WELL"] not in welspecs:
                            raise ValueError(
                                "WELSPECS must be provided when J is defaulted in COMPDAT"
                            )
                        rec_data["J"] = welspecs[rec_data["WELL"]]["J"]
                    compdatrecords.append(rec_data)
                else:
                    # go through all known wells and assign correct indices
                    for item in welspecs.items():
                        rec_data["WELL"] = item[0]
                        rec_data["I"] = item[1]["I"]
                        rec_data["J"] = item[1]["J"]
                        compdatrecords.append(rec_data.copy())
                # end of code changes
        elif kword.name == "WSEGSICD":
            for rec in kword:  # Loop over the lines inside WSEGSICD record
                rec_data = parse_opmio_deckrecord(rec, "WSEGSICD", renamer=WSEG_RENAMER)
                rec_data["DATE"] = date
                rec_data["KEYWORD_IDX"] = idx
                wsegsicdrecords.append(rec_data)
        elif kword.name == "WSEGAICD":
            for rec in kword:  # Loop over the lines inside WSEGAICD record
                rec_data = parse_opmio_deckrecord(rec, "WSEGAICD", renamer=WSEG_RENAMER)
                rec_data["DATE"] = date
                rec_data["KEYWORD_IDX"] = idx
                wsegaicdrecords.append(rec_data)
        elif kword.name == "WSEGVALV":
            for rec in kword:  # Loop over the lines inside WSEGVALV record
                rec_data = parse_opmio_deckrecord(rec, "WSEGVALV")
                rec_data["DATE"] = date
                rec_data["KEYWORD_IDX"] = idx
                wsegvalvrecords.append(rec_data)
        elif kword.name == "COMPSEGS":
            wellname = parse_opmio_deckrecord(
                kword[0], "COMPSEGS", itemlistname="records", recordindex=0
            )["WELL"]
            for recidx in range(1, len(kword)):
                rec = kword[recidx]
                rec_data = parse_opmio_deckrecord(
                    rec, "COMPSEGS", itemlistname="records", recordindex=1
                )
                rec_data["WELL"] = wellname
                rec_data["DATE"] = date
                compsegsrecords.append(rec_data)
        elif kword.name == "WELOPEN":
            for rec in kword:
                rec_data = parse_opmio_deckrecord(rec, "WELOPEN")
                rec_data["DATE"] = date
                rec_data["KEYWORD_IDX"] = idx
                if rec_data["STATUS"] not in ["OPEN", "SHUT", "STOP", "AUTO"]:
                    rec_data["STATUS"] = "SHUT"
                    logger.warning(
                        "WELOPEN status %s is not a valid "
                        "COMPDAT state. Using 'SHUT' instead.",
                        rec_data["STATUS"],
                    )
                welopenrecords.append(rec_data)
        elif kword.name == "WELSEGS":
            # First record contains meta-information for well
            # (opm deck returns default values for unspecified items.)
            welsegsdict = parse_opmio_deckrecord(
                kword[0], "WELSEGS", itemlistname="records", recordindex=0
            )
            # Loop over all subsequent records.
            for recidx in range(1, len(kword)):
                rec = kword[recidx]
                # WARNING: We assume that SEGMENT1 === SEGMENT2 (!!!) (if not,
                # we need to loop over a range just as for layer in compdat)
                rec_data = welsegsdict.copy()
                rec_data["DATE"] = date
                rec_data.update(
                    parse_opmio_deckrecord(
                        rec, "WELSEGS", itemlistname="records", recordindex=1
                    )
                )
                if "INFO_TYPE" in rec_data and rec_data["INFO_TYPE"] == "ABS":
                    rec_data["SEGMENT_MD"] = rec_data["SEGMENT_LENGTH"]
                welsegsrecords.append(rec_data)

    compdat_df = pd.DataFrame(compdatrecords)
    welopen_df = pd.DataFrame(welopenrecords)

    if unroll and not compdat_df.empty:
        compdat_df = unrolldf(compdat_df, "K1", "K2")

    if not welopen_df.empty:
        compdat_df = applywelopen(compdat_df, welopen_df)

    compsegs_df = pd.DataFrame(compsegsrecords)
    welsegs_df = pd.DataFrame(welsegsrecords)
    wsegsicd_df = pd.DataFrame(wsegsicdrecords)
    wsegaicd_df = pd.DataFrame(wsegaicdrecords)
    wsegvalv_df = pd.DataFrame(wsegvalvrecords)

    if unroll and not welsegs_df.empty:
        welsegs_df = unrolldf(welsegs_df, "SEGMENT1", "SEGMENT2")

    if unroll and not wsegsicd_df.empty:
        wsegsicd_df = unrolldf(wsegsicd_df, "SEGMENT1", "SEGMENT2")

    if unroll and not wsegaicd_df.empty:
        wsegaicd_df = unrolldf(wsegaicd_df, "SEGMENT1", "SEGMENT2")

    if "KEYWORD_IDX" in compdat_df.columns:
        compdat_df.drop(["KEYWORD_IDX"], axis=1, inplace=True)

    if "KEYWORD_IDX" in wsegsicd_df.columns:
        wsegsicd_df.drop(["KEYWORD_IDX"], axis=1, inplace=True)

    if "KEYWORD_IDX" in wsegaicd_df.columns:
        wsegaicd_df.drop(["KEYWORD_IDX"], axis=1, inplace=True)

    if "KEYWORD_IDX" in wsegvalv_df.columns:
        wsegvalv_df.drop(["KEYWORD_IDX"], axis=1, inplace=True)

    return dict(
        COMPDAT=compdat_df,
        COMPSEGS=compsegs_df,
        WELSEGS=welsegs_df,
        WSEGSICD=wsegsicd_df,
        WSEGAICD=wsegaicd_df,
        WSEGVALV=wsegvalv_df,
    )


def postprocess():
    """Postprocessing of the compdat data, merging.

    This function is NOT FINISHED"""
    # compdat_df = pd.read_csv("compdat.csv")
    compsegs_df = pd.read_csv("compsegs.csv")
    welsegs_df = pd.read_csv("welsegs.csv")

    #  We need different handling of ICD's and non-ICD wells due
    #  to the complex WELSEGS structure:
    #
    # ICD wells:
    # 1. First compdata is merged with compsegs (non-ICD
    #    should be stripped away).
    # 2. Then that product is merged with welsegs on 'branch'
    # 3. Then that product is merged again with welsegs, where
    #    we join on 'join_segment' and 'segment'
    # 4. Then we finally have the mapping between completed
    #    cells and branch number
    #
    # Non-ICD wells:
    # 1. Merge compdata and compsegs
    # 2. Then we are ready.. compsegs contains the correct branch number

    # compdatsegs = pd.merge(compdat_df,
    #                        compsegs_df, on=["date", "well", "i", "j", "k"])
    # WARNING: Only correct for dual-branch wells,
    # not triple-branach wells with ICD..
    compsegs_icd_df = compsegs_df[compsegs_df.branch > 2]
    # icd_wells = compsegs_icd_df.well.unique()
    compdatsegwel_icd_df = pd.merge(
        compsegs_icd_df, welsegs_df, on=["date", "well", "branch"]
    )
    del compdatsegwel_icd_df["segment"]  # we don't need this
    compdatsegwel_icd_df.rename(columns={"branch": "icd_branch"}, inplace=True)
    compdatsegwel_icd_df.rename(columns={"join_segment": "segment"}, inplace=True)
    # alldata_icd = pd.merge(
    #     compdatsegwel_icd_df, welsegs_df, on=["date", "well", "segment"]
    # )


def unrolldf(
    dframe: pd.DataFrame, start_column: str = "K1", end_column: str = "K2"
) -> pd.DataFrame:
    """Unroll dataframes, where some column pairs indicate
    a range where data applies.

    After unrolling, column pairs with ranges are transformed
    into multiple rows, with no ranges.

    Example: COMPDAT supports K1, K2 intervals for multiple cells::

      COMPDAT
        'OP1' 33 44 10 11 /
      /

    is transformed/unrolled so it would be equal to::

      COMPDAT
        'OP1' 33 44 10 10 /
        'OP1' 33 44 11 11 /
      /

    The latter is easier to work with in Pandas dataframes

    Args:
        dframe: Dataframe to be unrolled
        start_column: Column name that contains the start of
            a range.
        end_column Column name that contains the corresponding end
            of the range.

    Returns:
        Dataframe, unrolled version. Identical to input if none of
        rows had any ranges.
    """
    if dframe.empty:
        return dframe
    if start_column not in dframe and end_column not in dframe:
        logger.warning(
            "Cannot unroll on non-existing columns %s and %s", start_column, end_column
        )
        return dframe
    start_eq_end_bools = dframe[start_column] == dframe[end_column]
    unrolled = dframe[start_eq_end_bools]
    list_unrolled = []
    if (~start_eq_end_bools).any():
        for _, rangerow in dframe[~start_eq_end_bools].iterrows():
            for k_idx in range(
                int(rangerow[start_column]), int(rangerow[end_column]) + 1
            ):
                rangerow[start_column] = k_idx
                rangerow[end_column] = k_idx
                list_unrolled.append(rangerow.copy())
    if list_unrolled:
        unrolled = pd.concat([unrolled, pd.DataFrame(list_unrolled)], axis=0)
    return unrolled


def applywelopen(compdat_df: pd.DataFrame, welopen_df: pd.DataFrame) -> pd.DataFrame:
    """Apply WELOPEN actions to the COMPDAT dataframe.

    Each record in the WELOPEN keyword acts as an operator on existing connections
    in existing wells.

    Example: COMPDAT and WELOPEN keyword::

      COMPDAT
       'OP1' 33 44 10 11 'OPEN' /
       'OP2' 66 44 10 11 'OPEN' /
      /
      WELOPEN
       'OP1' SHUT /
       'OP2' SHUT 66 44 10 /
      /

    This deck would define two wells where OP1 and OP2 have two connected grid cells
    each. Although the COMPDAT defines all connections to be open, WELOPEN overwrites
    this: all connections in OP1 will be SHUT and in OP2 the upper connection will
    be SHUT.

    WELOPEN can also be used at different dates and changes therefore the state of
    connections without explicit use of the COMPDAT keyword. This function translates
    WELOPEN actions into explicit additional COMPDAT definitions in the exported df.

    Args:
        compdat_df: Dataframe with unrolled COMPDAT data
        welopen_df: Dataframe with WELOPEN actions

    Returns:
        Dataframe, compdat_df now including WELOPEN actions

    """
    welopen_df = welopen_df.astype(object).where(pd.notnull(welopen_df), None)
    # pylint: disable=too-many-boolean-expressions
    for _, row in welopen_df.iterrows():
        if (row["I"] is None and row["J"] is None and row["K"] is None) or (
            row["I"] <= 0 and row["J"] <= 0 and row["K"] <= 0
        ):
            previous_state = compdat_df[
                (compdat_df["WELL"] == row["WELL"])
                & (compdat_df["KEYWORD_IDX"] < row["KEYWORD_IDX"])
            ].drop_duplicates(subset=["I", "J", "K1", "K2"], keep="last")
        elif row["I"] and row["J"] and row["K"]:
            previous_state = compdat_df[
                (compdat_df["WELL"] == row["WELL"])
                & (compdat_df["KEYWORD_IDX"] < row["KEYWORD_IDX"])
                & (compdat_df["I"] == row["I"])
                & (compdat_df["J"] == row["J"])
                & (compdat_df["K1"] == row["K"])
                & (compdat_df["K2"] == row["K"])
            ].drop_duplicates(subset=["I", "J", "K1", "K2"], keep="last")
        elif row["I"] <= 0 and row["J"] <= 0 and row["K"] <= 0:
            previous_state = compdat_df[
                (compdat_df["WELL"] == row["WELL"])
                & (compdat_df["KEYWORD_IDX"] < row["KEYWORD_IDX"])
            ].drop_duplicates(subset=["I", "J", "K1", "K2"], keep="last")
        elif row["I"] <= 0 and row["J"] <= 0 and row["K"] <= 0:
            previous_state = compdat_df[
                (compdat_df["WELL"] == row["WELL"])
                & (compdat_df["KEYWORD_IDX"] < row["KEYWORD_IDX"])
            ].drop_duplicates(subset=["I", "J", "K1", "K2"], keep="last")
        else:
            raise ValueError(
                "A WELOPEN keyword contains data that could not be parsed. "
                f"\n {str(row)} "
            )

        if (row["C1"] is not None and row["C1"] > 0) or (
            row["C2"] is not None and row["C2"]
        ) > 0:
            raise ValueError(
                "Lumped connections are not supported by ecl2df in a WELOPEN keyword. "
                f"\n{str(row)} "
            )

        if previous_state.empty:
            raise ValueError(
                "A WELOPEN keyword is not acting on any existing connection. "
                f"\n {str(row)} "
            )

        new_state = previous_state

        # The COMPDAT DataFrame uses COMPDAT_RENAMER and therefore uses "OP/SH" as a
        # column name for the state of a well. WELOPEN uses "STATUS" for the state
        # column name and therefore a translation step needs to be done. The
        # underlying problem is that the opm-common definitions for the state of a
        # well in COMPDAT and WELOPEN are not identical. These translation steps can
        # be dropped when unity in the opm-common keyword definitions is reached.
        new_state["OP/SH"] = row["STATUS"]
        new_state["KEYWORD_IDX"] = row["KEYWORD_IDX"]
        new_state["DATE"] = row["DATE"]

        compdat_df = compdat_df.append(new_state)

    if not compdat_df.empty:
        compdat_df = (
            compdat_df.sort_values(by=["KEYWORD_IDX"])
            .drop_duplicates(subset=["I", "J", "K1", "K2", "DATE"], keep="last")
            .reset_index(drop=True)
        )

    return compdat_df


def fill_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up sys.argv parsers.

    Arguments:
        parser: parser to fill with arguments
    """
    parser.add_argument("DATAFILE", help="Name of Eclipse DATA file.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Name of output csv file.",
        default="compdat.csv",
    )
    parser.add_argument(
        "--initvectors",
        help="List of INIT vectors to merge into the data",
        nargs="+",
        default=None,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
    return parser


def compdat_main(args):
    """Entry-point for module, for command line utility"""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    eclfiles = EclFiles(args.DATAFILE)
    compdat_df = df(eclfiles, initvectors=args.initvectors)
    if compdat_df.empty:
        logger.warning("Empty COMPDAT data being written to disk!")
    write_dframe_stdout_file(compdat_df, args.output, index=False, caller_logger=logger)


def df(eclfiles: EclFiles, initvectors: Optional[List[str]] = None) -> pd.DataFrame:
    """Main function for Python API users

    Supports only COMPDAT information for now. Will
    add a zone-name if a zonefile is found alongside

    Returns:
        pd.Dataframe with one row pr cell to well connection
    """
    compdat_df = deck2dfs(eclfiles.get_ecldeck())["COMPDAT"]
    compdat_df = unrolldf(compdat_df)

    if initvectors:
        compdat_df = merge_initvectors(
            eclfiles, compdat_df, initvectors, ijknames=["I", "J", "K1"]
        )

    zonemap = eclfiles.get_zonemap()
    if zonemap:
        logger.info("Merging zonemap into compdat")
        compdat_df = merge_zones(compdat_df, zonemap)

    return compdat_df
