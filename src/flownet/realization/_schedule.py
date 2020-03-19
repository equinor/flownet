import os
import pathlib
from typing import List, Union, Optional, Dict
import datetime
from operator import attrgetter

import numpy as np
import pandas as pd

from ._simulation_keywords import Keyword, COMPDAT, WCONHIST, WCONINJH, WELSPECS
from ..network_model import NetworkModel


class Schedule:
    """
    Python class to to store all schedule items for the Flownet run.

    """

    def __init__(
        self,
        network: NetworkModel,
        df_production_data: pd.DataFrame,
        case_name: str,
        control_mode: str = "RESV",
    ):
        self._schedule_items: List = []
        self._network: NetworkModel = network
        self._df_production_data: pd.DataFrame = df_production_data
        self._control_mode: str = control_mode
        self._case_name: str = case_name

        self._create_schedule()

    def _create_schedule(self):
        """
        Helper function that calls all functions involved in creating the schedule for the FlowNet run.
        """
        self._calculate_entity_dates()
        self._calculate_compdat()
        self._calculate_welspecs()
        self._calculate_wconhist()
        self._calculate_wconinjh()

    def _calculate_entity_dates(self):
        """
        Helper function that calculate the start dates of all entities.

        Returns:
            Nothing

        """
        self._df_production_data["ent_date"] = pd.np.nan
        for _, row in self._network.df_entity_connections.iterrows():
            for entity in ["start_entity", "end_entity"]:
                date = None
                well_name = row[[entity]][0]
                if row[[entity]].values[0] and row[[entity]].values[0] != "aquifer":
                    date = self._retrieve_date_first_non_zero_prodinj(
                        self._df_production_data, well_name
                    )
                self._df_production_data.loc[
                    self._df_production_data["WELL_NAME"] == well_name, "ent_date"
                ] = date

    def _calculate_compdat(self):
        """
        Helper Function that generates the COMPDAT keywords based on geometrical information from the NetworkModel.

        Returns:
            Nothing

        """
        for index, row in self._network.df_entity_connections.iterrows():
            for e_index, entity in enumerate(["start_entity", "end_entity"]):
                if row[[entity]].values[0] and row[[entity]].values[0] != "aquifer":
                    entity_index = 0 - e_index  # first, or last
                    grid_mask = self._network.active_mask(model_index=index)
                    i = self._network.grid[grid_mask].index[entity_index]

                    date = self._retrieve_date_first_non_zero_prodinj(
                        self._df_production_data, row[[entity]][0]
                    )
                    if date:
                        self.append(
                            COMPDAT(
                                date=date,
                                well_name=row[[entity]][0],
                                i=i,
                                j=0,
                                k1=0,
                                k2=0,
                                rw=0.22,
                                status="OPEN",
                            )
                        )
                    else:
                        print(
                            f"Skipping COMPDAT for well {row[[entity]][0]} as no positive production or "
                            f"injection data has been supplied."
                        )

    def _calculate_welspecs(self):
        """
        Helper function that generates the WELSPECS keywords based which wells there are in the production data.

        Returns:
            Nothing

        """
        for well_name in self._df_production_data["WELL_NAME"].unique():
            date = self._retrieve_date_first_non_zero_prodinj(
                self._df_production_data, well_name
            )

            if date:
                well_dates = self._df_production_data[
                    (self._df_production_data["WELL_NAME"] == well_name)
                    & (self._df_production_data["date"] == date)
                ]

                phase = well_dates["PHASE"].values[0]
                self.append(
                    WELSPECS(
                        date=date,
                        well_name=well_name,
                        group_name="WELLS",  # Wells directly connected to FIELD gives
                        # segmentation fault in flow and error in Eclipse
                        i=self.get_compdat(well_name)[0].i,
                        j=self.get_compdat(well_name)[0].j,
                        phase=phase,
                    )
                )
            else:
                print(
                    f"Skipping WELSPECS for well {well_name} as no positive production or injection data"
                    f" has been supplied."
                )

    def _calculate_wconhist(self):
        """
        Helper function that generates the WCONHIST keywords based on loaded production data.
        Currently the data is loaded from hard-coded filepath.

        A WCONHIST keyword will be written out for each line in the provided production data.
        Therefore, potential up-sampling needs to be done before the schedule is created.

        Returns:
            Nothing

        """
        vfp_tables = self.get_vfp()

        for _, value in self._df_production_data.iterrows():
            start_date = self.get_well_start_date(value["WELL_NAME"])

            if value["TYPE"] == "OP" and start_date and value["date"] >= start_date:
                self.append(
                    WCONHIST(
                        date=value["date"],
                        well_name=value["WELL_NAME"],
                        control_mode=self._control_mode,
                        vfp_table=vfp_tables[value["WELL_NAME"]],
                        oil_rate=value["WOPR"],
                        water_rate=value["WWPR"],
                        gas_rate=value["WGPR"],
                        bhp=value["WBHP"],
                        thp=value["WTHP"],
                    )
                )

    def _calculate_wconinjh(self):
        """
        Helper function that generates the WCONINJ keywords based on loaded production data.
        Currently the data is loaded from hard-coded filepath.

        A WCONINJH keyword will be written out for each line in the provided
        production data. Therefore, potential up-sampling needs to be done before
        the schedule is created.

        Returns:
            Nothing

        """
        for _, value in self._df_production_data.iterrows():
            start_date = self.get_well_start_date(value["WELL_NAME"])
            if value["TYPE"] == "WI" and start_date and value["date"] >= start_date:
                self.append(
                    WCONINJH(
                        date=value["date"],
                        well_name=value["WELL_NAME"],
                        inj_type="WATER",
                        status="OPEN",
                        rate=value["WWIR"],
                        bhp=value["WBHP"],
                        thp=value["WTHP"],
                    )
                )
            elif value["TYPE"] == "GI" and start_date and value["date"] >= start_date:
                self.append(
                    WCONINJH(
                        date=value["date"],
                        well_name=value["WELL_NAME"],
                        inj_type="GAS",
                        status="OPEN",
                        rate=value["WGIR"],
                        bhp=value["WBHP"],
                        thp=value["WTHP"],
                    )
                )

    def __getitem__(
        self, item: Optional[Union[int, datetime.date, Keyword]]
    ) -> List[Keyword]:
        """

        Args:
            item: Input to do look-up for

        Returns:
            List of Keywords for the specified lookup filter
        """
        if isinstance(item, int):
            output = self._schedule_items[item]
        elif isinstance(item, datetime.date):
            output = self.get_keywords(date=item)
        elif isinstance(item, Keyword):
            output = self.get_keywords(kw_class=item)
        else:
            raise ValueError(f"Could not retrieve schedule items for '{item}'")

        return output

    def __len__(self) -> int:
        """
        Function to retrieve the number of schedule items defined in the Schedule
        object. This is *not* equal to the number of keywords defined - as
        keywords in the final output might consist of multiple wrapped entries.

        Returns:
            The number of defined schedule items

        """
        return len(self._schedule_items)

    def sort(self):
        """
        Function to return a list if schedule items (keywords) sorted on
        by date

        Returns:
            Date-sorted schedule items

        """
        return sorted(self._schedule_items, key=attrgetter("date"))

    def append(self, _keyword):
        """
        Helper function to append Keyword's to the Schedule instance

        Args:
            _keyword: Keyword instance to append

        Returns:
            Nothing

        """
        self._schedule_items.append(_keyword)

    def get_dates(self) -> List[datetime.date]:
        """
        Function to retrieve a sorted list of all unique dates in the schedule object.

        Returns:
            Listing of PyDate's in the Schedule

        """
        # pylint: disable=R1718
        return sorted(set([kw.date for kw in self._schedule_items]))

    def get_first_date(self) -> datetime.date:
        """
        Helper function to look-up the first date in the schedule.

        Returns:
            The Schedule's first date.

        """
        return self.get_dates()[0]

    def get_keywords(
        self,
        date: datetime.date = None,
        kw_class: Optional[Union[Keyword, str]] = None,
        well_name: str = None,
        ignore_nan: str = None,
    ) -> List[Keyword]:
        """
        Returns a list of all keywords at a given date and/or of a
        particular keyword class.

        Args:
            date: Date at which to lookup keywords
            kw_class: keyword class or class name string
            well_name: well name
            ignore_nan: keyword attribute to ignore nan values

        Returns:
            keywords at specified date and/or with a specific well name and/or of a specific keyword class

        """
        if date and not kw_class and not well_name:
            output = [kw for kw in self._schedule_items if kw.date == date]
        elif kw_class and not date and not well_name:
            output = [
                kw for kw in self._schedule_items if kw.__class__.__name__ == kw_class
            ]
        elif date and kw_class and not well_name:
            output = [
                kw
                for kw in self._schedule_items
                if kw.date == date and kw.__class__.__name__ == kw_class
            ]
        elif well_name and not date and not kw_class:
            output = [kw for kw in self._schedule_items if kw.well_name == well_name]
        elif date and not kw_class and well_name:
            output = [
                kw
                for kw in self._schedule_items
                if kw.date == date and kw.well_name == well_name
            ]
        elif kw_class and not date and well_name:
            output = [
                kw
                for kw in self._schedule_items
                if kw.__class__.__name__ == kw_class and kw.well_name == well_name
            ]
        elif date and kw_class and well_name:
            output = [
                kw
                for kw in self._schedule_items
                if kw.date == date
                and kw.__class__.__name__ == kw_class
                and kw.well_name == well_name
            ]
        else:
            raise ValueError(
                "Could not retrieve keywords. Provide either a date, a keyword type or a well_name."
            )

        if ignore_nan:
            output = [kw for kw in output if not np.isnan(getattr(kw, ignore_nan))]

        return output

    def get_wells(self, kw_class: Optional[Union[str, Keyword]] = None) -> List[str]:
        """
        Function to retrieve all wells that are associated to any keyword
        in the Schedule or associated to a particular kw_class.

        Args:
            kw_class: keyword class or class name string

        Returns:
            List of strings of unique well names

        """
        if kw_class:
            output = list(
                {
                    kw.well_name
                    for kw in self._schedule_items
                    if (kw.__class__.__name__ == str(kw_class))
                }
            )
        else:
            output = list({kw.well_name for kw in self._schedule_items})

        return output

    def num_wells(self) -> int:
        """
        Function to retrieve the number of unique wells in the Schedule associated
        to any keyword in the Schedule.

        Returns:
            Number of unique wells in the schedule

        """
        return len(self.get_wells())

    def num_connections(self, well_name: str) -> int:
        """
        Function to retrieve the number of well connections to the grid
        in a specific well.

        Args:
            well_name: Well name to retrieve the connections for

        Returns:
            Number of connections with the grid in the well

        """

        return len(self.get_compdat(well_name=well_name))

    def max_connections(self) -> int:
        """
        Function to retrieve the maximum number of well connections to the grid
        in any defined well in the Schedule.

        Returns:
            Maximum number of connections with the grid in a single well in the schedule

        """
        return max([self.num_connections(well) for well in self.get_wells()], default=0)

    def get_compdat(self, well_name: str = None) -> List[COMPDAT]:
        """
        Function to retrieve all COMPDAT entries defined for a particular well.

        Args:
            well_name: Name of the well to get COMPDAT's for

        Returns:
            COMPDAT keywords defined for a particular well

        """
        return [
            kw
            for kw in self._schedule_items
            if isinstance(kw, COMPDAT) and kw.well_name == well_name
        ]

    def get_well_start_date(
        self, well_name: Optional[str] = None
    ) -> Union[datetime.date, None]:
        """
        Function to retrieve the start date of a well.

        Args:
            well_name: Name of the well to get the start-date for

        Returns:
            Returns the date if found, otherwise None

        """
        compdat_dates = [
            kw.date
            for kw in self._schedule_items
            if isinstance(kw, COMPDAT) and kw.well_name == well_name
        ]

        return min(compdat_dates) if compdat_dates else None

    def get_vfp(self) -> Dict:
        """
         Helper function to retrieve the VFP tables associated with all wells.

         Returns:
             A dictionary of VFP table numbers or None if no such date exist

        """
        vfp_tables = {}
        module_folder = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
        static_source_path = (
            module_folder / "static" / f"SCHEDULE_{self._case_name}.inc"
        )
        wells = self.get_wells()
        for well in wells:
            if static_source_path.is_file():
                vfp_tables[well] = str(1)
            else:
                vfp_tables[well] = "1*"

        return vfp_tables

    def get_nr_observations(self, training_set_fraction: float) -> int:
        """
        Helper function to retrieve the number of unique observations in the training process.

        Args:
            training_set_fraction: Fraction of dates to use in the training process.

        Returns:
            The number of unique observations used in the training process

        """
        # pylint: disable=too-many-branches

        num_training_dates = round(len(self.get_dates()) * training_set_fraction)
        i = 0

        for date in self.get_dates()[0 : num_training_dates - 1]:
            keywords_wconhist: List[Keyword] = self.get_keywords(
                date=date, kw_class="WCONHIST"
            )
            if keywords_wconhist:
                for keyword_wconhist in keywords_wconhist:
                    if not self.get_well_start_date(keyword_wconhist.well_name) == date:
                        if not np.isnan(keyword_wconhist.oil_rate):
                            i += 1
                        if not np.isnan(keyword_wconhist.gas_rate):
                            i += 1
                        if not np.isnan(keyword_wconhist.bhp):
                            i += 1
                        if not np.isnan(keyword_wconhist.thp):
                            i += 1

            keywords_wconinjh: List[Keyword] = self.get_keywords(
                date=date, kw_class="WCONINJH"
            )
            if keywords_wconinjh:
                for keyword__wconinjh in keywords_wconinjh:
                    if (
                        not self.get_well_start_date(keyword__wconinjh.well_name)
                        == date
                    ):
                        if not np.isnan(keyword__wconinjh.bhp):
                            i += 1
                        if not np.isnan(keyword__wconinjh.thp):
                            i += 1

        return i

    @staticmethod
    def _retrieve_date_first_non_zero_prodinj(
        df_production_data: pd.DataFrame, well_name: str
    ) -> Union[datetime.date, None]:
        """
        Helper function to retrieve the date of first non-zero production or injection.

        Args:
            df_production_data: DataFrame with all production data
            well_name: Name of the well to retrieve date for

        Returns:
            The first date of non-zero production or None if no such date exist

        """
        well_data = df_production_data[df_production_data["WELL_NAME"] == well_name]

        well_data = well_data[well_data != 0.0][
            ["date"]
            + [
                column
                for column in ["WOPR", "WWPR", "WGPR", "WBHP", "WTHP", "WGIR", "WWIR"]
                if column in well_data.columns
            ]
        ].dropna(axis=0, how="all")

        return well_data.iloc[0]["date"] if well_data.shape[0] > 0 else None
