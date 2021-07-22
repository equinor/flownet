from pathlib import Path
from typing import Union

import pandas as pd


class CSVData:
    """
    CSV data source class

    Args:
         input_data: Full path to CSV file to load production data from

    """

    def __init__(
        self,
        input_data: Union[Path, str],
    ):
        super().__init__()

        self._input_data: Path = Path(input_data)

    # pylint: disable=too-many-branches
    def _production_data(self) -> pd.DataFrame:
        """
        Function to read production data for all producers and injectors from a CSV file.

        Returns:
            A DataFrame with a DateTimeIndex and the following columns:
                - date          equal to index
                - WELL_NAME     Well name as used in Flow
                - WOPR          Well Oil Production Rate
                - WGPR          Well Gas Production Rate
                - WWPR          Well Water Production Rate
                - WOPT          Well Cumulative Oil Production
                - WGPT          Well Cumulative Gas Production
                - WWPT          Well Cumulative Water Production
                - WBHP          Well Bottom Hole Pressure
                - WTHP          Well Tubing Head Pressure
                - WGIR          Well Gas Injection Rate
                - WWIR          Well Water Injection Rate
                - WSPR          Well Salt Production Rate
                - WSIR          Well Salt Injection Rate
                - WSPT          Well Cumulative Salt Production
                - WSIT          Well Cumulative Salt Injection
                - WSTAT         Well status (OPEN, SHUT, STOP)
                - TYPE          Well Type: "OP", "GP", "WI", "GI"
                - PHASE         Main producing/injecting phase fluid: "OIL", "GAS", "WATER"

        Todo:
            * Remove depreciation warning suppression when solved in LibEcl.
            * Improve robustness pf setting of Phase and Type.

        """
        df_production_data = pd.read_csv(self._input_data)
        df_production_data["date"] = pd.to_datetime(df_production_data["date"]).dt.date
        df_production_data = df_production_data.set_index("date", drop=False)
        return df_production_data

    @property
    def production(self) -> pd.DataFrame:
        """dataframe with all production data"""
        return self._production_data()
