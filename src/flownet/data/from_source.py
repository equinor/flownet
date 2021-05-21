from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class FromSource(ABC):
    """
    Abstract base class that defines the minimum requirements for a FromSource child class.
    """

    @property
    @abstractmethod
    def production(self) -> pd.DataFrame:
        raise NotImplementedError(
            "The production property is required to be implemented in a FromSource class."
        )

    @abstractmethod
    def get_well_connections(self, perforation_handling_strategy: str) -> pd.DataFrame:
        raise NotImplementedError(
            "The  get_well_connections method is required to be implemented in a FromSource class."
        )

    @property
    @abstractmethod
    def faults(self) -> Optional[pd.DataFrame]:
        raise NotImplementedError(
            "The faults property is required to be implemented in a FromSource class."
        )

    @property
    @abstractmethod
    def well_logs(self) -> Optional[pd.DataFrame]:
        raise NotImplementedError(
            "The well_log property is required to be implemented in a FromSource class."
        )
