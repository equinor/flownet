import abc
from typing import List, Dict, Union, Optional

import jinja2
import pandas as pd

from .probability_distributions import (
    ProbabilityDistribution,
    UniformDistribution,
    LogUniformDistribution,
    NormalDistribution,
    LogNormalDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    Constant,
)


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)


def parameter_probability_distribution_class(
    row: pd.Series, param: Optional[str] = None
) -> ProbabilityDistribution:
    """

    Args:
        row (pd.Series): Information used to initialize a ProbabilityDistribution class. Different probability
            distributions require different amount of parameters to be defined.
            The columns should be:
                * 'minimum': The minimum value of the distribution
                * 'mean': The mean value of the distribution
                * 'mode': The mode of the distribution
                * 'maximum': The maximum value of the distribution
                * 'stddev': The standard deviation of the distribution
                * 'distribution': The type of probability distribution that shold be initialized
        param (str): The name of the parameter if the column names in 'row' contains the name
            (e.g. if the column name is 'minimum_bulk_volume', param should be 'bulk_volume')

    Returns:
        ProbabilityDistribution class
    """
    # pylint: disable=too-many-return-statements
    param = "_" + param if param is not None else ""
    if row[f"distribution{param}"] == "uniform":
        return UniformDistribution(
            minimum=row[f"minimum{param}"],
            mean=row[f"mean{param}"],
            maximum=row[f"maximum{param}"],
        )
    if row[f"distribution{param}"] == "logunif":
        return LogUniformDistribution(
            minimum=row[f"minimum{param}"],
            mean=row[f"mean{param}"],
            maximum=row[f"maximum{param}"],
        )
    if row[f"distribution{param}"] == "normal":
        return NormalDistribution(
            mean=row[f"mean{param}"], stddev=row[f"stddev{param}"]
        )
    if row[f"distribution{param}"] == "lognormal":
        return LogNormalDistribution(
            mean=row[f"mean{param}"], stddev=row[f"stddev{param}"]
        )
    if row[f"distribution{param}"] == "truncated_normal":
        return TruncatedNormalDistribution(
            mean=row[f"mean{param}"],
            stddev=row[f"stddev{param}"],
            minimum=row[f"minimum{param}"],
            maximum=row[f"maximum{param}"],
        )
    if row[f"distribution{param}"] == "triangular":
        return TriangularDistribution(
            minimum=row[f"minimum{param}"],
            mean=row[f"mean{param}"],
            mode=row[f"base{param}"],
            maximum=row[f"maximum{param}"],
        )
    if row[f"distribution{param}"] == "const":
        return Constant(const=row[f"base{param}"])
    raise ValueError("Unknown probability distribution class.")


class Parameter(abc.ABC):
    """The abstract base class for any higher order parameter (permeability,
    relative permeability, contact depths, PVT etc.)

    Every class inheriting this abstract base class needs to define an attribute
    `_random_variables` and implement the class method `render_output`.

    """

    @property
    def random_variables(self) -> List[ProbabilityDistribution]:
        """List of all Probability distributions for the Parameter"""
        # pylint: disable=no-member
        return self._random_variables  # type: ignore[attr-defined]

    @property
    def random_samples(self) -> List[float]:
        """List of all random values for each distribution for the Parameter"""
        return self._random_samples

    @random_samples.setter
    def random_samples(self, values: List[float]):
        """Setter for the Parameter random samples."""
        if len(values) != len(self.random_variables):
            raise ValueError(
                f"Length of drawn random samples ({len(values)}) "
                f"differs from number of random variables ({len(self.random_variables)})."
            )
        self._random_samples = values

    @property
    def mean_values(self) -> List[float]:
        """List of all mean values for each distribution for the Parameter"""
        return self._mean_values

    @mean_values.setter
    def mean_values(self, values: List[float]):
        """Setter for the Parameter mean samples."""
        self._mean_values = values

    @property
    def names(self) -> List[str]:
        """List of all names for each distribution for the Parameter"""
        return self._names

    @names.setter
    def names(self, values: List[str]):
        """Setter for the Parameter names samples."""
        self._names = values

    @property
    def stddev_values(self) -> List[float]:
        """List of all standard deviation values for each distribution for the Parameter"""
        return self._stddev_values

    @stddev_values.setter
    def stddev_values(self, values: List[float]):
        """Setter for the Parameter standard deviation samples."""
        self._stddev_values = values

    # pylint: disable=no-self-use
    def get_dims(self) -> Union[None, Dict[str, int]]:
        """In case a parameter requires updates in runspec dimensions, a get_dims
        function will need to be implemented.
        """
        return None

    @abc.abstractmethod
    def render_output(self) -> Dict:
        """Returns a dictionary which defines content that should be included
        in the simulation model. The keys are the different sections on where to
        append the corresponding dictionary value.

        E.g:
            {"GRID": 'PERMX\n 0.1......", "SOLUTION": "EQUIL\n ..."}
        """
