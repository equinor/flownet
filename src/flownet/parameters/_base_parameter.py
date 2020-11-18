import abc
from typing import List, Dict, Union

import jinja2

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
    row, param=None
) -> ProbabilityDistribution:
    """

    Args:
        row:
        param:

    Returns:

    """
    # pylint: disable=too-many-return-statements
    param = "_" + param if param is not None else ""
    if row[f"distribution{param}"] == "uniform":
        return UniformDistribution(row[f"minimum{param}"], row[f"maximum{param}"])
    if row[f"distribution{param}"] == "logunif":
        return LogUniformDistribution(row[f"minimum{param}"], row[f"maximum{param}"])
    if row[f"distribution{param}"] == "normal":
        return NormalDistribution(row[f"mean{param}"], row[f"stddev{param}"])
    if row[f"distribution{param}"] == "lognormal":
        return LogNormalDistribution(row[f"mean{param}"], row[f"stddev{param}"])
    if row[f"distribution{param}"] == "truncated_normal":
        return TruncatedNormalDistribution(
            row[f"mean{param}"],
            row[f"stddev{param}"],
            row[f"minimum{param}"],
            row[f"maximum{param}"],
        )
    if row[f"distribution{param}"] == "triangular":
        return TriangularDistribution(
            row[f"minimum{param}"], row[f"base{param}"], row[f"maximum{param}"]
        )
    if row[f"distribution{param}"] == "constant":
        return Constant(row[f"constant{param}"])
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
