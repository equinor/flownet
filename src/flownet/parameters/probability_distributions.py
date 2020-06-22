"""Defines the different probability distributions. These are used by the
parameter class definitions.
"""

import abc

from hyperopt import hp


class ProbabilityDistribution(abc.ABC):
    @property
    @abc.abstractmethod
    def ert_gen_kw(self):
        """A string representing what ERT needs in GEN_KW"""

    @abc.abstractmethod
    def hyperopt_distribution(self, label: str):
        """A hyperopt distributio, given the name of the parameter"""


class UniformDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "UNFIORM MIN MAX" distribution keyword for use in GEN_KW"""
        return f"UNIFORM {self.minimum} {self.maximum}"

    def hyperopt_distribution(self, label: str) -> hp.uniform:
        """
        Hyperopt distribution for a "UNFIORM MIN MAX" distribution

        Args:
            label: name of the paramters

        Returns:
            Hyperopt uniform object
        """
        return hp.uniform(label, self.minimum, self.maximum)


class LogUniformDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGUNIF MIN MAX" distribution keyword for use in GEN_KW"""
        return f"LOGUNIF {self.minimum} {self.maximum}"

    def hyperopt_distribution(self, label: str) -> hp.loguniform:
        """
        Hyperopt distribution for a "LOGUNIF MIN MAX" distribution

        Args:
            label: name of the paramters

        Returns:
            Hyperopt uniform object
        """
        return hp.loguniform(label, self.minimum, self.maximum)


class TriangularDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum, mode):
        self.minimum = minimum
        self.maximum = maximum
        self.mode = mode

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRIANGULAR MIN MAX" distribution keyword for use in GEN_KW"""
        return f"TRIANGULAR {self.minimum} {self.mode} {self.maximum}"

    def hyperopt_distribution(self, label: str):
        """
        Not implemented.
        """
        raise NotImplementedError(
            "The triangular distribution is currently not supported for hyperopt runs."
        )
