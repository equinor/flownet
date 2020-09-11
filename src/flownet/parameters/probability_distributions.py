"""Defines the different probability distributions. These are used by the
parameter class definitions.
"""

import abc


class ProbabilityDistribution(abc.ABC):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    @property
    @abc.abstractmethod
    def ert_gen_kw(self):
        """A string representing what ERT needs in GEN_KW"""


class UniformDistribution(ProbabilityDistribution):
    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "UNFIORM MIN MAX" distribution keyword for use in GEN_KW"""
        return f"UNIFORM {self.minimum} {self.maximum}"


class LogUniformDistribution(ProbabilityDistribution):
    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGUNIF MIN MAX" distribution keyword for use in GEN_KW"""
        return f"LOGUNIF {self.minimum} {self.maximum}"


class TriangularDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum, mode):
        super().__init__(minimum, maximum)
        self.mode = mode

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRIANGULAR MIN MAX" distribution keyword for use in GEN_KW"""
        return f"TRIANGULAR {self.minimum} {self.mode} {self.maximum}"
