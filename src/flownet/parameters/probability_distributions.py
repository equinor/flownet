"""Defines the different probability distributions. These are used by the
parameter class definitions.
"""

import abc
import numpy as np


class ProbabilityDistribution(abc.ABC):
    @property
    @abc.abstractmethod
    def ert_gen_kw(self):
        """A string representing what ERT needs in GEN_KW"""


class UniformDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        self.mean = (minimum + maximum) / 2
        self.stddev = np.sqrt(np.power(maximum - minimum, 2) / 12)
        self.name = "UNIFORM"

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "UNIFORM MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.maximum}"


class LogUniformDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        self.mean = (maximum - minimum) / np.log(maximum / minimum)
        self.stddev = np.sqrt(
            (
                np.log(maximum / minimum)
                * (np.power(maximum, 2) - np.power(minimum, 2))
                - 2 * np.power(maximum - minimum, 2)
            )
            / (2 * np.power(np.log(maximum / minimum), 2))
        )
        self.name = "LOGUNIF"

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGUNIF MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.maximum}"


class TriangularDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum, mode):
        self.mode = mode
        self.minimum = minimum
        self.maximum = maximum
        self.mean = (mode + minimum + maximum) / 3
        self.stddev = (
            np.power(minimum, 2)
            + np.power(mode, 2)
            + np.power(maximum, 2)
            - (minimum * mode)
            - (minimum * maximum)
            - (mode * maximum)
        ) / 18
        self.name = "TRIANGULAR"

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRIANGULAR MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.mode} {self.maximum}"


class NormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
        self.minimum = None
        self.maximum = None
        self.name = "NORMAL"

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev}"


class TruncatedNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev, minimum, maximum):
        self.mean = mean
        self.stddev = stddev
        self.minimum = minimum
        self.maximum = maximum
        self.name = "TRUNCATED_NORMAL"

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRUNCATED_NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev} {self.minimum} {self.maximum}"


class LogNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
        self.minimum = None
        self.maximum = None
        self.name = "LOGNORMAL"

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGNORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev}"


class Constant(ProbabilityDistribution):
    def __init__(self, constant):
        self.mean = constant
        self.minimum = None
        self.maximum = None
        self.stddev = None
        self.name = "CONST"

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "CONST CONSTANT" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean}"
