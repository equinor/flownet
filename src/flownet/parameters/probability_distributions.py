"""Defines the different probability distributions. These are used by the
parameter class definitions.
"""

import abc


class ProbabilityDistribution(abc.ABC):
    @property
    @abc.abstractmethod
    def ert_gen_kw(self):
        """A string representing what ERT needs in GEN_KW"""


class UniformDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "UNIFORM MIN MAX" distribution keyword for use in GEN_KW"""
        return f"UNIFORM {self.minimum} {self.maximum}"


class LogUniformDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGUNIF MIN MAX" distribution keyword for use in GEN_KW"""
        return f"LOGUNIF {self.minimum} {self.maximum}"


class TriangularDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum, mode):
        self.mode = mode
        self.minimum = minimum
        self.maximum = maximum

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRIANGULAR MIN MAX" distribution keyword for use in GEN_KW"""
        return f"TRIANGULAR {self.minimum} {self.mode} {self.maximum}"


class NormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"NORMAL {self.mean} {self.stddev}"


class TruncatedNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev, minimum, maximum):
        self.mean = mean
        self.stddev = stddev
        self.minimum = minimum
        self.maximum = maximum

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return (
            f"TRUNCATED_NORMAL {self.mean} {self.stddev} {self.minimum} {self.maximum}"
        )


class LogNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGNORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"LOGNORMAL {self.mean} {self.stddev}"


class Constant(ProbabilityDistribution):
    def __init__(self, constant):
        self.constant = constant

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "CONST CONSTANT" distribution keyword for use in GEN_KW"""
        return f"CONST {self.constant}"
