"""Defines the different probability distributions. These are used by the
parameter class definitions.
"""

import abc
import numpy as np


class ProbabilityDistribution(abc.ABC):
    def __init__(self, minimum, maximum, mean, mode, stddev, name):
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.mode = mode
        self.stddev = stddev
        self.name = name

    @property
    @abc.abstractmethod
    def ert_gen_kw(self):
        """A string representing what ERT needs in GEN_KW"""


class UniformDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum):
        super().__init__(
            minimum,
            maximum,
            (minimum + maximum) / 2,
            None,
            np.sqrt(np.power(maximum - minimum, 2) / 12),
            "UNIFORM",
        )

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "UNIFORM MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.maximum}"


class LogUniformDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum):
        super().__init__(
            minimum,
            maximum,
            (maximum - minimum) / np.log(maximum / minimum),
            minimum,
            np.sqrt(
                (
                    np.log(maximum / minimum)
                    * (np.power(maximum, 2) - np.power(minimum, 2))
                    - 2 * np.power(maximum - minimum, 2)
                )
                / (2 * np.power(np.log(maximum / minimum), 2))
            ),
            "LOGUNIF",
        )

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGUNIF MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.maximum}"


class TriangularDistribution(ProbabilityDistribution):
    def __init__(self, minimum, maximum, mode):
        super().__init__(
            minimum,
            maximum,
            (mode + minimum + maximum) / 3,
            mode,
            (
                np.power(minimum, 2)
                + np.power(mode, 2)
                + np.power(maximum, 2)
                - (minimum * mode)
                - (minimum * maximum)
                - (mode * maximum)
            )
            / 18,
            "TRIANGULAR",
        )

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRIANGULAR MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.mode} {self.maximum}"


class NormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        super().__init__(
            None,
            None,
            mean,
            mean,
            stddev,
            "NORMAL",
        )

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev}"


class TruncatedNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev, minimum, maximum):
        super().__init__(
            minimum,
            maximum,
            mean,
            mean,
            stddev,
            "TRUNCATED_NORMAL",
        )

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRUNCATED_NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev} {self.minimum} {self.maximum}"


class LogNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        super().__init__(
            None,
            None,
            mean,
            None,
            stddev,
            "LOGNORMAL",
        )

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGNORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev}"


class Constant(ProbabilityDistribution):
    def __init__(self, constant):
        super().__init__(
            constant,
            constant,
            constant,
            constant,
            0,
            "CONST",
        )

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "CONST CONSTANT" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean}"
