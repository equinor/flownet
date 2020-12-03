"""Defines the different probability distributions. These are used by the
parameter class definitions.
"""

import abc
from typing import Optional
import numpy as np
from scipy.optimize import minimize


class ProbabilityDistribution(abc.ABC):
    def __init__(self, name):
        self.name: str = name
        self.mean: float
        self.stddev: float
        # not all distributions have a mode, minimum or maximum
        self.mode: Optional[float]
        self.minimum: Optional[float]
        self.maximum: Optional[float]

    @property
    @abc.abstractmethod
    def ert_gen_kw(self):
        """A string representing what ERT needs in GEN_KW"""

    @abc.abstractmethod
    def update_distribution(
        self,
        minimum: Optional[float],
        maximum: Optional[float],
        mean: Optional[float],
        mode: Optional[float],
        stddev: Optional[float],
    ):
        """Function to update parameters for the various distributions"""


class UniformDistribution(ProbabilityDistribution):
    """
    The UniformDistribution class

    The class is initialized by providing ONLY two of the following inputs:
        * The minimum value of the uniform distribution
        * The mean value of the uniform distribution
        * The maximum value of the uniform distribution

    Args:
        minimum (float): The minimum value of the distribution
        mean (float): The mean value of the distribution
        mode (float): The mode of the distribution
        maximum (float): The maximum value of the distribution
        stddev (float): The standard deviation of the distribution

    
    """
    def __init__(
        self,
        minimum: float = None,
        maximum: float = None,
        mean: float = None,
    ):
        super().__init__("UNIFORM")
        self.update_distribution(minimum=minimum, mean=mean, maximum=maximum)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "UNIFORM MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.maximum}"

    def update_distribution(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        mode: Optional[float] = None,
        stddev: Optional[float] = None,
    ):
        """
        Function that updates the parameters that defines the probability distribution.
        
        The following input combinations will make changes to the distribution:
            * Giving a new minimum value as input will change the minimum value, and a new mean and stddev will be calculated
            * Giving a new maximum value as input will change the maximum value, and a new mean and stddev will be calculated
            * Giving a new mean value as input requires a new minimum OR maximum value to be defined also
                - A new mean value and a new minimum value will trigged an update of the maximum value and the stddev
                - A new mean value and a new maximum value will trigged an update of the minimum value and the stddev

        Providing values for stddev or mode has no effect here, since the uniform distribution has no mode, and the stddev is caluculated from the minimum and maximum values
        
        Providing a new mean, a new minimum and a new maximum value means will trigger an error


        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if not mean and not maximum and minimum:
            self.minimum = minimum
            self.mean = (self.minimum + self.maximum) / 2
        elif not mean and not minimum and maximum:
            self.maximum = maximum
            self.mean = (self.minimum + self.maximum) / 2
        elif not mean and minimum and maximum:
            self.minimum = minimum
            self.maximum = maximum
            self.mean = (self.minimum + self.maximum) / 2
        elif not minimum and mean and maximum:
            self.mean = mean
            self.maximum = maximum
            self.minimum = self.mean - (self.maximum - self.mean)
        elif not maximum and mean and minimum:
            self.mean = mean
            self.minimum = minimum
            self.maximum = self.mean + (self.mean - self.minimum)
        else:
            raise ValueError(
                "Uniform distribution not properly defined."
                "Minimum/mean, minimum/maximum or mean/maximum needs to be defined, "
                "but not all of minimum, mean, maximum at the same time"
            )
        self.mode = None
        self.stddev = np.sqrt(np.power(self.maximum - self.minimum, 2) / 12)


class LogUniformDistribution(ProbabilityDistribution):
    def __init__(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
    ):
        super().__init__("LOGUNIF")
        self.update_distribution(minimum=minimum, mean=mean, maximum=maximum)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGUNIF MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.maximum}"

    def update_distribution(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        mode: Optional[float] = None,
        stddev: Optional[float] = None,
    ):
        """
        Function that updates the parameters that defines the probability distribution
        
        The following input combinations will make changes to the distribution:
            * Giving a new minimum value as input will change the minimum value, and a new mean and stddev will be calculated
            * Giving a new maximum value as input will change the maximum value, and a new mean and stddev will be calculated
            * Giving a new mean value as input requires a new minimum OR maximum value to be defined also
                - A new mean value and a new minimum value will trigged an update of the maximum value and the stddev
                - A new mean value and a new maximum value will trigged an update of the minimum value and the stddev

        Providing values for stddev or mode has no effect here, since in the loguniform distribution the mode is equal to the minimum value, and the stddev is caluculated from the minimum and maximum values
        
        Providing a new mean, a new minimum and a new maximum value means will trigger an error

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if not mean and not maximum and minimum:
            self.minimum = minimum
            self.mean = (self.maximum - self.minimum) / np.log(
                self.maximum / self.minimum
            )
        elif not mean and not minimum and maximum:
            self.maximum = maximum
            self.mean = (self.maximum - self.minimum) / np.log(
                self.maximum / self.minimum
            )
        elif not mean and minimum and maximum:
            self.minimum = minimum
            self.maximum = maximum
            self.mean = (self.maximum - self.minimum) / np.log(
                self.maximum / self.minimum
            )
        elif not minimum and mean and maximum:
            self.mean = mean
            self.maximum = maximum
            self.minimum = self._find_dist_minmax(
                mean_val=self.mean, min_val=None, max_val=self.maximum
            )
        elif not maximum and mean and minimum:
            self.mean = mean
            self.minimum = minimum
            self.maximum = self._find_dist_minmax(
                min_val=self.minimum, mean_val=self.mean, max_val=None
            )
        else:
            raise ValueError(
                "log-Uniform distribution not properly defined."
                "Minimum/mean, minimum/maximum or mean/maximum needs to be defined, "
                "but not all of minimum, mean, maximum at the same time"
            )

        self.mode = self.minimum
        self.stddev = np.sqrt(
            (
                np.log(self.maximum / self.minimum)
                * (np.power(self.maximum, 2) - np.power(self.minimum, 2))
                - 2 * np.power(self.maximum - self.minimum, 2)
            )
            / (2 * np.power(np.log(self.maximum / self.minimum), 2))
        )

    def _find_dist_minmax(
        self,
        mean_val: float,
        min_val: float = None,
        max_val: float = None,
    ) -> float:
        # pylint: disable=no-self-use
        """
        Find the distribution min or max for a loguniform distribution, assuming only
        one of these and the mean are given

        Args:
            min_val: minimum value for the distribution
            max_val: maximum value for the distribution
            mean_val: mean value for the distribution

        Returns:
            missing value (minimum if maximum is provided as input, maximum if minimum is provided)

        """
        # pylint: disable=cell-var-from-loop
        if min_val is None:
            result = minimize(
                lambda x: (mean_val - ((max_val - x) / np.log(max_val / x))) ** 2,
                x0=mean_val,
                tol=1e-9,
                method="L-BFGS-B",
                bounds=[(1e-9, mean_val)],
            ).x[0]
        if max_val is None:
            result = minimize(
                lambda x: (mean_val - ((x - min_val) / np.log(x / min_val))) ** 2,
                x0=mean_val,
                tol=1e-9,
                method="L-BFGS-B",
                bounds=[(mean_val, None)],
            ).x[0]
        return result


class TriangularDistribution(ProbabilityDistribution):
    def __init__(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        mode: Optional[float] = None,
    ):
        super().__init__("TRIANGULAR")
        self.update_distribution(minimum=minimum, maximum=maximum, mean=mean, mode=mode)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRIANGULAR MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.mode} {self.maximum}"

    def update_distribution(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        mode: Optional[float] = None,
        stddev: Optional[float] = None,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if not maximum and minimum and mean and mode:
            self.minimum = minimum
            self.mode = mode
            self.mean = mean
            self.maximum = 3 * mean - mode - minimum
        elif not minimum and mean and mode and maximum:
            self.maximum = maximum
            self.mode = mode
            self.mean = mean
            self.minimum = 3 * mean - mode - maximum
        elif not mode and mean and minimum and maximum:
            self.mean = mean
            self.minimum = minimum
            self.maximum = maximum
            self.mode = 3 * mean - maximum - minimum
        elif not mean and minimum and mode and maximum:
            self.mode = mode
            self.maximum = maximum
            self.minimum = minimum
            self.mean = mode + minimum + maximum / 3
        else:
            raise ValueError(
                "Triangular distribution not properly defined."
                "Three (and only three) of minimum, mode, mean and maximum needs to be defined."
            )

        self.stddev = (
            np.power(self.minimum, 2)
            + np.power(self.mode, 2)
            + np.power(self.maximum, 2)
            - (self.minimum * self.mode)
            - (self.minimum * self.maximum)
            - (self.mode * self.maximum)
        ) / 18


class NormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        super().__init__("NORMAL")
        self.minimum = None
        self.maximum = None
        self.update_distribution(mean=mean, stddev=stddev)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev}"

    def update_distribution(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        mode: Optional[float] = None,
        stddev: Optional[float] = None,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mean:
            self.mean = mean
            self.mode = self.mean
        if stddev:
            self.stddev = stddev


class TruncatedNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean: float, stddev, minimum, maximum):
        super().__init__("TRUNCATED_NORMAL")
        self.update_distribution(
            minimum=minimum, maximum=maximum, mean=mean, stddev=stddev
        )

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRUNCATED_NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev} {self.minimum} {self.maximum}"

    def update_distribution(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        mode: Optional[float] = None,
        stddev: Optional[float] = None,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mean:
            self.mean = mean
            self.mode = mean
        if stddev:
            self.stddev = stddev
        if minimum:
            self.minimum = minimum
        if maximum:
            self.maximum = maximum


class LogNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        super().__init__("LOGNORMAL")
        self.minimum = None
        self.maximum = None
        self.update_distribution(mean=mean, stddev=stddev)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGNORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev}"

    def update_distribution(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        mode: Optional[float] = None,
        stddev: Optional[float] = None,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mean:
            self.mean = mean
        if stddev:
            self.stddev = stddev


class Constant(ProbabilityDistribution):
    def __init__(self, constant):
        super().__init__("CONST")
        self.stddev = 0
        self.update_distribution(mode=constant)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "CONST CONSTANT" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean}"

    def update_distribution(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        mean: Optional[float] = None,
        mode: Optional[float] = None,
        stddev: Optional[float] = None,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mode:
            self.mode = mode
            self.mean = mode
            self.maximum = mode
            self.minimum = mode
