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
        self.mean: float = None
        self.stddev: float = None
        self.mode: float = None
        self.minimum: float = None
        self.maximum: float = None
        self.defined: bool = False

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

    The class is initialized by providing ONLY two of the following inputs different from None:
        * The minimum value of the uniform distribution
        * The mean value of the uniform distribution
        * The maximum value of the uniform distribution

    Args:
        minimum (float): The minimum value of the distribution
        mean (float): The mean value of the distribution
        maximum (float): The maximum value of the distribution

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
            * Giving a new minimum value as input will trigger calculation of a new mean and standard deviation
            * Giving a new maximum value as input will trigger calculation of a new mean and standard deviation
            * Giving a new mean value as input requires a new minimum OR maximum value to be defined also
                - A new mean value and a new minimum value will trigged an update of the maximum value and the stddev
                - A new mean value and a new maximum value will trigged an update of the minimum value and the stddev

        Providing values for stddev or mode has no effect here, since the uniform distribution has no mode, and the
        stddev is calculated from the minimum and maximum values

        Providing a new mean, a new minimum and a new maximum value (all three of them) will trigger an error


        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mean is None and maximum is None and minimum is not None and self.defined:
            # not possible on first call of function (initialization) since the full distribution must be defined then
            self.minimum = minimum
            self.mean = (self.minimum + self.maximum) / 2
        elif mean is None and minimum is None and maximum is not None and self.defined:
            # not possible on first call of function (initialization) since the full distribution must be defined then
            self.maximum = maximum
            self.mean = (self.minimum + self.maximum) / 2
        elif mean is None and minimum is not None and maximum is not None:
            self.minimum = minimum
            self.maximum = maximum
            self.mean = (self.minimum + self.maximum) / 2
        elif minimum is None and mean is not None and maximum is not None:
            self.mean = mean
            self.maximum = maximum
            self.minimum = self.mean - (self.maximum - self.mean)
        elif maximum is None and mean is not None and minimum is not None:
            self.mean = mean
            self.minimum = minimum
            self.maximum = self.mean + (self.mean - self.minimum)
        elif mean is not None:
            raise ValueError(
                "It is not possible to update the mean of the uniform distribution without "
                "providing either a new minimum value or a new maximum value at the same time."
            )
        elif mode is not None:
            raise ValueError(
                "The mode in a uniform distribution is either all possible values or non-existing. "
                "You can choose yourself, but don't try to update it!"
            )
        elif stddev is not None:
            raise ValueError(
                "It is currently not possible to update the uniform distribution using the standard deviation"
            )
        else:
            raise ValueError(
                "Uniform distribution not properly defined."
                "Minimum/mean, minimum/maximum or mean/maximum needs to be defined, "
                "but not all of minimum, mean, maximum at the same time"
            )
        self.stddev = np.sqrt(np.power(self.maximum - self.minimum, 2) / 12)
        self.defined = True


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
            * Giving a new minimum value as input will trigger a calculation of a new mean and stddev
            * Giving a new maximum value as input will trigger a calculation of a new mean and stddev
            * Giving a new mean value as input requires a new minimum OR maximum value to be defined also
                - A new mean value and a new minimum value will trigger an update of the maximum value and the stddev
                - A new mean value and a new maximum value will trigger an update of the minimum value and the stddev

        Providing values for stddev or mode has no effect here, since in the loguniform distribution the mode is
        equal to the minimum value, and the stddev is caluculated from the minimum and maximum values

        Providing a new mean, a new minimum and a new maximum value (all three of them) will trigger an error

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mean is None and maximum is None and minimum is not None and self.defined:
            # not possible on first call of function since the full distribution must be defined then
            self.minimum = minimum
            self.mean = (self.maximum - self.minimum) / np.log(
                self.maximum / self.minimum
            )
        elif mean is None and minimum is None and maximum is not None and self.defined:
            # not possible on first call of function since the full distribution must be defined then
            self.maximum = maximum
            self.mean = (self.maximum - self.minimum) / np.log(
                self.maximum / self.minimum
            )
        elif mean is None and minimum is not None and maximum is not None:
            self.minimum = minimum
            self.maximum = maximum
            self.mean = (self.maximum - self.minimum) / np.log(
                self.maximum / self.minimum
            )
        elif minimum is None and mean is not None and maximum is not None:
            self.mean = mean
            self.maximum = maximum
            self.minimum = self._find_dist_minmax(
                mean_val=self.mean, min_val=None, max_val=self.maximum
            )
        elif maximum is None and mean is not None and minimum is not None:
            self.mean = mean
            self.minimum = minimum
            self.maximum = self._find_dist_minmax(
                min_val=self.minimum, mean_val=self.mean, max_val=None
            )
        elif mean is not None:
            raise ValueError(
                "It is not possible to update the mean of the loguniform distribution without "
                "providing either a new minimum value or a new maximum value at the same time."
            )
        elif mode is not None or stddev is not None:
            raise ValueError(
                "It is currently not possible to update the loguniform distribution "
                "using the standard deviation or the mode"
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
        self.defined = True

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
        """string representing an ERT "TRIANGULAR MIN MODE MAX" distribution keyword for use in GEN_KW"""
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

        The following input combinations will make changes to the distribution:
            * Giving a new minimum value as input will trigger a calculation of a new mean and stddev
            * Giving a new maximum value as input will trigger a calculation of a new mean and stddev
            * Giving a new mode as input will trigger a calculation of a new mean and stddev
            * Giving a new mean value as input requires a new minimum/maximum OR mode/maximum OR minimum/mode values
            to be defined also:
                - A new mean, minimum and mode will trigger an update of the maximum value and the stddev
                - A new mean, mode and maximum value will trigger an update of the minimum value and the stddev
                - A new mean, minimum and maximum value will trigger an update of the mode and the stddev

        Providing values for stddev has no effect here.

        Providing a new mean, minimum, maximum and mode (all four of them) will trigger an error

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if (
            maximum is not None
            and minimum is None
            and mean is None
            and mode is None
            and self.defined
        ):
            self.maximum = maximum
            self.mean = (self.mode + self.minimum + self.maximum) / 3
        elif (
            minimum is not None
            and maximum is None
            and mean is None
            and mode is None
            and self.defined
        ):
            self.minimum = minimum
            self.mean = (self.mode + self.minimum + self.maximum) / 3
        elif (
            mode is not None
            and maximum is None
            and mean is None
            and minimum is None
            and self.defined
        ):
            self.mode = mode
            self.mean = (self.mode + self.minimum + self.maximum) / 3
        elif (
            maximum is None
            and minimum is not None
            and mean is not None
            and mode is not None
        ):
            self.minimum = minimum
            self.mode = mode
            self.mean = mean
            self.maximum = 3 * mean - mode - minimum
        elif (
            minimum is None
            and maximum is not None
            and mean is not None
            and mode is not None
        ):
            self.maximum = maximum
            self.mode = mode
            self.mean = mean
            self.minimum = 3 * mean - mode - maximum
        elif (
            mode is None
            and maximum is not None
            and mean is not None
            and minimum is not None
        ):
            self.mean = mean
            self.minimum = minimum
            self.maximum = maximum
            self.mode = 3 * mean - maximum - minimum
        elif (
            mean is None
            and maximum is not None
            and minimum is not None
            and mode is not None
        ):
            self.mode = mode
            self.maximum = maximum
            self.minimum = minimum
            self.mean = (mode + minimum + maximum) / 3
        elif mean is not None:
            raise ValueError(
                "It is not possible to update the mean of the triangular distribution without "
                "providing two of the following: minimum, mode or maximum."
            )
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
        self.defined = True


class NormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        super().__init__("NORMAL")
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
        Function that updates the parameters that defines the probability distribution.

        In the normal distribution one or both of the mean and the standard deviation can be changed.
        Providing any other value as input here (mode, minimum, maximum) will have no effect.

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mean is not None:
            self.mean = mean
            self.mode = self.mean
        if stddev is not None:
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

        In the truncated normal distribution one or more of the following can be changed:
            *The mean
            *The standard deviation
            *The minimum value
            *The maximum value

        Providing a value for the mode has no effect here

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mean is not None:
            self.mean = mean
            self.mode = mean
        if stddev is not None:
            self.stddev = stddev
        if minimum is not None:
            self.minimum = minimum
        if maximum is not None:
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

        In the lognormal distribution one or both of the mean and the standard deviation can be changed.
        Providing any other value as input here (mode, minimum, maximum) will have no effect.

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if mean is not None:
            self.mean = mean
        if stddev is not None:
            self.stddev = stddev


class Constant(ProbabilityDistribution):
    def __init__(self, const):
        super().__init__("CONST")
        self.stddev = 0
        self.update_distribution(mode=const)

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

        Providing a value for either the mean, the mode, the minimum or the maximum will change the constant value,
        but ONLY one value can be provided.

        Args:
            minimum: The minimum values of the updated distribution
            mean: The mean value of the updated distribution
            mode: The mode of the updated distribution
            maximum: The maximum value of the updated distribution
            stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if sum(var is not None for var in (mode, minimum, maximum, mean)) > 1:
            raise ValueError("A constant can only be defined by one value!")
        for var in (mode, minimum, maximum, mean):
            if var is not None:
                self.mode = var
                self.mean = var
                self.maximum = var
                self.minimum = var
        self.stddev = 0
