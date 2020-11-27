"""Defines the different probability distributions. These are used by the
parameter class definitions.
"""

import abc
from typing import Union
import numpy as np
from scipy.optimize import minimize


class ProbabilityDistribution(abc.ABC):
    def __init__(self, name):
        self.name: str = name
        self.minimum: float
        self.maximum: float
        self.mean: float
        self.mode: Union[float, None]
        self.stddev: float

    @property
    @abc.abstractmethod
    def ert_gen_kw(self):
        """A string representing what ERT needs in GEN_KW"""

    @abc.abstractmethod
    def update_distribution(
        self,
        **kwargs,
    ):
        """Function to update parameters for the various distributions"""


class UniformDistribution(ProbabilityDistribution):
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
        **kwargs,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            **kwargs: Arbitrary keyword arguments

            Keyword arguments (of which two (and only two)) needs to be defined/different from None:
                minimum: The minimum values of the updated distribution
                mean: The mean value of the updated distribution
                maximum: The maximum value of the updated distribution

        Returns:
            Nothing
        """
        assert (
            sum(
                value is not None
                for value in list(
                    map(kwargs.get, {"minimum", "maximum", "mean"}.intersection(kwargs))
                )
            )
            == 2
        ), "Min/mean, min/max or mean/max needs to be defined"

        if "mean" not in kwargs or kwargs.get("mean") is None:
            self.minimum = kwargs.get("minimum")
            self.maximum = kwargs.get("maximum")
            self.mean = (self.minimum + self.maximum) / 2
        if "minimum" not in kwargs or kwargs.get("minimum") is None:
            self.mean = kwargs.get("mean")
            self.maximum = kwargs.get("maximum")
            self.minimum = self.mean - (self.maximum - self.mean)
        if "maximum" not in kwargs or kwargs.get("maximum") is None:
            self.mean = kwargs.get("mean")
            self.minimum = kwargs.get("minimum")
            self.maximum = self.mean + (self.mean - self.minimum)

        self.mode = None
        self.stddev = np.sqrt(np.power(self.maximum - self.minimum, 2) / 12)


class LogUniformDistribution(ProbabilityDistribution):
    def __init__(
        self,
        minimum: Union[float, None] = None,
        maximum: Union[float, None] = None,
        mean: Union[float, None] = None,
    ):
        super().__init__("LOGUNIF")
        self.update_distribution(minimum=minimum, mean=mean, maximum=maximum)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGUNIF MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.maximum}"

    def update_distribution(
        self,
        **kwargs,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            **kwargs: Arbitrary keyword arguments

            Keyword arguments (of which two (and only two)) needs to be defined/different from None:
                minimum: The minimum values of the updated distribution
                mean: The mean value of the updated distribution
                maximum: The maximum value of the updated distribution

        Returns:
            Nothing
        """
        assert (
            sum(
                value is not None
                for value in list(
                    map(kwargs.get, {"minimum", "maximum", "mean"}.intersection(kwargs))
                )
            )
            == 2
        ), "Min/mean, min/max or mean/max needs to be defined"

        if "mean" not in kwargs or kwargs.get("mean") is None:
            self.minimum = kwargs.get("minimum")
            self.maximum = kwargs.get("maximum")
            self.mean = (self.maximum - self.minimum) / np.log(
                self.maximum / self.minimum
            )
        if "minimum" not in kwargs or kwargs.get("minimum") is None:
            self.mean = kwargs.get("mean")
            self.maximum = kwargs.get("maximum")
            self._find_dist_minmax(
                mean_val=self.mean, min_val=None, max_val=self.maximum
            )
        if "maximum" not in kwargs or kwargs.get("maximum") is None:
            self.mean = kwargs.get("mean")
            self.minimum = kwargs.get("minimum")
            self._find_dist_minmax(
                min_val=self.minimum, mean_val=self.mean, max_val=None
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
    ):
        """
        Find the distribution min or max for a loguniform distribution, assuming only
        one of these and the mean are given

        Args:
            min_val: minimum value for the distribution
            max_val: maximum value for the distribution
            mean_val: mean value for the distribution

        Returns:
            Nothing

        """
        # pylint: disable=cell-var-from-loop
        if min_val is None:
            self.minimum = minimize(
                lambda x: (mean_val - ((max_val - x) / np.log(max_val / x))) ** 2,
                x0=mean_val,
                tol=1e-9,
                method="L-BFGS-B",
                bounds=[(1e-9, mean_val)],
            ).x[0]
        if max_val is None:
            self.maximum = minimize(
                lambda x: (mean_val - ((x - min_val) / np.log(x / min_val))) ** 2,
                x0=mean_val,
                tol=1e-9,
                method="L-BFGS-B",
                bounds=[(mean_val, None)],
            ).x[0]


class TriangularDistribution(ProbabilityDistribution):
    def __init__(
        self,
        minimum: Union[float, None] = None,
        maximum: Union[float, None] = None,
        mean: Union[float, None] = None,
        mode: Union[float, None] = None,
    ):
        super().__init__("TRIANGULAR")
        self.update_distribution(minimum=minimum, maximum=maximum, mean=mean, mode=mode)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "TRIANGULAR MIN MAX" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.minimum} {self.mode} {self.maximum}"

    def update_distribution(
        self,
        **kwargs,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            **kwargs: Arbitrary keyword arguments

            Keyword arguments (of which three (and only three)) needs to be defined/different from None:
                minimum: The minimum values of the updated distribution
                mean: The mean value of the updated distribution
                mode: The mode of the updated distribution
                maximum: The maximum value of the updated distribution

        Returns:
            Nothing
        """
        assert (
            sum(
                value is not None
                for value in list(
                    map(
                        kwargs.get,
                        {"minimum", "maximum", "mean", "mode"}.intersection(kwargs),
                    )
                )
            )
            == 3
        ), "Triangular distributions needs three parameters to be defined"
        assert (
            {"minimum", "mean", "maximum"}.issubset(kwargs)
            or {"minimum", "mode", "maximum"}.issubset(kwargs)
            or {"mode", "maximum", "mean"}.issubset(kwargs)
            or {"mode", "minimum", "mean"}.issubset(kwargs)
        )

        self.maximum = (
            kwargs.get("maximum")
            if "maximum" in kwargs
            else 3 * kwargs.get("mean") - kwargs.get("mode") - kwargs.get("minimum")
        )
        self.minimum = (
            kwargs.get("minimum")
            if "minimum" in kwargs
            else 3 * kwargs.get("mean") - kwargs.get("mode") - kwargs.get("maximum")
        )
        self.mean = (
            kwargs.get("mean")
            if "mean" in kwargs
            else (kwargs.get("mode") + kwargs.get("minimum") + kwargs.get("maximum"))
            / 3
        )
        self.mode = (
            kwargs.get("mode")
            if "mode" in kwargs
            else 3 * kwargs.get("mean") - kwargs.get("mode") - kwargs.get("minimum")
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
        self.update_distribution(mean=mean, stddev=stddev)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "NORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev}"

    def update_distribution(
        self,
        **kwargs,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            **kwargs: Arbitrary keyword arguments

            Keyword arguments:
                mean: The mean value of the updated distribution
                stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        if "mean" in kwargs:
            self.mean = kwargs.get("mean")
            self.mode = self.mean
        if "stddev" in kwargs:
            self.stddev = kwargs.get("stddev")

        self.minimum = self.mean - 3 * self.stddev
        self.maximum = self.mean + 3 * self.stddev


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
        **kwargs,
    ):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            **kwargs: Arbitrary keyword arguments

            Keyword arguments:
                mean: The mean value of the updated distribution
                stddev: The standard deviation of the updated distribution
                minimum: The minimum/lower truncation value of the updated distribution
                maximum: The maximum/upper truncation value of the updated distribution

        Returns:
            Nothing
        """
        for item in {"mean", "stddev", "minimum", "maximum"}:
            if item in kwargs:
                setattr(self, item, kwargs.get(item))
        self.mode = self.mean


class LogNormalDistribution(ProbabilityDistribution):
    def __init__(self, mean, stddev):
        super().__init__("LOGNORMAL")
        self.update_distribution(mean=mean, stddev=stddev)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "LOGNORMAL MEAN STDDEV" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean} {self.stddev}"

    def update_distribution(self, **kwargs):
        """
        Function that updates the parameters that defines the probability distribution

        Args:
            **kwargs: Arbitrary keyword arguments

            Keyword arguments:
                mean: The mean value of the updated distribution
                stddev: The standard deviation of the updated distribution

        Returns:
            Nothing
        """
        for item in {"mean", "stddev"}:
            if item in kwargs:
                setattr(self, item, kwargs.get(item))


class Constant(ProbabilityDistribution):
    def __init__(self, constant):
        super().__init__("CONST")
        self.update_distribution(constant=constant)

    @property
    def ert_gen_kw(self) -> str:
        """string representing an ERT "CONST CONSTANT" distribution keyword for use in GEN_KW"""
        return f"{self.name} {self.mean}"

    def update_distribution(self, **kwargs):
        """

        Args:
            **kwargs:

        Returns:

        """
        if "constant" in kwargs:
            self.mean = kwargs.get("constant")
            self.minimum = kwargs.get("constant")
            self.maximum = kwargs.get("constant")
            self.mode = kwargs.get("constant")
