from typing import Dict, List

import jinja2

from .probability_distributions import UniformDistribution, ProbabilityDistribution
from ._base_parameter import Parameter


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)


class RockCompressibility(Parameter):
    """
    Parameter type which takes care of stochastically drawn rock compressibility parameters.
    The rock compressibility will always have a Uniform Distribution between the min and
    max values.

    Args:
        reference_pressure: reference pressure for rock compressibility definition
        min_compressibility: minimum rock compressibility
        max_compressibility: maxmimum rock compressibility

    """

    def __init__(
        self,
        reference_pressure: float,
        min_compressibility: float,
        max_compressibility: float,
    ):
        self._reference_pressure: float = reference_pressure
        self._random_variables: List[ProbabilityDistribution] = [
            UniformDistribution(min_compressibility, max_compressibility)
        ]

    def render_output(self) -> Dict:
        """
        Creates ROCK include content - which are given to the PROPS section.

        Returns:
            ROCK include content

        """
        return {
            "PROPS": _TEMPLATE_ENVIRONMENT.get_template("ROCK.jinja2").render(
                {
                    "reference_pressure": self._reference_pressure,
                    "compressibility": self.random_samples[0],
                }
            )
        }
