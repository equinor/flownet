import os
import pathlib
from typing import Dict, Optional, List, Union

import yaml
import configsuite
from configsuite import types, MetaKeys as MK, ConfigSuite

from ..data.from_flow import FlowData


# Small workaround while waiting for https://github.com/equinor/configsuite/pull/157
# to be merged and released in upstream ConfigSuite:
def create_schema_without_arguments() -> Dict:
    return create_schema()


def create_schema(config_folder: Optional[pathlib.Path] = None) -> Dict:
    """
    Returns a configsuite type schema, where configuration value types are defined, together
    with which values are optional and/or has default values.

    Args:
        _to_abs_path: Use absolute path transformation

    Returns:
        Dictionary to be used as configsuite type schema

    """

    @configsuite.transformation_msg("Convert string to lower case")
    def _to_lower(input_data: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(input_data, str):
            return input_data.lower()

        return [x.lower() for x in input_data]

    @configsuite.transformation_msg("Convert input string to absolute path")
    def _to_abs_path(path: Optional[str]) -> str:
        """
        Helper function for the configsuite. Take in a path as a string and
        attempts to convert it to an absolute path.

        Args:
            path: A relative or absolute path or None

        Returns:
            Absolute path or empty string

        """
        if path is None:
            return ""
        if pathlib.Path(path).is_absolute():
            return str(pathlib.Path(path).resolve())
        return str((config_folder / pathlib.Path(path)).resolve())  # type: ignore

    return {
        MK.Type: types.NamedDict,
        MK.Content: {
            "name": {
                MK.Type: types.String,
                MK.Description: "Name of the FlowNet model, used e.g. in generated output report",
            },
            "flownet": {
                MK.Type: types.NamedDict,
                MK.Content: {
                    "data_source": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "simulation": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "input_case": {
                                        MK.Type: types.String,
                                        MK.Transformation: _to_abs_path,
                                        MK.Description: "Simulation input case to be used as data source for FlowNet",
                                    },
                                    "vectors": {
                                        MK.Type: types.NamedDict,
                                        MK.Description: "Which vectors to use as observation data sources",
                                        MK.Content: {
                                            "WTHP": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "rel_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                    "min_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                            "WBHP": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "rel_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                    "min_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                            "WOPR": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "rel_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                    "min_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                            "WGPR": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "rel_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                    "min_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                            "WWPR": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "rel_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                    "min_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                            "WWIR": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "rel_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                    "min_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                            "WGIR": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "rel_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                    "min_error": {
                                                        MK.Type: types.Number,
                                                        MK.Required: False,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                            "concave_hull": {MK.Type: types.Bool, MK.AllowNone: True},
                        },
                    },
                    "pvt": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "rsvd": {
                                MK.Type: types.String,
                                MK.Transformation: _to_abs_path,
                                MK.AllowNone: True,
                            },
                        },
                    },
                    "phases": {
                        MK.Type: types.List,
                        MK.Content: {MK.Item: {MK.Type: types.String}},
                        MK.Transformation: _to_lower,
                        MK.Description: "List of phases to be used in FlowNet "
                        "(valid phases are oil, gas, water, disgas, vapoil",
                    },
                    "cell_length": {MK.Type: types.Number},
                    "training_set_end_date": {
                        MK.Type: types.Date,
                        MK.AllowNone: True,
                    },
                    "training_set_fraction": {
                        MK.Type: types.Number,
                        MK.AllowNone: True,
                    },
                    "additional_flow_nodes": {
                        MK.Type: types.Integer,
                        MK.Default: 100,
                    },
                    "additional_node_candidates": {
                        MK.Type: types.Integer,
                        MK.Default: 1000,
                        MK.Description: "Number of additional nodes to create "
                        "(using Mitchell's best candidate algorithm)",
                    },
                    "hull_factor": {MK.Type: types.Number, MK.Default: 1.2},
                    "random_seed": {
                        MK.Type: types.Number,
                        MK.AllowNone: True,
                        MK.Description: "Adding this makes sure two FlowNet "
                        "runs create the exact same output",
                    },
                    "perforation_handling_strategy": {
                        MK.Type: types.String,
                        MK.Default: "bottom_point",
                    },
                    "fast_pyscal": {
                        MK.Type: types.Bool,
                        MK.Default: True,
                        MK.Description: "If True, pyscal uses some reasonable "
                        "approximation to reduce running time. It also skips "
                        "validity tests",
                    },
                    "fault_tolerance": {MK.Type: types.Number, MK.Default: 1.0e-5},
                    "max_distance": {MK.Type: types.Number, MK.Default: 1e12},
                    "max_distance_fraction": {MK.Type: types.Number, MK.Default: 0},
                },
            },
            "ert": {
                MK.Type: types.NamedDict,
                MK.Content: {
                    "runpath": {
                        MK.Type: types.String,
                        MK.Default: "output/runpath/realization-%d/iter-%d",
                    },
                    "enspath": {
                        MK.Type: types.String,
                        MK.Default: "output/storage",
                    },
                    "eclbase": {
                        MK.Type: types.String,
                        MK.Default: "./eclipse/model/FLOWNET_REALIZATION",
                    },
                    "static_include_files": {
                        MK.Type: types.String,
                        MK.Transformation: _to_abs_path,
                        MK.Default: pathlib.Path(
                            os.path.dirname(os.path.realpath(__file__))
                        )
                        / "static_include_files"
                        / ".."
                        / "static",
                    },
                    "realizations": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "num_realizations": {
                                MK.Type: types.Integer,
                                MK.Description: "Number of realizations to create",
                            },
                            "required_success_percent": {
                                MK.Type: types.Number,
                                MK.Default: 20,
                                MK.Description: "Percentage of the realizations that "
                                "need to suceed, if not the FlowNet worklow is stopped.",
                            },
                            "max_runtime": {
                                MK.Type: types.Integer,
                                MK.Default: 300,
                                MK.Description: "Maximum number of seconds allowed "
                                "for a single realization, if not that realization "
                                "is treated as a failed realization.",
                            },
                        },
                    },
                    "simulator": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "name": {
                                MK.Type: types.String,
                                MK.Transformation: _to_lower,
                                MK.Default: "flow",
                                MK.Description: "Simulator to use (typically either "
                                "'flow' or 'eclipse'",
                            },
                            "version": {
                                MK.Type: types.String,
                                MK.AllowNone: True,
                                MK.Description: "Version of the simulator to use",
                            },
                        },
                    },
                    "queue": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "system": {MK.Type: types.String},
                            "name": {MK.Type: types.String, MK.AllowNone: True},
                            "server": {MK.Type: types.String, MK.AllowNone: True},
                            "max_running": {MK.Type: types.Integer},
                        },
                    },
                    "ensemble_weights": {
                        MK.Type: types.List,
                        MK.Content: {MK.Item: {MK.Type: types.Number}},
                    },
                    "yamlobs": {
                        MK.Type: types.String,
                        MK.Default: "./observations.yamlobs",
                        MK.Description: "Observation file used by fmu-ensemble "
                        "and webviz",
                    },
                    "analysis": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "metric": {MK.Type: types.String, MK.AllowNone: True},
                            "quantity": {
                                MK.Type: types.String,
                                MK.AllowNone: True,
                            },
                            "start": {MK.Type: types.String, MK.AllowNone: True},
                            "end": {MK.Type: types.String, MK.AllowNone: True},
                            "outfile": {
                                MK.Type: types.String,
                                MK.AllowNone: True,
                            },
                        },
                    },
                },
            },
            "model_parameters": {
                MK.Type: types.NamedDict,
                MK.Content: {
                    "permeability": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of the permeability prior "
                        "distribution. You define either min and max, or one of "
                        "the endpoints and mean",
                        MK.Content: {
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "mean": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number},
                            "loguniform": {MK.Type: types.Bool, MK.Default: True},
                        },
                    },
                    "porosity": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of the porosity prior "
                        "distribution. You define either min and max, or one of "
                        "the endpoints and mean",
                        MK.Content: {
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "mean": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number},
                            "loguniform": {MK.Type: types.Bool, MK.Default: False},
                        },
                    },
                    "bulkvolume_mult": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of bulk volume multiplier "
                        "prior distribution. You define either min and max, or one "
                        "of the endpoints and mean. One multiplies is drawn per tube "
                        "which decreases/increases the default tube bulk volume "
                        "(which again is a proportional part of model convex hull bulk "
                        "with respect to tube length compared to other tubes).",
                        MK.Content: {
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "mean": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number},
                            "loguniform": {MK.Type: types.Bool, MK.Default: True},
                        },
                    },
                    "fault_mult": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of the fault multiplicator "
                        "prior distribution. You define either min and max, or one "
                        "of the endpoints and mean",
                        MK.Content: {
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "mean": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number, MK.AllowNone: True},
                            "loguniform": {MK.Type: types.Bool, MK.AllowNone: True},
                        },
                    },
                    "relative_permeability": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Add relative permeability uncertainty. "
                        "Which endpoints to give depends on phases requested in the "
                        "FlowNet model.",
                        MK.Content: {
                            "scheme": {
                                MK.Type: types.String,
                                MK.Description: "Either 'global' (one set of relative "
                                "permeability curves for the whole model), or "
                                "'individual' (one set of curves per tube).",
                                MK.Transformation: _to_lower,
                            },
                            "swirr": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "swl": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "swcr": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "sorw": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number},
                                    "max": {MK.Type: types.Number},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "krwend": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "krowend": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "nw": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "now": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "sorg": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "sgcr": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "ng": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "nog": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "krgend": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "krogend": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                        },
                    },
                    "equil": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "scheme": {
                                MK.Type: types.String,
                                MK.Description: "Number of degrees of freedom. Either "
                                "'global' (the whole model treated as one EQLNUM region),"
                                "'individual' (each tube treated as an EQLNUM region) or"
                                "'regions_from_sim' (EQLNUM regions extracted from input sinm",
                                MK.Default: "global",
                                MK.Transformation: _to_lower,
                            },
                            "eqlnum_region": {
                                MK.Type: types.List,
                                MK.Content: {
                                    MK.Item: {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "id": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "datum_depth": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "datum_pressure": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {MK.Type: types.Number},
                                                    "max": {MK.Type: types.Number},
                                                },
                                            },
                                            "owc_depth": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                            "gwc_depth": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                            "goc_depth": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "rock_compressibility": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Add uncertainty on subsurface rock "
                        "compressibility.",
                        MK.Content: {
                            "reference_pressure": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Description: "Reference pressure at which drawn "
                                "rock compressibility is to be valid at",
                            },
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number, MK.AllowNone: True},
                        },
                    },
                    "aquifer": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "scheme": {
                                MK.Type: types.String,
                                MK.AllowNone: True,
                                MK.Description: "Number of aquifers. Either 'global' "
                                "(all aquifer connections goes to the same aquifer) or "
                                "'individual' (one individual aquifer per aquifer "
                                "connection)",
                            },
                            "fraction": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Description: "Fraction of the deepest wells to be "
                                "connected to an aquifer.",
                            },
                            "delta_depth": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Description: "Vertical depth difference between "
                                "aquifer an connected well location",
                            },
                            "size_in_bulkvolumes": {
                                MK.Type: types.NamedDict,
                                MK.Description: "Description of aquifer volume prior "
                                "distribution (drawn number is multiplied with model "
                                "bulk volume, excluding aquifers, to get aquifer "
                                "volume).",
                                MK.Content: {
                                    "min": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                    },
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def parse_config(configuration_file: pathlib.Path) -> ConfigSuite.snapshot:
    """
    Takes in path to a yaml configuration file, parses it, populates with default values
    where that is defined and the has not provided his/her own value. Also error checks input
    arguments, and making sure they are of expected type.

    Args:
        configuration_file: Path to configuration file.

    Returns:
        Parsed config, where values can be extracted like e.g. 'config.ert.queue.system'.

    """
    # pylint: disable=too-many-branches, too-many-statements, too-many-lines

    input_config = yaml.safe_load(configuration_file.read_text())

    suite = ConfigSuite(
        input_config,
        create_schema(config_folder=configuration_file.parent),
        deduce_required=True,
    )

    if not suite.valid:
        raise ValueError(
            "The configuration is not valid:"
            + ", ".join([error.msg for error in suite.errors])
        )

    config = suite.snapshot

    req_relp_parameters: List[str] = []
    if (
        config.model_parameters.equil.scheme != "regions_from_sim"
        and config.model_parameters.equil.scheme != "individual"
        and config.model_parameters.equil.scheme != "global"
    ):
        raise ValueError(
            f"The relative permeability scheme "
            f"'{config.model_parameters.equil.scheme}' is not valid.\n"
            f"Valid options are 'global', 'regions_from_sim' or 'individual'."
        )
    if config.model_parameters.equil.scheme == "regions_from_sim":
        if config.flownet.data_source.simulation.input_case is None:
            raise ValueError(
                "Input simulation case is not defined - EQLNUM regions can not be extracted"
            )
        field_data = FlowData(config.flownet.data_source.simulation.input_case)
        unique_regions = field_data.get_unique_regions("EQLNUM")
        default_exists = False
        defined_regions = []
        for reg in config.model_parameters.equil.eqlnum_region:
            if reg.id is None:
                default_exists = True
            else:
                if reg.id in defined_regions:
                    raise ValueError(f"EQLNUM region {reg.id} defined multiple times")
                defined_regions.append(reg.id)

            if reg.id not in unique_regions and reg.id is not None:
                raise ValueError(
                    f"EQLNUM regions {reg.id} is not found in the input simulation case"
                )

        if set(defined_regions) != set(unique_regions):
            print(
                "Values not defined for all EQLNUM regions. Default values will be used if defined."
            )
            if not default_exists:
                raise ValueError("Default values for EQLNUM regions not defined")

    if (
        config.model_parameters.equil.scheme != "regions_from_sim"
        and config.model_parameters.equil.eqlnum_region[0].id is not None
    ):
        raise ValueError(
            "Id for first equilibrium region parameter should not be set, or set to 'None'\n"
            "when using the 'global' or 'individual' options"
        )

    for phase in config.flownet.phases:
        if phase not in ["oil", "gas", "water", "disgas", "vapoil"]:
            raise ValueError(
                f"The {phase} phase is not a valid phase\n"
                f"The valid phases are 'oil', 'gas', 'water', 'disgas' and 'vapoil'"
            )
    if (
        not {"vapoil", "disgas"}.isdisjoint(config.flownet.phases)
        and not {
            "oil",
            "gas",
        }.issubset(config.flownet.phases)
    ):
        raise ValueError(
            "The phases 'vapoil' and 'disgas' can not be defined without the phases 'oil' and 'gas'"
        )

    if {"oil", "water"}.issubset(config.flownet.phases):
        req_relp_parameters = req_relp_parameters + [
            "scheme",
            "swirr",
            "swl",
            "swcr",
            "sorw",
            "nw",
            "now",
            "krwend",
            "krowend",
        ]
        for reg in config.model_parameters.equil.eqlnum_region:
            if (
                reg.owc_depth.min is None
                or reg.owc_depth.max is None
                or reg.owc_depth.max < reg.owc_depth.min
            ):
                raise ValueError(
                    "Ambiguous configuration input: OWC not properly specified. Min or max missing, or max < min."
                )

    if {"oil", "gas"}.issubset(config.flownet.phases):
        req_relp_parameters = req_relp_parameters + [
            "scheme",
            "swirr",
            "swl",
            "sgcr",
            "sorg",
            "ng",
            "nog",
            "krgend",
            "krogend",
        ]
        for reg in config.model_parameters.equil.eqlnum_region:
            if (
                reg.goc_depth.min is None
                or reg.goc_depth.max is None
                or reg.goc_depth.max < reg.goc_depth.min
            ):
                raise ValueError(
                    "Ambiguous configuration input: GOC not properly specified. Min or max missing, or max < min."
                )

    for parameter in set(req_relp_parameters):
        if parameter == "scheme":
            if (
                getattr(config.model_parameters.relative_permeability, parameter)
                != "global"
                and getattr(config.model_parameters.relative_permeability, parameter)
                != "individual"
                and getattr(config.model_parameters.relative_permeability, parameter)
                != "regions_from_sim"
            ):
                raise ValueError(
                    f"The relative permeability scheme "
                    f"'{config.model_parameters.relative_permeability.scheme}' is not valid.\n"
                    f"Valid options are 'global', 'regions_from_sim' or 'individual'."
                )
        else:
            if (
                getattr(config.model_parameters.relative_permeability, parameter).min
                is None
                or getattr(config.model_parameters.relative_permeability, parameter).max
                is None
            ):
                raise ValueError(
                    f"Ambiguous configuration input: The {parameter} parameter is missing or not properly defined."
                )
            if (
                getattr(config.model_parameters.relative_permeability, parameter).max
                < getattr(config.model_parameters.relative_permeability, parameter).min
            ):
                raise ValueError(
                    f"Ambiguous configuration input: The {parameter} setting 'max' is higher than the 'min'"
                )

    for parameter in set(config.model_parameters.relative_permeability._fields) - set(
        req_relp_parameters
    ):
        if (
            getattr(config.model_parameters.relative_permeability, parameter).min
            is not None
            and getattr(config.model_parameters.relative_permeability, parameter).max
            is not None
        ):
            raise ValueError(f"The {parameter} parameter should not be specified.")

    if config.ert.queue.system.upper() != "LOCAL" and (
        config.ert.queue.name is None or config.ert.queue.server is None
    ):
        raise ValueError(
            "Queue name and server needs to be provided if system is not 'LOCAL'."
        )

    for parameter in ["bulkvolume_mult", "porosity", "permeability", "fault_mult"]:
        if not getattr(config.model_parameters, parameter):
            continue
        if (
            getattr(config.model_parameters, parameter).min is not None
            and getattr(config.model_parameters, parameter).max is not None
            and getattr(config.model_parameters, parameter).mean is not None
        ):
            raise ValueError(
                f"You have set min, max and mean for parameter '{parameter}' - please only specify two at a time."
            )
        if (
            getattr(config.model_parameters, parameter).mean
            and getattr(config.model_parameters, parameter).max
            < getattr(config.model_parameters, parameter).mean
        ):
            raise ValueError(
                f"The {parameter} setting 'max' is set to a value "
                f"({getattr(config.model_parameters, parameter).max}) which is smaller than the value "
                f"set as 'mean' ({getattr(config.model_parameters, parameter).mean})."
            )
        if (
            getattr(config.model_parameters, parameter).min
            and getattr(config.model_parameters, parameter).max
            < getattr(config.model_parameters, parameter).min
        ):
            raise ValueError(
                f"The {parameter} setting 'max' is set to a value "
                f"({getattr(config.model_parameters, parameter).max}) which is smaller than the value "
                f"set as 'min' ({getattr(config.model_parameters, parameter).min})."
            )

    for suffix in [".DATA", ".EGRID", ".UNRST", ".UNSMRY", ".SMSPEC"]:
        input_file = pathlib.Path(
            config.flownet.data_source.simulation.input_case
        ).with_suffix(suffix)
        if not input_file.is_file():
            raise ValueError(f"The file {input_file} does not exist")

    if config.flownet.training_set_end_date and config.flownet.training_set_fraction:
        raise ValueError(
            "Ambiguous configuration input: 'training_set_fraction' and 'training_set_end_date' are "
            "both defined in the configuration file."
        )

    if any(config.model_parameters.rock_compressibility) and not all(
        config.model_parameters.rock_compressibility
    ):
        raise ValueError(
            "Ambiguous configuration input: 'rock_compressibility' needs to be defined using "
            "'reference_pressure', 'min' and 'max'. Currently one or more parameters are missing."
        )

    if (
        any(config.model_parameters.aquifer[0:3])
        or any(config.model_parameters.aquifer.size_in_bulkvolumes)
    ) and not (
        all(config.model_parameters.aquifer[0:3])
        and all(config.model_parameters.aquifer.size_in_bulkvolumes)
    ):
        raise ValueError(
            "Ambiguous configuration input: 'aquifer' needs to be defined using "
            "'scheme', 'fraction', 'delta_depth' and a 'size_in_bulkvolumes' distribution ('min', 'max', 'logunif')."
            "Currently one or more parameters are missing."
        )
    if all(config.model_parameters.aquifer[0:3]) and not all(
        config.model_parameters.aquifer.size_in_bulkvolumes
    ):
        raise ValueError(
            "Ambiguous configuration input: 'size_in_bulkvolumes' in 'aquifer' needs to be defined using "
            "'min', 'max' and 'log_unif'. Currently one or more parameters are missing."
        )

    return config
