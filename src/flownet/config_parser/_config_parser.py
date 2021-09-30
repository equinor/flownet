import warnings
import os
import pathlib
from typing import Dict, Optional, List

import numpy as np
import yaml
from configsuite import transformation_msg, types, MetaKeys as MK, ConfigSuite
import pandas as pd

from ._merge_configs import merge_configs
from ._config_transformations import (
    _integer_to_list,
    _str_none_to_none,
    _to_lower,
    _to_upper,
)
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
        config_folder: Optional path to folder with config file

    Returns:
        Dictionary to be used as configsuite type schema

    """

    @transformation_msg("Convert input string to absolute path")
    def _to_abs_path(path: Optional[str]) -> str:
        """
        Helper function for the configsuite. Takes in a path as a string and
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
                            "database": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "input_data": {
                                        MK.Type: types.String,
                                        MK.Transformation: _to_abs_path,
                                        MK.AllowNone: True,
                                        MK.Description: "CSV to be used as data source for FlowNet",
                                    },
                                },
                            },
                            "simulation": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "input_case": {
                                        MK.Type: types.String,
                                        MK.Transformation: _to_abs_path,
                                        MK.Description: "Simulation input case to be used as data source for FlowNet",
                                    },
                                    "well_logs": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                    "layers": {
                                        MK.Type: types.List,
                                        MK.Content: {
                                            MK.Item: {
                                                MK.Type: types.List,
                                                MK.Content: {
                                                    MK.Item: {
                                                        MK.Type: types.Integer,
                                                        MK.AllowNone: True,
                                                    }
                                                },
                                            },
                                        },
                                    },
                                },
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
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WBHP": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WOPT": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WGPT": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WWPT": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WOPR": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WGPR": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WWPR": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WWIR": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WGIR": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WWIT": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WGIT": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WSPR": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WSPT": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WSIR": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                    "WSIT": {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "rel_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "min_error": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                        },
                                    },
                                },
                            },
                            "resampling": {
                                MK.Type: types.String,
                                MK.AllowNone: True,
                                MK.Description: "Requested observation frequency, all pandas resampling options "
                                "are supported,  e.g. weekly (W), monthly (M), quarterly (Q), "
                                "yearly (A)",
                            },
                            "concave_hull": {
                                MK.Type: types.Bool,
                                MK.AllowNone: True,
                                MK.Description: "When true, the bounding boxes of the gridcells of the "
                                "original reservoir model are used to check if the generated additional "
                                "nodes are positioned within the reservoir volume.",
                            },
                        },
                    },
                    "constraining": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "kriging": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "enabled": {
                                        MK.Type: types.Bool,
                                        MK.Default: False,
                                        MK.Description: "Switch to enable or disable kriging on well log data.",
                                    },
                                    "n": {
                                        MK.Type: types.Integer,
                                        MK.Default: 20,
                                        MK.Description: "Number of kriged values in each direct. E.g, n = 10 -> "
                                        "10x10x10 = 1000 values.",
                                    },
                                    "n_lags": {
                                        MK.Type: types.Integer,
                                        MK.Default: 6,
                                        MK.Description: "Number of averaging bins for the semivariogram.",
                                    },
                                    "anisotropy_scaling_z": {
                                        MK.Type: types.Number,
                                        MK.Default: 10,
                                        MK.Description: "Scalar stretching value to take into account anisotropy. ",
                                    },
                                    "variogram_model": {
                                        MK.Type: types.String,
                                        MK.Default: "spherical",
                                        MK.Description: "Specifies which variogram model to use. See PyKridge "
                                        "documentation for valid options.",
                                    },
                                    "permeability_variogram_parameters": {
                                        MK.Type: types.Dict,
                                        MK.Description: "Parameters that define the specified variogram model. "
                                        "Permeability model sill and nugget are in log scale. See "
                                        "PyKridge documentation for valid options.",
                                        MK.Content: {
                                            MK.Key: {MK.Type: types.String},
                                            MK.Value: {MK.Type: types.Number},
                                        },
                                    },
                                    "porosity_variogram_parameters": {
                                        MK.Type: types.Dict,
                                        MK.Description: "Parameters that define the specified variogram model. See "
                                        "PyKridge documentation for valid options.",
                                        MK.Content: {
                                            MK.Key: {MK.Type: types.String},
                                            MK.Value: {MK.Type: types.Number},
                                        },
                                    },
                                },
                            },
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
                        MK.Type: types.List,
                        MK.Description: "List of additional flow nodes to add "
                        "for each layer or single integer which will be "
                        "split over the layers, when layers are defined.",
                        MK.LayerTransformation: _integer_to_list,
                        MK.Content: {
                            MK.Item: {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                            },
                        },
                    },
                    "additional_node_candidates": {
                        MK.Type: types.Integer,
                        MK.Default: 1000,
                        MK.Description: "Number of additional nodes to create "
                        "(using Mitchell's best candidate algorithm)",
                    },
                    "place_nodes_in_volume_reservoir": {
                        MK.Type: types.Bool,
                        MK.AllowNone: True,
                        MK.Description: "When true use boundary of reservoir/layer volume as "
                        "bounding volume to place initial candidates instead of convex hull of well perforations.",
                    },
                    "hull_factor": {
                        MK.Type: types.Number,
                        MK.Default: 1.2,
                        MK.Description: "Increase the size of the bounding volume around the well "
                        "perforations to place additional nodes in.",
                    },
                    "random_seed": {
                        MK.Type: types.Number,
                        MK.AllowNone: True,
                        MK.Description: "Adding this makes sure two FlowNet "
                        "runs create the exact same output",
                    },
                    "mitchells_algorithm": {
                        MK.Type: types.String,
                        MK.Default: "normal",
                        MK.Transformation: _to_lower,
                        MK.Description: "Choose which mitchell's best candidate algorithm to run for the placement "
                        "of additional nodes. Options: normal or fast. The normal algorithm will place the nodes more "
                        "evenly acros the volume, but takes long. "
                        "The fast option is faster, but results in a less even spread of the nodes. "
                        "This can be improved by increasing the number of additional "
                        "node candidates.",
                    },
                    "perforation_handling_strategy": {
                        MK.Type: types.String,
                        MK.Default: "bottom_point",
                        MK.Description: "Strategy to be used when creating perforations. Valid options are "
                        "'bottom'_point', 'top_point', 'multiple', 'time_avg_open_location' and "
                        "'multiple_based_on_workovers'.",
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
                    "prod_control_mode": {
                        MK.Type: types.String,
                        MK.Default: "RESV",
                    },
                    "inj_control_mode": {
                        MK.Type: types.String,
                        MK.Default: "RATE",
                    },
                    "angle_threshold": {
                        MK.Type: types.Number,
                        MK.Default: 150,
                        MK.Description: "Angle threshold used, after Delaunay triagulation,"
                        "to remove sides/tubes opposite angles larger than the supplied treshold.",
                    },
                    "n_non_reservoir_evaluation": {
                        MK.Type: types.Number,
                        MK.Default: 10,
                        MK.Description: "Number of points along a tube to check whether they are in"
                        "non reservoir for removal purposes.",
                    },
                    "min_permeability": {
                        MK.Type: types.Number,
                        MK.AllowNone: True,
                        MK.Description: "Minimum allowed permeability in mD before a tube is removed "
                        "(i.e., its cells are made inactive).",
                    },
                    "prior_volume_distribution": {
                        MK.Type: types.String,
                        MK.Default: "tube_length",
                        MK.Description: "Volume distribution method of tubes (or cells in tube) to be "
                        "applied on the prior volume distribution. Based on tube length by default. "
                        "Valid options are: "
                        "* tube_length: distrubutes the volume of the convex hull of the FlowNet model,"
                        "    based on the length of a tube. I.e., if all tubes have equeal lenght, they"
                        "    will have equal volume."
                        "* voronoi_per_tube: distributes the input models bulk volume of active cells"
                        "    to the nearest FlowNet tube of a cell. The total volume of the tube is then"
                        "    devided equally over the cells of the tube. I.e., in areas with a higher"
                        "    FlowNet tube density, the volume per cell is lower. Mind that if the FlowNet"
                        "    model, i.e., the convex hull of the well connections, is much smaller than the"
                        "    original model volume outside of the well connection convex hull might be"
                        "    collapsed at the borders of the model. I.e., the borders of your model could"
                        "    het unrealisticly large volumes. This can be mitigated by increasing the hull"
                        "    factor of the FlowNet model generation process or by setting the "
                        "    place_nodes_in_volume_reservoir to true.",
                    },
                    "hyperopt": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "n_runs": {
                                MK.Type: types.Number,
                                MK.Default: 10,
                                MK.Description: "Number of runs flownet ahm runs in Hyperopt run.",
                            },
                            "mode": {
                                MK.Type: types.String,
                                MK.Default: "random",
                                MK.Description: "Hyperopt mode to run with. Valid options are 'random', "
                                "'tpe' and 'adaptive_tpe'.",
                            },
                            "loss": {
                                MK.Type: types.NamedDict,
                                MK.Description: "Definition of the hyperopt loss function. The definitions "
                                "refer to the first analysis workflow ONLY.",
                                MK.Content: {
                                    "keys": {
                                        MK.Type: types.List,
                                        MK.Content: {
                                            MK.Item: {
                                                MK.Type: types.String,
                                                MK.AllowNone: True,
                                            },
                                        },
                                        MK.Description: "List of keys, as defined in the analysis section, "
                                        "to be used as loss function for Hyperopt.",
                                    },
                                    "factors": {
                                        MK.Type: types.List,
                                        MK.Content: {
                                            MK.Item: {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            }
                                        },
                                        MK.Description: "List of factors to scale the keys.",
                                    },
                                    "metric": {
                                        MK.Type: types.String,
                                        MK.Default: "RMSE",
                                        MK.Description: "Metric to be used in Hyperopt.",
                                    },
                                },
                            },
                        },
                    },
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
                    "timeout": {
                        MK.Type: types.Number,
                        MK.Default: 3600,
                        MK.Description: "Maximum number of seconds of inactivity from ERT before a FlowNet "
                        "run is killed. When running many realizations, with many parameters this timeout "
                        "should be set to a high value. When you are running a hyperopt run you might want "
                        "to lower this number as to not waste too much time in cases where ERT hangs.",
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
                        MK.Type: types.List,
                        MK.Description: "List of analysis workflows to run.",
                        MK.Content: {
                            MK.Item: {
                                MK.Type: types.NamedDict,
                                MK.Description: "Definitions of the analysis workflow.",
                                MK.Content: {
                                    "metric": {
                                        MK.Type: types.List,
                                        MK.Content: {
                                            MK.Item: {
                                                MK.Type: types.String,
                                                MK.AllowNone: True,
                                            }
                                        },
                                        MK.Transformation: _to_upper,
                                        MK.Description: "List of accuracy metrics to be computed "
                                        "in FlowNet analysis workflow. "
                                        "Supported metrics: MSE, RMSE, NRMSE, MAE, NMAE, R2",
                                    },
                                    "quantity": {
                                        MK.Type: types.List,
                                        MK.Content: {
                                            MK.Item: {
                                                MK.Type: types.String,
                                                MK.AllowNone: True,
                                            }
                                        },
                                        MK.Transformation: _to_upper,
                                        MK.Description: "List of summary vectors for which accuracy "
                                        "is to be computed",
                                    },
                                    "start": {
                                        MK.Type: types.Date,
                                        MK.AllowNone: True,
                                        MK.Description: "Start date in YYYY-MM-DD format.",
                                    },
                                    "end": {
                                        MK.Type: types.Date,
                                        MK.AllowNone: True,
                                        MK.Description: "End date in YYYY-MM-DD format.",
                                    },
                                    "outfile": {
                                        MK.Type: types.String,
                                        MK.AllowNone: True,
                                        MK.Description: "The filename of the output of the workflow. "
                                        "In case multiple analysis workflows are run this name should be unique.",
                                    },
                                },
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
                            "min": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "mean": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "max": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "base": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "stddev": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "distribution": {
                                MK.Type: types.String,
                                MK.Default: "uniform",
                                MK.Transformation: _to_lower,
                            },
                        },
                    },
                    "permeability_regional_scheme": {
                        MK.Type: types.String,
                        MK.Description: "The individual flow tubes will always have an individual permeability "
                        "prior distribution attached (specified as permeability in the config file). Setting this "
                        "option to 'individual' makes the individual uncertainty for each flow tube the only "
                        "permeability uncertainty. Using the 'global' option will add one additional multiplier "
                        "for the entire FlowNet model, and 'regions_from_sim' will add one multiplier for each "
                        "of the regions present in a region parameter from an input simulation model "
                        "(parameter name specified in 'permeability_parameter_from_sim_model').",
                        MK.Default: "individual",
                        MK.Transformation: _to_lower,
                    },
                    "permeability_parameter_from_sim_model": {
                        MK.Type: types.String,
                        MK.Description: "The name of the region parameter in the simulation model to "
                        "base a regional permeability multiplier on.",
                        MK.Default: "EQLNUM",
                    },
                    "permeability_regional": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of the regional permeability multiplier prior "
                        "distribution.",
                        MK.Content: {
                            "min": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "mean": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "max": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "base": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "stddev": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "distribution": {
                                MK.Type: types.String,
                                MK.Default: "uniform",
                                MK.Transformation: _to_lower,
                            },
                        },
                    },
                    "porosity": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of the porosity prior "
                        "distribution. You define either min and max, or one of "
                        "the endpoints and mean",
                        MK.Content: {
                            "min": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "mean": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "max": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "base": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "stddev": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "distribution": {
                                MK.Type: types.String,
                                MK.Default: "uniform",
                                MK.Transformation: _to_lower,
                            },
                        },
                    },
                    "porosity_regional_scheme": {
                        MK.Type: types.String,
                        MK.Description: "The individual flow tubes will always have an individual porosity "
                        "prior distribution attached (specified as porosity in the config file). Setting this "
                        "option to 'individual' makes the individual uncertainty for each flow tube the only "
                        "porosity uncertainty. Using the 'global' option will add one additional multiplier "
                        "for the entire FlowNet model, and 'regions_from_sim' will add one multiplier for each "
                        "of the regions present in a region parameter from an input simulation model "
                        "(parameter name specified in 'porosity_parameter_from_sim_model').",
                        MK.Default: "individual",
                        MK.Transformation: _to_lower,
                    },
                    "porosity_parameter_from_sim_model": {
                        MK.Type: types.String,
                        MK.Description: "The name of the region parameter in the simulation model to "
                        "base a regional permeability multiplier on.",
                        MK.Default: "EQLNUM",
                    },
                    "porosity_regional": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of the regional porosity multiplier prior "
                        "distribution.",
                        MK.Content: {
                            "min": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "mean": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "max": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "base": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "stddev": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "distribution": {
                                MK.Type: types.String,
                                MK.Default: "uniform",
                                MK.Transformation: _to_lower,
                            },
                        },
                    },
                    "bulkvolume_mult_regional_scheme": {
                        MK.Type: types.String,
                        MK.Description: "The individual flow tubes will always have an individual bulk volume "
                        "multiplier attached (specified as bulkvolume_mult in the config file). Setting this "
                        "option to 'individual' makes the individual multipliers for each flow tube the only "
                        "bulk volume uncertainty. Using the 'global' option will add one additional multiplier "
                        "for the entire FlowNet model, and 'regions_from_sim' will add one multiplier for each "
                        "of the regions present in a region parameter from an input simulation model "
                        "(parameter name specified in 'bulkvolume_mult_parameter_from_sim_model').",
                        MK.Default: "individual",
                        MK.Transformation: _to_lower,
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
                            "min": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "mean": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "max": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "base": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "stddev": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "distribution": {
                                MK.Type: types.String,
                                MK.Default: "uniform",
                                MK.Transformation: _to_lower,
                            },
                        },
                    },
                    "bulkvolume_mult_parameter_from_sim_model": {
                        MK.Type: types.String,
                        MK.Description: "The name of the region parameter in the simulation model to "
                        "base a regional bulk volume parameter on.",
                        MK.Default: "EQLNUM",
                    },
                    "bulkvolume_mult_regional": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of bulk volume multiplier "
                        "prior distribution. You define either min and max, or one "
                        "of the endpoints and mean. One multiplies is drawn per tube "
                        "which decreases/increases the default tube bulk volume "
                        "(which again is a proportional part of model convex hull bulk "
                        "with respect to tube length compared to other tubes).",
                        MK.Content: {
                            "min": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "mean": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "max": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "base": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "stddev": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "distribution": {
                                MK.Type: types.String,
                                MK.Default: "uniform",
                                MK.Transformation: _to_lower,
                            },
                        },
                    },
                    "fault_mult": {
                        MK.Type: types.NamedDict,
                        MK.Description: "Description of the fault multiplicator "
                        "prior distribution. You define either min and max, or one "
                        "of the endpoints and mean",
                        MK.Content: {
                            "min": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "mean": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "max": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "base": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "stddev": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                                MK.Transformation: _str_none_to_none,
                            },
                            "distribution": {
                                MK.Type: types.String,
                                MK.Default: "uniform",
                                MK.Transformation: _to_lower,
                            },
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
                                "permeability curves for the whole model), 'regions_from_sim' "
                                "(one set of curves for each SATNUM region in data source simulation) "
                                "or 'individual' (one set of curves per tube).",
                                MK.Default: "global",
                                MK.Transformation: _to_lower,
                            },
                            "region_parameter_from_sim_model": {
                                MK.Type: types.String,
                                MK.Description: "The name of the regions parameter in the simulation model to "
                                "base the relative permeability region parameter on.",
                                MK.Default: "SATNUM",
                            },
                            "interpolate": {
                                MK.Type: types.Bool,
                                MK.Description: "Uses the interpolation option between low/base/high "
                                "relative permeability curves if set to True (one interpolation "
                                "per SATNUM region. Only available for three phase problems.",
                                MK.Default: False,
                            },
                            "independent_interpolation": {
                                MK.Type: types.Bool,
                                MK.Description: "The interpolation between low/base/high relative permeability curves "
                                "is performed independently for oil/gas and oil/water "
                                "per SATNUM region. Only available for three phase problems.",
                                MK.Default: False,
                            },
                            "swcr_add_to_swl": {
                                MK.Type: types.Bool,
                                MK.Description: "Allows for calculating SWCR by adding a number to SWL. Especially "
                                "useful to avoid non-physical values when defining prior distributions. If this "
                                "parameter is set to true, the numbers defined under swcr will be used to define "
                                "a prior distribution for the delta value added to SWL, instead of defining the "
                                "prior distribution for SWCR directly.",
                                MK.Default: False,
                            },
                            "krwmax_add_to_krwend": {
                                MK.Type: types.Bool,
                                MK.Description: "Allows for calculating KRWMAX by adding a number to KRWEND. "
                                "Especially useful to avoid non-physical values when defining prior distributions. "
                                "If this parameter is set to true, the numbers defined under KRWMAX will be used to "
                                "define a prior distribution for the delta value added to KRWEND, instead of defining "
                                "the prior distribution for KRWMAX directly.",
                                MK.Default: False,
                            },
                            "regions": {
                                MK.Type: types.List,
                                MK.Content: {
                                    MK.Item: {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "id": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                                MK.Transformation: _str_none_to_none,
                                            },
                                            "swirr": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: False,
                                                        MK.Description: "This only has effect if interpolation "
                                                        "between low/base/high relative permeability curves is chosen. "
                                                        "If 'low_optimistic' is set to True, the minimum value for "
                                                        "this model parameter will be used to generate the high case "
                                                        "SCAL recommendation curves, and the maximum value will be "
                                                        "used for the low case SCAL recommendation curves.",
                                                    },
                                                },
                                            },
                                            "swl": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: False,
                                                    },
                                                },
                                            },
                                            "swcr": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: False,
                                                    },
                                                },
                                            },
                                            "sorw": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: True,
                                                    },
                                                },
                                            },
                                            "krwend": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: True,
                                                    },
                                                },
                                            },
                                            "krwmax": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: True,
                                                    },
                                                },
                                            },
                                            "kroend": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: False,
                                                    },
                                                },
                                            },
                                            "nw": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: False,
                                                    },
                                                },
                                            },
                                            "now": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: True,
                                                    },
                                                },
                                            },
                                            "sorg": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: True,
                                                    },
                                                },
                                            },
                                            "sgcr": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: False,
                                                    },
                                                },
                                            },
                                            "ng": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "loguniform": {
                                                        MK.Type: types.Bool,
                                                        MK.AllowNone: True,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: False,
                                                    },
                                                },
                                            },
                                            "nog": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: True,
                                                    },
                                                },
                                            },
                                            "krgend": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                    "low_optimistic": {
                                                        MK.Type: types.Bool,
                                                        MK.Default: True,
                                                    },
                                                },
                                            },
                                        },
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
                                "'regions_from_sim' (EQLNUM regions extracted from input simulation",
                                MK.Default: "global",
                                MK.Transformation: _to_lower,
                            },
                            "region_parameter_from_sim_model": {
                                MK.Type: types.String,
                                MK.Description: "The name of the regions parameter in the simulation model to "
                                "base the equilibrium parameter on.",
                                MK.Default: "EQLNUM",
                            },
                            "regions": {
                                MK.Type: types.List,
                                MK.Content: {
                                    MK.Item: {
                                        MK.Type: types.NamedDict,
                                        MK.Content: {
                                            "id": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                                MK.Transformation: _str_none_to_none,
                                            },
                                            "datum_depth": {
                                                MK.Type: types.Number,
                                                MK.AllowNone: True,
                                            },
                                            "datum_pressure": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                },
                                            },
                                            "owc_depth": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                },
                                            },
                                            "gwc_depth": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
                                                    },
                                                },
                                            },
                                            "goc_depth": {
                                                MK.Type: types.NamedDict,
                                                MK.Content: {
                                                    "min": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "mean": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "max": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "base": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "stddev": {
                                                        MK.Type: types.Number,
                                                        MK.AllowNone: True,
                                                        MK.Transformation: _str_none_to_none,
                                                    },
                                                    "distribution": {
                                                        MK.Type: types.String,
                                                        MK.Default: "uniform",
                                                        MK.Transformation: _to_lower,
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
                                        MK.Transformation: _str_none_to_none,
                                    },
                                    "mean": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                        MK.Transformation: _str_none_to_none,
                                    },
                                    "max": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                        MK.Transformation: _str_none_to_none,
                                    },
                                    "base": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                        MK.Transformation: _str_none_to_none,
                                    },
                                    "stddev": {
                                        MK.Type: types.Number,
                                        MK.AllowNone: True,
                                        MK.Transformation: _str_none_to_none,
                                    },
                                    "distribution": {
                                        MK.Type: types.String,
                                        MK.Default: "uniform",
                                        MK.Transformation: _to_lower,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def parse_config(
    base_config: pathlib.Path, update_config: pathlib.Path = None
) -> ConfigSuite.snapshot:
    """
    Takes in path to a yaml configuration file, parses it, populates with default values
    where that is defined and the has not provided his/her own value. Also error checks input
    arguments, and making sure they are of expected type.

    Args:
        base_config: Path to the main configuration file.
        update_config: Optional configuration file with
            key/values to override in main configuration file.

    Returns:
        Parsed config, where values can be extracted like e.g. 'config.ert.queue.system'.

    """
    # pylint: disable=too-many-branches, too-many-statements, too-many-lines, too-many-locals

    if update_config is None:
        input_config = yaml.safe_load(base_config.read_text())
    else:
        input_config = merge_configs(
            yaml.safe_load(base_config.read_text()),
            yaml.safe_load(update_config.read_text()),
        )

    suite = ConfigSuite(
        input_config,
        create_schema(config_folder=base_config.parent),
        deduce_required=True,
    )

    if not suite.valid:
        raise ValueError(
            "The configuration is not valid:"
            + ", ".join([error.msg for error in suite.errors])
        )

    available_region_schemes = ["global", "individual", "regions_from_sim"]
    config = suite.snapshot
    if (
        config.model_parameters.relative_permeability.interpolate
        and config.model_parameters.relative_permeability.swcr_add_to_swl
    ):
        raise ValueError(
            "SWCR_ADD_TO_SWL can not be used together with the "
            "interpolation option for relative permeability."
        )

    # If 'regions_from_sim' is defined for any parameter, or a csv file with
    # rsvd tables is defined, we need to import the simulation case to check
    # number of regions
    if (
        "regions_from_sim"
        in (
            config.model_parameters.equil.scheme,
            config.model_parameters.relative_permeability.scheme,
            config.model_parameters.porosity_regional_scheme,
            config.model_parameters.permeability_regional_scheme,
            config.model_parameters.bulkvolume_mult_regional_scheme,
        )
        or config.flownet.pvt.rsvd
    ):
        if config.flownet.data_source.simulation.input_case is None:
            raise ValueError("Input simulation case is not defined.")
        field_data = FlowData(config.flownet.data_source.simulation.input_case)

    region_parameters: dict = {
        "equil": "EQLNUM",
        "relative_permeability": "SATNUM",
        "permeability": None,
        "porosity": None,
        "bulkvolume_mult": None,
    }
    unique_regions: dict = {}
    for reg_param, flow_region_name in region_parameters.items():
        if flow_region_name is not None:
            scheme = getattr(getattr(config.model_parameters, reg_param), "scheme")
        else:
            scheme = getattr(config.model_parameters, reg_param + "_regional_scheme")
        if scheme not in available_region_schemes:
            raise ValueError(
                f"The {reg_param} scheme "
                f"'{scheme}' is not valid.\n"
                f"Valid options are {available_region_schemes}"
            )
        if scheme == "regions_from_sim":
            if flow_region_name is not None:
                reg_param_sim_model = getattr(
                    getattr(config.model_parameters, reg_param),
                    "region_parameter_from_sim_model",
                )
            else:
                reg_param_sim_model = getattr(
                    config.model_parameters, reg_param + "_parameter_from_sim_model"
                )
            try:
                unique_regions[reg_param] = field_data.get_unique_regions(
                    reg_param_sim_model
                )
            except KeyError as err:
                raise ValueError(
                    f"REGION parameter {reg_param_sim_model} "
                    "not found in input simulation model."
                ) from err
            if flow_region_name:
                _check_if_all_region_priors_defined(
                    getattr(config.model_parameters, reg_param),
                    unique_regions[reg_param],
                    flow_region_name,
                )
        else:
            if flow_region_name is not None:
                regions = getattr(
                    getattr(config.model_parameters, reg_param), "regions"
                )
                if regions[0].id is not None:
                    raise ValueError(
                        f"The region number for the first {reg_param} region parameter should not be set, \n"
                        "or set to 'None' when using the 'global' or 'individual' options"
                    )

    layers = config.flownet.data_source.simulation.layers
    if len(layers) > 0:
        if not all(
            d == 1
            for d in [layers[i][0] - layers[i - 1][-1] for i in range(1, len(layers))]
        ):
            raise ValueError(
                f"The layering definition "
                f"'{layers}' is not valid.\n"
                f"Layer groups should be adjacent, e.g. ((start, end),(end+1, ..)) ."
            )

        if not all(l == 2 for l in [len(layer) for layer in layers]):
            raise ValueError(
                f"The layering definition "
                f"'{layers}' is not valid.\n"
                f"Valid way to define the layering is as nested list with "
                f"layer intervals. e.g. ((1, 5),(6, 9),(10, 15))."
            )
    if layers and not config.flownet.data_source.concave_hull:
        raise ValueError(
            "Concave_hull should be True when layers are defined. "
            "The concave hulls of the layers are used to split "
            "the number of additional nodes between the layers."
        )
    if (
        layers
        and not len(layers) == len(config.flownet.additional_flow_nodes)
        and len(config.flownet.additional_flow_nodes) != 1
    ):
        raise ValueError(
            "Either add for each layer in the FlowNet model an entry in the "
            "additional flow nodes list or fill out a single additional flow "
            f"node value to be split over the layers. Currenly you have {str(len(layers))} layers "
            f"and {str(len(config.flownet.additional_flow_nodes))} additional flow node "
            "defitions."
        )
    if not layers and len(config.flownet.additional_flow_nodes) > 1:
        raise ValueError(
            "You supplied multiple entries for the additional flow nodes but "
            "there is only a single layer."
        )

    if (
        config.flownet.place_nodes_in_volume_reservoir
        and not config.flownet.data_source.concave_hull
    ):
        raise ValueError(
            "concave_hull needs to be true for flownet to be able to "
            "place candidates within the reservoir volume."
        )

    if config.flownet.mitchells_algorithm not in ("normal", "fast"):
        raise ValueError(
            f"'{config.flownet.mitchells_algorithm}' is not a valid mitchells_algorithm."
        )

    prod_control_modes = {"ORAT", "GRAT", "WRAT", "LRAT", "RESV", "BHP"}
    if config.flownet.prod_control_mode not in prod_control_modes:
        raise ValueError(
            f"The injection control mode "
            f"'{config.flownet.prod_control_mode}' is not valid.\n"
            f"Valid options are {prod_control_modes}. "
        )
    inj_control_modes = {"RATE", "BHP"}
    if config.flownet.inj_control_mode not in inj_control_modes:
        raise ValueError(
            f"The injection control mode "
            f"'{config.flownet.inj_control_mode}' is not valid.\n"
            f"Valid options are {inj_control_modes}. "
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

    req_relp_parameters: List[str] = []
    if {"oil", "water"}.issubset(config.flownet.phases):
        req_relp_parameters = req_relp_parameters + [
            "swirr",
            "swl",
            "swcr",
            "sorw",
            "nw",
            "now",
            "krwend",
            "kroend",
        ]
        if config.model_parameters.relative_permeability.krwmax_add_to_krwend:
            req_relp_parameters = req_relp_parameters + ["krwmax"]
        for reg in config.model_parameters.equil.regions:
            _check_distribution(reg, "owc_depth")

    if {"oil", "gas"}.issubset(config.flownet.phases):
        req_relp_parameters = req_relp_parameters + [
            "swirr",
            "swl",
            "sgcr",
            "sorg",
            "ng",
            "nog",
            "krgend",
            "kroend",
        ]
        for reg in config.model_parameters.equil.regions:
            _check_distribution(reg, "goc_depth")

    for parameter in set(req_relp_parameters):
        for satreg in config.model_parameters.relative_permeability.regions:
            if config.model_parameters.relative_permeability.interpolate:
                _check_interpolate(satreg, parameter)
            else:
                _check_distribution(satreg, parameter)

    for parameter in (
        set(config.model_parameters.relative_permeability.regions[0]._fields)
        - set(req_relp_parameters)
        - {"id", "krwmax"}
    ):
        for satreg in config.model_parameters.relative_permeability.regions:
            if len(_check_defined(satreg, parameter)) > 0:
                raise ValueError(f"The {parameter} parameter should not be specified.")

    if config.ert.queue.system.upper() != "LOCAL" and (
        config.ert.queue.name is None or config.ert.queue.server is None
    ):
        raise ValueError(
            "Queue name and server needs to be provided if system is not 'LOCAL'."
        )

    for parameter in [
        "bulkvolume_mult",
        "porosity",
        "permeability",
        "fault_mult",
        "permeability_regional",
        "porosity_regional",
        "bulkvolume_mult_regional",
    ]:
        if not len(_check_defined(config.model_parameters, parameter)) > 0:
            continue
        _check_distribution(config.model_parameters, parameter)

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

    if any(config.model_parameters.aquifer[0:3]) or _check_defined(
        config.model_parameters.aquifer, "size_in_bulkvolumes"
    ):
        if not all(config.model_parameters.aquifer[0:3]):
            raise ValueError(
                "Ambiguous configuration input: 'aquifer' needs to be defined using "
                "'scheme', 'fraction', and 'delta_depth'."
                "Currently one or more parameters are missing."
            )
        if config.model_parameters.aquifer.scheme not in ("global", "individual"):
            raise ValueError(
                f"The aquifer scheme "
                f"'{config.model_parameters.aquifer.scheme}' is not valid.\n"
                f"Valid options are 'global' or 'individual'."
            )
        _check_distribution(config.model_parameters.aquifer, "size_in_bulkvolumes")

    if (
        config.flownet.constraining.kriging.enabled
        and not config.flownet.data_source.simulation.well_logs
    ):
        raise ValueError(
            "Ambiguous configuration input: well log data needs to be loaded (from the simulation model) in order "
            "to allow for enabling of kriging."
        )

    if (config.flownet.hyperopt.mode) not in ("random", "tpe", "adaptive_tpe"):
        raise ValueError(
            f"The hyperopt mode '{config.flownet.hyperopt.mode}' is not valid."
            "Valid options are ('random', 'tpe', 'adaptive_tpe')."
        )

    for key in config.flownet.hyperopt.loss.keys:
        if (
            not len(config.ert.analysis) > 0
            or not key in config.ert.analysis[0].quantity
        ):
            raise ValueError(
                f"Key {key} is not defined as an analysis quantity ({config.flownet.hyperopt.loss.keys})."
            )

    if (
        hasattr(config.ert, "analysis")
        and len(config.ert.analysis) > 0
        and config.flownet.hyperopt.loss.metric not in config.ert.analysis[0].metric
    ):
        raise ValueError(
            f"Key {config.flownet.hyperopt.loss.metric} is not defined as an analysis"
            "quantity ({config.ert.analysis.metric})."
        )

    if len(config.flownet.hyperopt.loss.keys) != len(
        config.flownet.hyperopt.loss.factors
    ):
        raise ValueError(
            "For each loss function metric specified, factors need to be specified as well."
        )

    if config.flownet.hyperopt.n_runs < 1:
        raise ValueError("The minimum number of hyperopt runs 'n_runs' is 1.")

    if config.flownet.pvt.rsvd:
        df_rsvd = pd.read_csv(config.flownet.pvt.rsvd)
        if len(df_rsvd.columns) == 2:
            if not set(df_rsvd.columns.str.lower()) == {"depth", "rs"}:
                raise ValueError(
                    "With only one rsvd table as input the "
                    "column names should be 'depth' and 'rs'."
                )
        elif (
            len(df_rsvd.columns) == 3
            and config.model_parameters.equil.scheme == "regions_from_sim"
        ):
            if not set(df_rsvd.columns.str.lower()) == {"depth", "eqlnum", "rs"}:
                raise ValueError(
                    "Column names in csv file with rsvd values should be "
                    "'depth', 'rs' and 'eqlnum' (in any order)."
                )
            if not set(df_rsvd["eqlnum"]) == set(unique_regions["equil"]):
                raise ValueError(
                    "Rsvd tables not defined for all EQLNUM regions. Must be defined as one "
                    "table used for all regions, or one table for each region."
                )
        else:
            raise ValueError(
                "Something is wrong with the csv file containing the rsvd tables."
                "It should contain either two columns with headers 'depth' and 'rs', "
                "or three columns with headers 'depth','rs' and 'eqlnum'."
                "The option with three columns should only be used together with the "
                "'regions_from_sim' scheme for equilibrium regions."
            )

    if not config.flownet.prior_volume_distribution in [
        "voronoi_per_tube",
        "tube_length",
    ]:
        raise ValueError(
            f"'{config.flownet.prior_volume_distribution}' is not a valid prior volume "
            "distribution method. You can either use 'voronoi_per_tube' or 'tube_length'."
        )
    if (config.flownet.prior_volume_distribution == "voronoi_per_tube") and (
        config.flownet.data_source.simulation.input_case is None
    ):
        raise ValueError(
            f"'The {config.flownet.prior_volume_distribution}' volume distribution "
            "method can only be used when a simulation model is supplied as datasource."
        )

    fld = []
    for idx, _ in enumerate(config.flownet.data_source.vectors):
        obs = config.flownet.data_source.vectors[idx]
        if (obs.min_error is not None and obs.rel_error is None) or (
            obs.min_error is None and obs.rel_error is not None
        ):
            fld.append(config.flownet.data_source.vectors._fields[idx])

    if fld:
        raise ValueError(
            f"Both min_error and rel_error must be set. "
            f"This requirement is not satisfied for "
            f" {fld}  observations"
        )

    return config


def _check_interpolate(path_in_config_dict: dict, parameter: str):
    """
    Helper function to check the parameter in question is defined correctly for using
    interpolation between SCAL recommendation curves in pyscal

    Args:
        path_in_config_dict (str): a location in the config schema dictionary
        parameter (str): a parameter/dictionary found at the given location

    Returns:
       Nothing

    Raises:
        ValueError: If the relative permeability input does not contain information for min/base/max
            when the interpolation option is selected
    """
    defined_parameters = _check_defined(path_in_config_dict, parameter)
    _check_for_negative_values(path_in_config_dict, parameter)
    _check_order_of_values(path_in_config_dict, parameter)
    if len({"min", "base", "max"} - defined_parameters) > 0:
        raise ValueError(
            f"Ambigous configuration input for parameter {parameter}. "
            f"When interpolating between 'low', 'base' and 'high' relative permeability curves, "
            f"'min', 'base' and 'max' must all be defined."
        )


def _check_for_negative_values(path_in_config_dict: dict, parameter: str):
    """
    Helper function to check if there are any negative values defined
    for a given parameter.

    Args:
        path_in_config_dict (str): a location in the config schema dictionary
        parameter (str): a parameter/dictionary found at the given location

    Returns:
        Nothing

    Raises:
        ValueError: If a parameter has negative input value defined for min/max/mean/base/stddev
    """
    defined_parameters = _check_defined(path_in_config_dict, parameter)
    # check for negative values
    for attr in defined_parameters:
        if getattr(getattr(path_in_config_dict, parameter), attr) < 0:
            raise ValueError(
                f"Ambiguous configuration input for {parameter}. The '{attr}' is negative."
            )


def _check_order_of_values(path_in_config_dict: dict, parameter: str):
    """
    Helper function to check the order of the defined values
    for a given parameter. If defined, the following should be true:
        * Min < Base
        * Min < Mean
        * Min < Max
        * Mean < Max
        * Base < Max

    Args:
        path_in_config_dict (str): a location in the config schema dictionary
        parameter (str): a parameter/dictionary found at the given location

    Returns:
        Nothing

    Raises:
        ValueError: If the order of the defined min/base/mean/max values for a parameter is not correct
    """
    defined_parameters = _check_defined(path_in_config_dict, parameter)
    if {"min", "max"}.issubset(defined_parameters):
        if (
            getattr(path_in_config_dict, parameter).min
            > getattr(path_in_config_dict, parameter).max
        ):
            raise ValueError(
                f"Ambiguous configuration input for {parameter}. 'Min' is larger than 'max'."
            )
    if {"min", "mean"}.issubset(defined_parameters):
        if (
            getattr(path_in_config_dict, parameter).min
            > getattr(path_in_config_dict, parameter).mean
        ):
            raise ValueError(
                f"Ambiguous configuration input for {parameter}. 'Min' is larger than 'mean'."
            )
    if {"max", "mean"}.issubset(defined_parameters):
        if (
            getattr(path_in_config_dict, parameter).mean
            > getattr(path_in_config_dict, parameter).max
        ):
            raise ValueError(
                f"Ambiguous configuration input for {parameter}. 'Mean' is larger than 'max'."
            )
    if {"min", "base"}.issubset(defined_parameters):
        if (
            getattr(path_in_config_dict, parameter).min
            > getattr(path_in_config_dict, parameter).base
        ):
            raise ValueError(
                f"Ambiguous configuration input for {parameter}. 'Min' is larger than 'base'."
            )
    if {"max", "base"}.issubset(defined_parameters):
        if (
            getattr(path_in_config_dict, parameter).base
            > getattr(path_in_config_dict, parameter).max
        ):
            raise ValueError(
                f"Ambiguous configuration input for {parameter}. 'Base' is larger than 'max'."
            )


def _check_distribution(path_in_config_dict: dict, parameter: str):
    """
    Helper function that performs a number of checks to make sure that the values defined in the config file
    for the parameter in question makes sense

    Args:
        path_in_config_dict (str): a location in the config schema dictionary
        parameter (str): a parameter/dictionary found at the given location

    Returns:
       Nothing

    Raises:
        ValueError: If the choice of input distribution is not valid
        ValueError: If the combination of min/max/base/mean/stddev deinfed in the config is not right for the
            chosen distribution
    """
    # pylint: disable=too-many-branches
    if not {getattr(path_in_config_dict, parameter).distribution}.issubset(
        {
            "uniform",
            "logunif",
            "const",
            "normal",
            "lognormal",
            "truncated_normal",
            "triangular",
        }
    ):
        raise ValueError(
            f"The defined distribution ({getattr(path_in_config_dict, parameter).distribution}) for {parameter} "
            f"is not a valid distribution choice. Use 'uniform', 'logunif', 'normal', 'lognormal', 'truncated_normal' "
            f"'triangular' or 'const'."
        )
    defined_parameters = _check_defined(path_in_config_dict, parameter)
    _check_for_negative_values(path_in_config_dict, parameter)
    _check_order_of_values(path_in_config_dict, parameter)
    # uniform can be defined by either min/max, min/mean or mean/max
    if getattr(path_in_config_dict, parameter).distribution in ("uniform", "logunif"):
        if {"min", "max", "mean"}.issubset(defined_parameters):
            raise ValueError(
                f"The {parameter} has values defined for 'min', 'mean' and 'max'. Only two of them should be defined"
            )
        if {"mean", "max"}.issubset(defined_parameters):
            if getattr(path_in_config_dict, parameter).distribution == "uniform":
                # check that calculated 'min' will be non negative
                if (
                    2 * getattr(path_in_config_dict, parameter).mean
                    - getattr(path_in_config_dict, parameter).max
                    < 0
                ):
                    raise ValueError(
                        f"The 'mean' and 'max' values for {parameter} will give negative 'min' "
                        f"in the uniform distribution."
                    )
            else:
                warnings.warn(
                    f"Check not implemented, but be aware that the defined 'mean' and 'max' value for {parameter} "
                    f"may provide a negative minimum value that will lead to the run stopping."
                )
        if not (
            {"min", "max"}.issubset(defined_parameters)
            or {"min", "mean"}.issubset(defined_parameters)
            or {"max", "mean"}.issubset(defined_parameters)
        ):
            raise ValueError(
                f"Distribution for {parameter} can not be defined. "
                f"For a '{getattr(path_in_config_dict, parameter).distribution}' distribution FlowNet needs "
                f"'min'/'max', 'min'/'mean' or 'mean'/'max' to be defined."
            )

    if getattr(path_in_config_dict, parameter).distribution == "truncated_normal":
        if not {"min", "max"}.issubset(_check_defined(path_in_config_dict, parameter)):
            raise ValueError(
                f"The '{getattr(path_in_config_dict, parameter).distribution}' distribution for {parameter} "
                f"requires 'min' and 'max' to be defined."
            )

    # check that mean and stddev is defined for the distributions that need it
    if getattr(path_in_config_dict, parameter).distribution in (
        "normal",
        "lognormal",
        "truncated_normal",
    ):
        if not {"mean", "stddev"}.issubset(
            _check_defined(path_in_config_dict, parameter)
        ):
            raise ValueError(
                f"The '{getattr(path_in_config_dict, parameter).distribution}' distribution for {parameter} "
                f"requires 'mean' and 'stddev' to be defined."
            )
    # check that base is defined for the distributions that need it
    if getattr(path_in_config_dict, parameter).distribution == "const":
        if not {"base"}.issubset(_check_defined(path_in_config_dict, parameter)):
            raise ValueError(
                f"The '{getattr(path_in_config_dict, parameter).distribution}' distribution for {parameter} "
                f"requires 'base' to be defined."
            )

    if getattr(path_in_config_dict, parameter).distribution == "triangular":
        if {"min", "max", "base", "mean"}.issubset(defined_parameters):
            raise ValueError(
                f"The {parameter} has values defined for 'min', 'base', 'mean' and 'max'. "
                "Only three of them should be defined"
            )
        if {"base", "mean", "max"}.issubset(defined_parameters):
            if (
                3 * getattr(path_in_config_dict, parameter).mean
                - getattr(path_in_config_dict, parameter).max
                - getattr(path_in_config_dict, parameter).base
                < 0
            ):
                raise ValueError(
                    f"The 'base', 'mean' and 'max' values for {parameter} will "
                    "give negative 'min' in the triangular distribution."
                )
        if not (
            {"min", "base", "max"}.issubset(defined_parameters)
            or {"min", "mean", "max"}.issubset(defined_parameters)
            or {"min", "base", "mean"}.issubset(defined_parameters)
            or {"max", "base", "mean"}.issubset(defined_parameters)
        ):
            raise ValueError(
                "The triangular distribution needs 'min', 'base', and 'max' to be defined. "
                "Alternatively two of the former in addition to the 'mean'."
            )


def _check_defined(path_in_config_dict: dict, parameter: str):
    """

    Args:
        path_in_config_dict (str): a location in the config schema dictionary
        parameter (str): a parameter/dictionary found at the given location

    Returns:
        The keys that are different from None in the parameter/dictionary given as input
    """
    param_dict = getattr(path_in_config_dict, parameter)._asdict()
    param_dict.pop("distribution")
    param_dict.pop("low_optimistic", None)
    return {key for key, value in param_dict.items() if value is not None}


def _check_if_all_region_priors_defined(
    path_in_config_dict: dict, unique_regions: np.ndarray, parameter_name: str
):
    """
    The function checks that prior distributions are defined for all regions is the simulation model
    if the option 'regions_from_sim' is selected for equilibration or relative permeability.

    If all regions do not have specific input for all regions in the simulation model,
    a region with default values needs to be defined.

    Args:
        path_in_config_dict: a location in the config schema dictionary
        unique_regions: a list of the unique region numbers in the input simulation model
        parameter_name: the name of the output region parameter

    Returns:
        Nothing

    Raises:
        ValueError: If there are multiple definitions of models parameters for the same region
        ValueError: If there are model parameters defined for a region that does no exist in the simulation model
        ValueError: If default values for model parameters are required but not defined
    """

    default_exists = False
    defined_regions: List[int] = []
    for reg in getattr(path_in_config_dict, "regions"):
        if reg.id is None:
            default_exists = True
        else:
            if reg.id in defined_regions:
                raise ValueError(
                    f"{parameter_name} region {reg.id} defined multiple times"
                )
            defined_regions.append(reg.id)

        if reg.id not in unique_regions and reg.id is not None:
            raise ValueError(
                f"{parameter_name} regions {reg.id} is not found in the input simulation case"
            )

    if set(defined_regions) != set(unique_regions):
        print(
            f"Values not defined for all {parameter_name} regions. Default values will be used if defined."
        )
        if not default_exists:
            raise ValueError(f"Default values for {parameter_name} regions not defined")
