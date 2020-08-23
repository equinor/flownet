import os
import pathlib
from typing import Dict, Optional, List

import yaml
import configsuite
from configsuite import types, MetaKeys as MK, ConfigSuite


def create_schema(_to_abs_path) -> Dict:
    """
    Returns a configsuite type schema, where configuration value types are defined, together
    with which values are optional and/or has default values.

    Args:
        _to_abs_path: Use absolute path transformation

    Returns:
        Dictionary to be used as configsuite type schema

    """
    return {
        MK.Type: types.NamedDict,
        MK.Content: {
            "name": {MK.Type: types.String},
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
                                    },
                                    "vectors": {
                                        MK.Type: types.NamedDict,
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
                        MK.Transformation: lambda names: [x.lower() for x in names],
                    },
                    "cell_length": {MK.Type: types.Number},
                    "training_set_end_date": {MK.Type: types.Date, MK.AllowNone: True,},
                    "training_set_fraction": {
                        MK.Type: types.Number,
                        MK.AllowNone: True,
                    },
                    "additional_flow_nodes": {MK.Type: types.Integer, MK.Default: 100,},
                    "additional_node_candidates": {
                        MK.Type: types.Integer,
                        MK.Default: 1000,
                    },
                    "hull_factor": {MK.Type: types.Number, MK.Default: 1.2},
                    "random_seed": {MK.Type: types.Number},
                    "perforation_handling_strategy": {
                        MK.Type: types.String,
                        MK.Default: "bottom_point",
                    },
                    "fast_pyscal": {MK.Type: types.Bool, MK.Default: True},
                    "fault_tolerance": {MK.Type: types.Number, MK.Default: 1.0e-5},
                },
            },
            "ert": {
                MK.Type: types.NamedDict,
                MK.Content: {
                    "runpath": {
                        MK.Type: types.String,
                        MK.Default: "output/runpath/realization-%d/iter-%d",
                    },
                    "enspath": {MK.Type: types.String, MK.Default: "output/storage",},
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
                            "num_realizations": {MK.Type: types.Integer},
                            "required_success_percent": {
                                MK.Type: types.Number,
                                MK.Default: 20,
                            },
                            "max_runtime": {MK.Type: types.Integer, MK.Default: 300,},
                        },
                    },
                    "simulator": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "name": {
                                MK.Type: types.String,
                                MK.Transformation: lambda name: name.lower(),
                                MK.Default: "flow",
                            },
                            "version": {MK.Type: types.String, MK.AllowNone: True},
                        },
                    },
                    "queue": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "system": {MK.Type: types.String},
                            "name": {MK.Type: types.String},
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
                    },
                    "analysis": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "metric": {MK.Type: types.String, MK.Default: "[RMSE]"},
                            "quantity": {
                                MK.Type: types.String,
                                MK.Default: "[WOPR:BR-P-]",
                            },
                            "start": {MK.Type: types.String, MK.Default: "2001-04-01",},
                            "end": {MK.Type: types.String, MK.Default: "2006-01-01",},
                            "outfile": {
                                MK.Type: types.String,
                                MK.Default: "analysis_metrics_iteration",
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
                        MK.Content: {
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "mean": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number},
                            "loguniform": {MK.Type: types.Bool, MK.Default: True},
                        },
                    },
                    "porosity": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "mean": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number},
                            "loguniform": {MK.Type: types.Bool, MK.Default: False},
                        },
                    },
                    "bulkvolume_mult": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "mean": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number},
                            "loguniform": {MK.Type: types.Bool, MK.Default: True},
                        },
                    },
                    "fault_mult": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "mean": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number, MK.AllowNone: True},
                            "loguniform": {MK.Type: types.Bool, MK.AllowNone: True},
                        },
                    },
                    "relative_permeability": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "scheme": {
                                MK.Type: types.String,
                                MK.Transformation: lambda name: name.lower(),
                            },
                            "swirr": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "swl": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "swcr": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
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
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "krowend": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "nw": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "now": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "sorg": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "sgcr": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "ng": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                },
                            },
                            "nog": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "krgend": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "loguniform": {
                                        MK.Type: types.Bool,
                                        MK.AllowNone: True,
                                    },
                                },
                            },
                            "krogend": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
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
                                MK.Default: "global",
                                MK.Transformation: lambda name: name.lower(),
                            },
                            "datum_depth": {MK.Type: types.Number, MK.AllowNone: True,},
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
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                },
                            },
                            "gwc_depth": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                },
                            },
                            "goc_depth": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
                                },
                            },
                        },
                    },
                    "rock_compressibility": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "reference_pressure": {
                                MK.Type: types.Number,
                                MK.AllowNone: True,
                            },
                            "min": {MK.Type: types.Number, MK.AllowNone: True},
                            "max": {MK.Type: types.Number, MK.AllowNone: True},
                        },
                    },
                    "aquifer": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "scheme": {MK.Type: types.String, MK.AllowNone: True},
                            "fraction": {MK.Type: types.Number, MK.AllowNone: True},
                            "delta_depth": {MK.Type: types.Number, MK.AllowNone: True,},
                            "size_in_bulkvolumes": {
                                MK.Type: types.NamedDict,
                                MK.Content: {
                                    "min": {MK.Type: types.Number, MK.AllowNone: True,},
                                    "max": {MK.Type: types.Number, MK.AllowNone: True,},
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
    # pylint: disable=too-many-branches

    input_config = yaml.safe_load(configuration_file.read_text())

    @configsuite.transformation_msg("Tries to convert input to absolute path")
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
        return str((configuration_file.parent / pathlib.Path(path)).resolve())

    suite = ConfigSuite(
        input_config, create_schema(_to_abs_path=_to_abs_path), deduce_required=True
    )

    if not suite.valid:
        raise ValueError(
            "The configuration is not valid:"
            + ", ".join([error.msg for error in suite.errors])
        )

    config = suite.snapshot

    req_relp_parameters: List[str] = []

    for phase in config.flownet.phases:
        if phase not in ["oil", "gas", "water", "disgas", "vapoil"]:
            raise ValueError(
                f"The {phase} phase is not a valid phase\n"
                f"The valid phases are 'oil', 'gas', 'water', 'disgas' and 'vapoil'"
            )
    if not {"vapoil", "disgas"}.isdisjoint(config.flownet.phases) and not {
        "oil",
        "gas",
    }.issubset(config.flownet.phases):
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
        if (
            config.model_parameters.equil.owc_depth.min is None
            or config.model_parameters.equil.owc_depth.max is None
            or config.model_parameters.equil.owc_depth.max
            < config.model_parameters.equil.owc_depth.min
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
        if (
            config.model_parameters.equil.goc_depth.min is None
            or config.model_parameters.equil.goc_depth.max is None
            or config.model_parameters.equil.goc_depth.max
            < config.model_parameters.equil.goc_depth.min
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
            ):
                raise ValueError(
                    f"The relative permeability scheme "
                    f"'{config.model_parameters.relative_permeability.scheme}' is not valid.\n"
                    f"Valid options are 'global' or 'individual'."
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
        if (
            not pathlib.Path(config.flownet.data_source.simulation.input_case)
            .with_suffix(suffix)
            .is_file()
        ):
            raise ValueError(f"The input case {suffix} file does not exist")

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
