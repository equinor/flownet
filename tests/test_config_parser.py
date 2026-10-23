import pathlib

import pytest

from flownet.config_parser import parse_config
from flownet.config_parser._config_parser import _normalize_analysis_entries

CONFIG_FOLDER = pathlib.Path(__file__).resolve().parent / "configs"


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        parse_config(CONFIG_FOLDER / "missing_arguments.yml")


def test_normalize_single_analysis_entry() -> None:
    config = {"ert": {"analysis": {"metric": ["RMSE"]}}}

    assert _normalize_analysis_entries(config) == {
        "ert": {"analysis": [{"metric": ["RMSE"]}]}
    }
