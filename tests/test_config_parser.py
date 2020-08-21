import pathlib

import pytest

from flownet.config_parser import parse_config


CONFIG_FOLDER = pathlib.Path(__file__).resolve().parent / "configs"


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        parse_config(CONFIG_FOLDER / "missing_arguments.yml")
