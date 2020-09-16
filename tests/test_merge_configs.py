from flownet.config_parser._merge_configs import merge_configs


def test_merge_configs() -> None:

    base = {
        "some_key": 42,
        "some_nested_value": {"a": 1, "b": 2, "c": {"d": 3, "e": 4}},
    }
    update = {"some_key": 3, "some_nested_value": {"b": 999, "c": {"e": None}}}

    expected = {
        "some_key": 3,
        "some_nested_value": {"a": 1, "b": 999, "c": {"d": 3, "e": None}},
    }

    assert merge_configs(base, update) == expected
