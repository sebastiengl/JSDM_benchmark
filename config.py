import ast
from argparse import Namespace
from dataclasses import dataclass
from configobj import ConfigObj


def _convert_value(key, value):
    if isinstance(value, list):
        if key == "label":
            return ", ".join(value)
        return [_convert_value(key, item) for item in value]
    try:
        return ast.literal_eval(value.strip())
    except (ValueError, SyntaxError):
        return value.strip()


def _convert_section(section):
    return {key: _convert_value(key, value) if not isinstance(value, dict)
            else _convert_section(value) for key, value in section.items()}


def load_config(path="config.cfg"):
    """Load config.cfg using ConfigObj and return typed nested dictionaries."""
    return _convert_section(ConfigObj(path, interpolation=False, list_values=True,
                                      raise_errors=True))


def as_namespace(values):
    """Convert a config section to the attribute interface used by the scripts."""
    return Namespace(**{key.replace("-", "_"): value for key, value in values.items()})
