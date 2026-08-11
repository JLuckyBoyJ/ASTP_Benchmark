"""ATSP instance loading and generation."""

from .instance import ATSPInstance
from .tsplib import load_best_known, load_tsplib_atsp, parse_atsp_file
from .synthetic import (
    FAMILIES,
    generate_instance,
    generate_dataset,
    load_dataset,
    save_dataset,
    reference_cost,
)
from .registry import describe_split, resolve_split

__all__ = [
    "ATSPInstance",
    "parse_atsp_file",
    "load_best_known",
    "load_tsplib_atsp",
    "FAMILIES",
    "generate_instance",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
    "reference_cost",
    "resolve_split",
    "describe_split",
]
