"""Configuration loading for ATSP EoH runs.

A config is a YAML file under ``configs/llm/EoH/``. Files may ``extends`` another
config (resolved relative to the file itself), values may reference environment
variables as ``${VAR}`` or ``${VAR:-fallback}``, and any key can be overridden
on the command line with ``--set eoh.n_pop=5``.

Secrets never live in the YAML. API keys are read from the environment, which is
populated from ``envs/.env`` (git-ignored) if that file exists — see
``envs/.env.example``.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

#: Searched in order, relative to the repository root. The first match wins for
#: any given variable; real environment variables always take precedence.
DOTENV_CANDIDATES = (os.path.join("envs", ".env"), ".env")

#: Fallback values for anything the YAML omits.
DEFAULTS: dict[str, Any] = {
    "llm": {
        "api_endpoint": "api.openai.com",
        "api_key": None,
        "model": "gpt-4o-mini",
        "timeout": 150,
        "use_local": False,
        "local_url": None,
    },
    "eoh": {
        "pop_size": 10,
        "n_pop": 20,
        "n_parents": 5,
        "operators": ["e1", "e2", "m1", "m2"],
        "operator_weights": None,
        "num_samplers": 4,
        "num_evaluators": 4,
        "max_sample_nums": None,
        "debug": False,
        "use_seed": False,
        "seed_path": None,
        "use_continue": False,
        "continue_path": None,
        "continue_id": 0,
    },
    "run": {
        "output_root": "runs/llm/EoH",
        "tag": "",
        "seed": 2024,
        "trace_llm": True,
    },
    "task": {
        "name": "construct",
        "timeout": 60,
        "params": {},
    },
    "data": {
        # A split is either one spec or a list of specs whose instances are
        # concatenated (see atsp/data/registry.py).
        "train": {"source": "synthetic", "family": "uniform", "size": 50,
                  "count": 8, "seed": 2024, "effort": "medium", "path": None},
        "test": {"source": "tsplib", "dir": "data/raw/atsp",
                 "best_known": "data/raw/atsp/bestSolutions.txt",
                 "max_n": None, "min_n": None, "names": None},
    },
    "eval": {
        "params": {},
    },
}


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env`` content into a dict.

    Supports ``KEY=value``, a leading ``export``, ``#`` comments, blank lines and
    single/double quoted values. Deliberately does not expand ``$VAR`` — a key
    containing ``$`` must survive verbatim.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key.isidentifier():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(root: str | None = None, path: str | None = None) -> list[str]:
    """Populate ``os.environ`` from ``envs/.env`` (or ``$ENV_FILE``).

    Existing environment variables are never overwritten, so
    ``OPENAI_API_KEY=... python ...`` still wins over the file.
    Returns the list of files that were read.
    """
    root = root or repo_root()
    candidates = []
    if path:
        candidates.append(path)
    elif os.environ.get("ENV_FILE"):
        candidates.append(os.environ["ENV_FILE"])
    else:
        candidates.extend(os.path.join(root, name) for name in DOTENV_CANDIDATES)

    loaded = []
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                pairs = parse_dotenv(fh.read())
        except OSError:
            continue
        for key, value in pairs.items():
            os.environ.setdefault(key, value)
        loaded.append(candidate)
    return loaded


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name, fallback = match.group(1), match.group(2)
            return os.environ.get(name, fallback if fallback is not None else "")
        expanded = _ENV_PATTERN.sub(repl, value)
        return None if expanded == "" and _ENV_PATTERN.search(value) else expanded
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml_with_extends(path: str, _seen: set[str] | None = None) -> dict:
    path = os.path.abspath(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular 'extends' chain at {path}")
    _seen.add(path)

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    parent = data.pop("extends", None)
    if parent:
        parent_path = parent if os.path.isabs(parent) else os.path.join(
            os.path.dirname(path), parent)
        data = deep_merge(_read_yaml_with_extends(parent_path, _seen), data)
    return data


def _coerce(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_override(config: dict, assignment: str) -> dict:
    """Apply a single ``a.b.c=value`` override in place.

    A numeric path element indexes into a list, so an entry of a multi-spec
    data split can be targeted individually::

        --set data.train.0.count=16      # first spec of the training split
    """
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    key, _, raw = assignment.partition("=")
    parts = key.strip().split(".")
    value = _coerce(raw.strip())

    node = config
    for depth, part in enumerate(parts[:-1]):
        nxt = parts[depth + 1]
        if isinstance(node, list):
            node = node[_list_index(node, part, assignment)]
            continue
        if not isinstance(node, dict):
            raise ValueError(f"--set {assignment!r}: {part!r} is not a section")
        if part not in node or node[part] is None:
            node[part] = [] if nxt.lstrip("-").isdigit() else {}
        node = node[part]

    last = parts[-1]
    if isinstance(node, list):
        node[_list_index(node, last, assignment)] = value
    elif isinstance(node, dict):
        node[last] = value
    else:
        raise ValueError(f"--set {assignment!r}: {'.'.join(parts[:-1])!r} is not a section")
    return config


def _list_index(node: list, part: str, assignment: str) -> int:
    if not part.lstrip("-").isdigit():
        raise ValueError(
            f"--set {assignment!r}: {part!r} indexes a list, so it must be a number "
            f"(list has {len(node)} entries)")
    index = int(part)
    if not -len(node) <= index < len(node):
        raise ValueError(
            f"--set {assignment!r}: index {index} is out of range "
            f"(list has {len(node)} entries)")
    return index


def load_config(path: str, overrides: list[str] | None = None,
                use_dotenv: bool = True) -> dict:
    """Read a config file, apply defaults, env expansion and CLI overrides."""
    dotenv_files = load_dotenv() if use_dotenv else []
    config = deep_merge(DEFAULTS, _read_yaml_with_extends(path))
    for assignment in overrides or []:
        apply_override(config, assignment)
    config = _expand_env(config)
    config["_config_path"] = os.path.abspath(path)
    config["_dotenv"] = dotenv_files
    return config


def dump_config(config: dict) -> str:
    """YAML text of a resolved config with the API key redacted."""
    safe = copy.deepcopy(config)
    key = safe.get("llm", {}).get("api_key")
    if key:
        safe["llm"]["api_key"] = f"<redacted:{len(str(key))} chars>"
    return yaml.safe_dump(safe, sort_keys=False, allow_unicode=True)


def repo_root() -> str:
    """Absolute path of the repository root (three levels above EoH/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
