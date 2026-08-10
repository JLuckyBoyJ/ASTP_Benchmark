"""Make the vendored `eoh` framework importable without installing it.

The upstream package lives at ``solvers/llm/EoH/eoh/src/eoh``. Installing it
(``pip install ./solvers/llm/EoH/eoh``) is the recommended route, but importing
this module also works for a plain ``git clone``.

Importing this module additionally puts the repository root on ``sys.path`` so
that ``solvers.llm.EoH.atsp.*`` and ``evaluation.*`` resolve from anywhere.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

#: solvers/llm/EoH
EOH_DIR = os.path.dirname(_HERE)
#: solvers/llm/EoH/eoh/src  (contains the importable `eoh` package)
EOH_SRC = os.path.join(EOH_DIR, "eoh", "src")
#: repository root
REPO_ROOT = os.path.abspath(os.path.join(EOH_DIR, "..", "..", ".."))


def ensure_paths() -> None:
    for path in (EOH_SRC, REPO_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_paths()
