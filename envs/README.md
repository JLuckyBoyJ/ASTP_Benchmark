# envs/

Environment **specifications** and **secrets** live here; the environments
themselves do not. `.gitignore` keeps the specs, the `.env.example` template and
this README, and ignores anything you create (`envs/atsp/`, `envs/.env`), so
neither a local virtualenv nor an API key can land in git.

| path | for |
|---|---|
| `.env.example` | template for secrets and machine-local settings — **copy to `envs/.env`** |
| `.env` | your real credentials (git-ignored, never committed) |
| `llm/EoH/eoh-atsp.yml` | conda spec for the EoH + ATSP stack (Python 3.11) |
| `llm/EoH/eoh-atsp-pip.txt` | pinned pip requirements, for `venv` users and CI |
| `llm/<family>/requirements.txt` | pip deps for ReEvo, HSEvo, MCTS-AHD and MoH |
| `llm/<family>/*.yml` | the matching conda specs |

## Secrets

```bash
cp envs/.env.example envs/.env
$EDITOR envs/.env            # set OPENAI_API_KEY=sk-...
```

`envs/.env` is loaded automatically by every Python entry point and by the shell
scripts — nothing else to do. Precedence, highest first:

1. a real environment variable (`OPENAI_API_KEY=sk-x python ...`)
2. `$ENV_FILE` if you set it (handy for several key sets)
3. `envs/.env`
4. `.env` at the repository root

Keys are read at run time and **never written to disk**: the `config.yaml`
snapshot saved in each run directory stores `<redacted:N chars>` instead, and
`meta.json` records only *which* `.env` file was used.

Anything else in `.env` is an ordinary shell variable — reference it explicitly
when you want it, e.g. `--model "$EOH_MODEL"` or `REPEATS=$REPEATS`.

## Quick start

```bash
# conda
bash scripts/setup_environment.sh            # after: conda activate eoh-atsp
CONDA=1 bash scripts/setup_environment.sh    # create the env first

# venv
VENV=envs/atsp bash scripts/setup_environment.sh
source envs/atsp/bin/activate
```

`scripts/setup_environment.sh` installs the requirements *and* the vendored EoH
package, generates the synthetic training instances and runs the test suite, so
prefer it over installing by hand.

## Why the EoH package is installed separately

`solvers/llm/EoH/eoh` is the upstream ICML-2024 framework, kept vendored and
unmodified so it can be diffed against the original release. Installing it
editable (`pip install -e ./solvers/llm/EoH/eoh`) makes `import eoh` work
everywhere; if you skip that, `solvers/llm/EoH/atsp/_bootstrap.py` puts it on
`sys.path` at import time, so a plain `git clone` still runs.
