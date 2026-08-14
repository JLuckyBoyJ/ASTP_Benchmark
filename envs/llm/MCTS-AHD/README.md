# envs/llm/MCTS-AHD/

Dependencies for the MCTS-AHD solver, on top of `requirements.txt` at the
repository root.

```bash
# pip
pip install -r envs/llm/MCTS-AHD/requirements.txt

# conda
conda env create -f envs/llm/MCTS-AHD/mcts-ahd-atsp.yml
conda activate mcts-ahd-atsp
```

Five packages: Hydra and its colorlog plugin for the config system, `omegaconf`,
the OpenAI client, and `pyyaml` — which MCTS-AHD needs more than the other
frameworks do, because its data splits and evaluation budgets are YAML files
read at evaluation time rather than constants in the code.

Effectively the same set as `envs/llm/ReEvo/requirements.txt`, so if you already
have the ReEvo environment you can run MCTS-AHD in it.

The upstream MCTS-AHD release also pins `torch`, `botorch`, `gpytorch`,
`gymnasium` and `numba`. None are needed here: they belong to the
Bayesian-optimisation, mountain-car and JIT-compiled TSP tasks the ATSP retarget
removed, and the ATSP engines are plain NumPy. That is why this file is five
lines rather than fifty — see `solvers/llm/MCTS-AHD/README.md` §2.

## Secrets

Exactly as for EoH and ReEvo: `envs/.env` supplies `OPENAI_API_KEY`, and
MCTS-AHD reads it through Hydra's `${oc.env:OPENAI_API_KEY,null}` interpolation.
`main.py` loads that file before Hydra resolves the config, so nothing needs
exporting. A real environment variable always wins.
