# envs/llm/EoH/

Environment specs for running EoH on the ATSP benchmark.

| file | use |
|---|---|
| `eoh-atsp.yml` | `conda env create -f envs/llm/EoH/eoh-atsp.yml` |
| `eoh-atsp-pip.txt` | `pip install -r envs/llm/EoH/eoh-atsp-pip.txt` |

The stack is deliberately small — NumPy, PyYAML, joblib, requests, pytest — so
runs are reproducible and easy to stand up on a cluster node. EoH itself is
vendored in `solvers/llm/EoH/eoh` and installed on top:

```bash
conda env create -f envs/llm/EoH/eoh-atsp.yml
conda activate eoh-atsp
pip install -e ./solvers/llm/EoH/eoh
python data/generate_atsp.py --all
pytest tests -q
```

Nothing here pins an LLM SDK: EoH talks to any OpenAI-compatible endpoint over
plain HTTPS, so only the `OPENAI_API_KEY` environment variable is needed.

To pin an exact environment for a paper run:

```bash
pip freeze > envs/llm/EoH/eoh-atsp-pip.txt
```

Each run records the interpreter and NumPy versions it used in its own
`meta.json`, so a result can always be traced back to an environment.
