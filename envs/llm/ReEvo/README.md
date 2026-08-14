# envs/llm/ReEvo/

Extra dependencies for the ReEvo solver, on top of `requirements.txt` at the
repository root.

```bash
pip install -r envs/llm/ReEvo/requirements.txt
```

Hydra drives ReEvo's configuration and run directories; `openai` is its default
LLM client. Note `hydra-colorlog` is a Hydra **plugin**, not the plain
`colorlog` library — `cfg/config.yaml` does
`override hydra/job_logging: colorlog`, and without the plugin Hydra fails with
*"Could not find 'hydra/job_logging/colorlog'"*. Upstream's optional `numba` (GLS) and `torch` (ACO, neural solvers)
extras are **not** needed here — both ATSP engines were ported to plain NumPy,
so the repo has no compiler or GPU dependency.

Secrets come from `envs/.env` exactly as for EoH; ReEvo reads `OPENAI_API_KEY`
through Hydra's `${oc.env:OPENAI_API_KEY,null}` interpolation.
