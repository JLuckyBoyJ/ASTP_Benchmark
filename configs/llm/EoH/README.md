# configs/llm/EoH/

One YAML per ATSP task, all inheriting from `base.yaml` via `extends`.

| file | task | LLM designs |
|---|---|---|
| `base.yaml` | — | LLM, EC, logging and data defaults shared by all tasks |
| `atsp_construct.yaml` | `construct` | `select_next_node` — greedy tour construction |
| `atsp_gls.yaml` | `gls` | `update_edge_distance` — guided local search landscape |
| `atsp_aco.yaml` | `aco` | `update_pheromone` — directed pheromone rule |
| `atsp_rnr.yaml` | `rnr` | `destroy_nodes` — ruin-and-recreate destroy operator |

## Structure of a config

```yaml
llm:   # endpoint, model (gpt-4o-mini), api key from ${OPENAI_API_KEY}, timeout
eoh:   # population size, generations, operators, parallelism  (paper defaults)
run:   # output_root (runs/llm/EoH), tag, LLM tracing
task:  # which task, its per-evaluation budget, and the evaluation timeout
data:  # train (synthetic) and test (TSPLIB) splits
eval:  # the larger budget used when benchmarking on the test set
```

## Training splits can mix families and sizes

`data.train` accepts either one spec or a **list** of specs whose instances are
concatenated:

```yaml
data:
  train:
    - {source: synthetic, family: asymmetric_clustered, size: 50,  count: 8, effort: medium}
    - {source: synthetic, family: asymmetric_clustered, size: 200, count: 2, effort: low}
    - {source: synthetic, family: uniform,              size: 200, count: 2, effort: low}
```

This is what `atsp_gls.yaml` ships with, for two measured reasons.

**TSPLIB ATSP is not one distribution.** The `ftv`/`kro` instances are
near-symmetric with a directional perturbation — corr(d[i,j], d[j,i]) between
0.6 and 1.0, which `asymmetric_clustered` reproduces — while the `rbg`
instances have essentially independent directions (corr ≈ 0.03, which
`uniform` reproduces). A heuristic evolved on one family alone wins on that
half of the benchmark and loses on the other.

**Cost only matters when the clock binds.** With `ite_max` capping the search,
an update rule costing O(n²) per call is free at n=50 and ruinous at n=443.
Setting `ite_max` very high makes `time_limit` the only budget, so the fitness
measures tour quality *per second* — and the large training instances make the
search feel it.

## Overriding

Nothing needs to be edited to change a setting — pass `--set key.path=value`.
A numeric path element indexes into a list:

```bash
# change one entry of a mixed split
--set data.train.0.count=16

# replace the whole split with a single-family one
--set 'data.train=[{source: synthetic, family: asymmetric_clustered, size: 50, count: 16}]'
```

More examples:

```bash
# shorter pilot run
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls \
  --set eoh.n_pop=5 --set eoh.pop_size=4

# paper-scale GLS: 64 training instances of 100 cities, 10 s per instance
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls \
  --set data.train.size=100 --set data.train.count=64 \
  --set task.params.time_limit=10 --set task.params.ite_max=1000

# a different provider
python python_scripts/llm/EoH/run_eoh_atsp.py --task aco \
  --set llm.api_endpoint=api.deepseek.com --set llm.model=deepseek-chat

# a local model
python python_scripts/llm/EoH/run_eoh_atsp.py --task rnr \
  --set llm.use_local=true --set llm.local_url=http://127.0.0.1:11012/completions
```

The resolved config — after `extends`, env substitution and every `--set` — is
written to `config.yaml` inside the run directory, with the API key redacted, so
a run can always be reproduced from its own output.
