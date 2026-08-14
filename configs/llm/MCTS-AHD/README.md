# configs/llm/MCTS-AHD/

MCTS-AHD's whole configuration, moved out of the solver folder so every config
in the repository lives under `configs/`. `main.py` points Hydra here with
`config_path="../../../configs/llm/MCTS-AHD/cfg"`.

```
configs/llm/MCTS-AHD/cfg/
├── config.yaml                     search budget, the paper's MCTS parameters, data choice
├── problem/atsp_constructive.yaml  greedy construction (LLM designs select_next_node)
├── problem/atsp_gls.yaml           guided local search (LLM designs a static guide matrix)
├── problem/atsp_kgls.yaml          knowledge-guided LS  (LLM designs a dynamic badness rule)
├── problem/atsp_aco.yaml           ant colony          (LLM designs the desirability matrix)
├── data/tsplib.yaml                default split: evolve on rbg323+rbg403, report on all 19
├── data/synthetic.yaml             evolve on generated matrices, no benchmark instance seen
├── data/mcts_ahd_native.yaml       upstream's train/val/test protocol, translated to ATSP
├── evaluation/atsp_*.yaml          per-task engine budgets (iterations, ants, time limits)
├── llm_client/openai.yaml          model: gpt-4o-mini
├── llm_client/litellm.yaml         any LiteLLM-supported provider
├── llm_client/stub.yaml            offline fake model, for plumbing checks
└── hydra/output/local.yaml         run dir -> runs/llm/MCTS-AHD/<problem>/<timestamp>/
```

## Two config groups the EoH and ReEvo sides do not have

`data/` and `evaluation/` exist because of a problem every Hydra-driven AHD
framework has and neither upstream solves: **`eval.py` runs as a subprocess with
no Hydra context**. Upstream's answer is to hardcode the split and the budgets
as module-level constants at the top of each evaluation script, which is why in
ReEvo the training instances live in `atsp_utils.py` and the iteration limits at
the top of `problems/*/eval.py`.

Here they are YAML, read by `solvers/llm/MCTS-AHD/atsp_utils.py:load_config()`
with plain PyYAML. `problem_adapter` forwards the choice to each subprocess as
`ATSP_DATA` and `ATSP_EVAL_BUDGET`, so:

* a split can be changed per run (`data=synthetic`) instead of by editing code;
* the *reported* set is the same config's `test` split, so what you evolve on
  and what you report against cannot drift apart;
* the benchmark runner in `evaluation/llm/MCTS-AHD/` reads the same budget file
  as the task's own `eval.py`, so a post-hoc score and a run's own validation
  mean the same thing by construction.

## Overrides

From `solvers/llm/MCTS-AHD`:

```bash
python main.py problem=atsp_gls                        # pick the task
python main.py problem=atsp_kgls data=synthetic        # pick the split
python main.py problem=atsp_aco max_fe=1000            # the paper's budget
python main.py problem=atsp_gls init_pop_size=4 pop_size=10
python main.py problem=atsp_gls llm_client.model=gpt-4o
python main.py problem=atsp_gls llm_client=stub        # no API calls at all
python main.py problem=atsp_gls problem.timeout=120    # per-evaluation kill
```

Or from the repository root, which is usually easier:

```bash
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls \
    --max-fe 200 --data synthetic --set exploration_constant=0.05
```

## Which knob is which symbol in the paper

| config key | paper | default | what it does |
|---|---|---|---|
| `max_fe` | *T* | 100 | heuristic evaluations the search may spend |
| `init_pop_size` | *N_I* | 4 | initial tree nodes hung off the virtual root |
| `pop_size` | \|*E*\| | 10 | the elite set action **e2** samples its reference from |
| `exploration_constant` | *λ₀* | 0.1 | UCT exploration, decayed linearly by Eq. (7) |
| `progressive_widening_alpha` | *α* | 0.5 | widen when ⌊N(n)^α⌋ ≥ \|children(n)\| |
| `expansion_children` | *k* | 2 | an expansion makes 2k+2 children |
| `max_tree_depth` | — | 10 | depth cap on selection |
| `crossover_parents` | *m* | 5 | upper bound on parents sampled for action **e1** |
| `seed` | — | 2024 | seeds node selection |
| `data` | — | tsplib | which `cfg/data/*.yaml` split to use |
| `timeout` | — | per task | hard kill for one heuristic evaluation |
| `debug` | — | false | log every prompt and every extracted heuristic |

The paper's own default for `max_fe` is 1000. This repo defaults to 100 to match
`configs/llm/ReEvo/cfg/config.yaml`, because the interesting question is whether
the tree search spends a *fixed* budget better than a population does.
