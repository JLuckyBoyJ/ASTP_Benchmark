# tests/llm/MCTS-AHD/

```bash
pytest tests/llm/MCTS-AHD -q                       # everything, ~5 minutes
pytest tests/llm/MCTS-AHD -q -k "not eval_prints"  # skip the slow subprocess runs, ~1 second
```

66 tests. None of them calls an LLM or costs anything. They cover the five
places this port could quietly break:

* **prompts** — every task has its four files, the signature parses with the
  exact regex `problem_adapter` uses, the seed function is valid Python, and the
  seed in `prompts/<task>/seed_func.txt` is equivalent *by AST* to the one
  shipped in `problems/<task>/gpt.py`. Those two copies drifting apart would
  make the reported baseline a different heuristic from the advertised one.
* **engines** — the ATSP matrices really are asymmetric; guided local search
  beats nearest neighbour; KGLS with `badness = cost` reduces to plain GLS and
  is handed the live penalty counters; ACO produces a valid tour and deposits
  pheromone on the traversed arc *only* (depositing on the reverse arc too would
  be correct for TSP and wrong here); a rule that returns the wrong shape, a
  NaN or an infinity is rejected rather than producing a silently wrong tour.
* **configuration** — every data config declares three splits, every task has an
  evaluation budget, `iter_limit: auto` resolves and shrinks as *n* grows, the
  GLS and KGLS budgets are *identical* (otherwise a difference between the two
  tasks would be confounded by the engine), `ATSP_DATA` and its legacy alias
  select the right file, and `configs/solvers/llm.yaml` points at files that
  exist.
* **the evaluation contract** — every `eval.py` prints a parseable objective as
  its last stdout line, since `float(stdout.split('\n')[-2])` is the only thing
  the search reads.
* **the search** — UCT survives a degenerate value range, `repr` works,
  backpropagation lifts the best child, inferior nodes are kept, progressive
  widening fires when Eq. (4) says it should, every action is recognisable from
  its prompt, and the response parser handles fenced code, unfenced code and a
  refusal without looping.

Two tests are worth knowing about because they will fail on purpose if the
repository drifts:

* `test_train_split_is_shared_with_reevo` compares this solver's training split
  against ReEvo's and fails if they diverge — the two frameworks are only
  comparable while their fitness function is the same function. It skips if the
  ReEvo solver is absent.
* `test_no_dead_modules_in_the_solver` fails when a `.py` file under
  `solvers/llm/MCTS-AHD/` is not imported by anything. Files that are reached
  some other way — `gpt.py`, which the search writes, and the LLM clients Hydra
  instantiates from a `_target_` string — are excused by name in the test, so
  adding one means saying so out loud.
