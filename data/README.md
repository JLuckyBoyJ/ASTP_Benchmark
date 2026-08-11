# data/

Instances for the ATSP benchmark. **Both solver families read this folder**;
neither writes datasets anywhere else.

```
data/
├── raw/atsp/          19 TSPLIB ATSP instances + bestSolutions.txt   (test set)
├── synthetic/         generated training instances, cached as .npz
│   ├── uniform/
│   └── asymmetric_clustered/
├── processed/         (unused so far)
└── generate_atsp.py   builds everything under synthetic/
```

## Who reads what

| consumer | training instances | test instances |
|---|---|---|
| EoH | `data/synthetic/**` via `configs/llm/EoH/*.yaml` (`data.train`) | `data/raw/atsp` via `data.test` |
| ReEvo | `data/synthetic/**` via `TRAIN_SPLIT` in `solvers/llm/ReEvo/atsp_utils.py` | `data/raw/atsp` via `TEST_SPLIT` |

The two frameworks use **the same six cached datasets** — 8 clustered n=50,
4 clustered n=200, 4 uniform n=200, plus the three uniform n=50 sets the other
EoH tasks use. Same generator, same seeds, same files, so an EoH result and a
ReEvo result are measured on identical matrices.

```bash
python data/generate_atsp.py --all          # build every set both frameworks need
python data/generate_atsp.py --all --force  # recompute reference costs
```

## How this differs from upstream ReEvo

Upstream generates its own data inside the solver: `problems/<problem>/gen_inst.py`
writes `problems/<problem>/dataset/{train,val,test}<n>_dataset.npy`, sampled as
uniform random points in the unit square, and `eval.py` loads those `.npy`
files. That layout is gone here for three reasons:

1. ATSP instances are cost **matrices**, not 2-D point sets — a point cloud
   cannot express `d[i][j] != d[j][i]`.
2. Scoring against proven optima needs TSPLIB, which upstream never loads.
3. Keeping one copy of the data under `data/` is what lets the two frameworks be
   compared at all.

So `problems/atsp_*/gen_inst.py` does not exist; `atsp_utils.load_instances()`
reads this folder instead, and `data/generate_atsp.py` is the single generator.

## Reference costs

TSPLIB instances carry proven optima (`ref_kind="optimal"`), so gaps there are
true optimality gaps. Synthetic instances are scored against a strong
deterministic solver (`ref_kind="heuristic"`) — see the EoH README §6. A gap
against a heuristic reference can go slightly negative once a designed heuristic
beats it; on TSPLIB it cannot, and a negative value there means a bug.
