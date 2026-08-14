# data/

Instances for the ATSP benchmark. **All three solver families read this folder**;
none of them writes datasets anywhere else.

```
data/
├── raw/atsp/          19 TSPLIB ATSP instances + bestSolutions.txt   (test set)
├── synthetic/         generated training instances, cached as .npz
│   ├── uniform/
│   ├── asymmetric_clustered/
│   ├── scheduling_constrained/
│   └── stacker_crane/
├── cache/             gls_calibration.json — iterations/second on this machine
├── processed/         (unused so far)
└── generate_atsp.py   builds every set the EoH configs reference
```

## Who reads what

| consumer | training instances | test instances | declared in |
|---|---|---|---|
| EoH | `data/synthetic/**` | `data/raw/atsp` | `configs/llm/EoH/*.yaml` → `data.train` / `data.test` |
| ReEvo | `data/raw/atsp` (rbg323, rbg403) | `data/raw/atsp` | `solvers/llm/ReEvo/atsp_utils.py` → `TRAIN_SPLIT` / `TEST_SPLIT` |
| MCTS-AHD | depends on the data config | `data/raw/atsp` | `configs/llm/MCTS-AHD/cfg/data/*.yaml` |

MCTS-AHD is the only one whose split is a config file rather than a constant, so
it can be switched per run:

```bash
cd solvers/llm/MCTS-AHD
python main.py problem=atsp_gls data=tsplib            # default, = ReEvo's split
python main.py problem=atsp_gls data=synthetic         # no TSPLIB instance is seen
python main.py problem=atsp_gls data=mcts_ahd_native   # upstream's protocol
```

## Two generators, one cache

Both write the same `data/synthetic/<family>/atsp_<family>_n<size>_c<count>_s<seed>.npz`
files, so a set is computed once no matter who asks for it — and the expensive
part is the reference cost, not the matrix.

```bash
python data/generate_atsp.py --all                        # everything EoH needs
python data/generate_atsp.py --all --force                # recompute references

python python_scripts/llm/MCTS-AHD/prepare_data.py --config synthetic
python python_scripts/llm/MCTS-AHD/prepare_data.py --all  # every MCTS-AHD config
```

`prepare_data.py` builds only what one MCTS-AHD data config references, so adding
a config there never lengthens anybody else's setup. A TSPLIB-only config prints
"nothing to generate" and exits.

## How MCTS-AHD's original data handling maps onto this folder

Upstream MCTS-AHD generates its data **inside the solver**: each task ships a
`problems/<task>/gen_inst.py` that writes
`problems/<task>/dataset/{train,val,test}<n>_dataset.npy` — uniform random points
in the unit square — and `eval.py` loads the `.npy` matching the mood and the
configured `problem_size`. The protocol is:

| split | upstream (`tsp_constructive`) | here |
|---|---|---|
| train | 64 instances, one size (n=50) | 2 TSPLIB rbg instances, or 8–16 generated matrices |
| val | 64 instances, sizes 20 / 50 / 100 / 200 | the ≤100-city TSPLIB instances, or generated matrices at two sizes |
| test | 1000 instances, sizes 20 … 1000 | the 19 TSPLIB instances with proven optima |

Three things make a literal copy impossible, and all three are about ATSP rather
than about MCTS-AHD:

1. **An ATSP instance is a cost matrix, not a point set.** A 2-D point cloud
   cannot express `d[i][j] != d[j][i]`, so arrays of coordinates are replaced by
   cached `.npz` matrices.
2. **Scoring against proven optima needs TSPLIB**, which upstream never loads.
   1000 random instances have no reference cost, so a gap cannot be computed
   against them — only raw tour length, which is not comparable across sizes.
3. **An ATSP evaluation runs real local search**, not a greedy pass over 50
   points. 64 training instances at 10 s each would be ten minutes per candidate
   heuristic, and a run evaluates a hundred of them.

What *is* preserved, in `configs/llm/MCTS-AHD/cfg/data/mcts_ahd_native.yaml`: a
single-size training set of many small instances, a multi-size validation set,
and a test set the search never sees. Use it when you want upstream's protocol;
use the default `tsplib` config when you want the number to be comparable with
the ReEvo runs in this repo.

## Reference costs

TSPLIB instances carry proven optima (`ref_kind="optimal"`), so gaps there are
true optimality gaps. Synthetic instances are scored against a strong
deterministic solver (`ref_kind="heuristic"`) — see the EoH README §6. A gap
against a heuristic reference can go slightly negative once a designed heuristic
beats it; on TSPLIB it cannot, and a negative value there means a bug.

## Calibration

`data/cache/gls_calibration.json` records how many local-search iterations one
second buys on this machine, by instance size. The improvement tasks bound
*training* by iterations (so a score never depends on machine load) and the
*benchmark* by wall clock; those two only agree if the iteration count is chosen
from this machine's speed. Regenerate with:

```bash
bash scripts/llm/MCTS-AHD/calibrate.sh
```

The file is shared with the ReEvo side, which reads the same numbers.
