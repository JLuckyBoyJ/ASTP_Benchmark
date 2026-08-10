# runs/

Everything an experiment produces, namespaced per solver family exactly like
`solvers/`. Git-ignored apart from the READMEs, so it is safe to let it grow.

```
runs/
├── llm/EoH/                          see runs/llm/EoH/README.md
├── slurm/                            cluster stdout/stderr
└── benchmark_tables.md               cross-family paper tables
```

`python_scripts/llm/EoH/generate_paper_tables.py` writes the top-level tables;
each family also writes its own (`runs/llm/EoH/benchmark_tables.md`).
