# Manuscript Results completion workflow

The heavy bootstrap and simulation stages are vectorized and distributed with `ProcessPoolExecutor`. The default configuration is tuned for a Windows workstation with 32 physical cores and 63 logical threads:

- `parallel_workers: 31`
- `simulation_batch_size: 32`
- `blas_threads_per_worker: 1`

Limiting BLAS to one thread per process prevents each worker from spawning its own large OpenMP pool.

## Full run

```powershell
python .\scripts\31_complete_manuscript_results.py `
  --config .\config\results_completion.yaml `
  --workers 31
```

## Resume from checkpoints

```powershell
python .\scripts\31_complete_manuscript_results.py `
  --config .\config\results_completion.yaml `
  --resume `
  --workers 31
```

The runner writes base checkpoints after architecture, attribution, and switching construction. It also writes separate heavy-stage checkpoints after profit-gate sensitivity, randomisation, time-shift simulation, and applied-consequence estimation. A later `--resume` run reuses every completed stage.

Progress messages report the number of process tasks and completed tasks. Time-shift is split by model, reference benchmark, and donor design, producing up to 36 independent process tasks. Randomisation and issuer-cluster inference are also distributed by model or outcome group.

The runner rebuilds the outputs committed in the locked research design: four model families, four historical architectures, three reference-state constructions, raw/normalised/signed Shapley attribution, issuer-cluster inference, common post-audit CDF ranks and categories, switching magnitudes and coverage, randomisation benchmarks, time-shift donors, all 24 applied-consequence comparisons, and three Holm-adjusted intersection-union families.

Raw FiinPro X data remain outside the repository. The runner reads `data/processed/accrual_panel.csv.gz`, writes CSV tables to `artifacts/manuscript_results`, and records a SHA-256 manifest. Manuscript values must be regenerated from this manifest rather than inferred from legacy aggregate tables.
