# Manuscript Results completion workflow

Run:

```bash
python scripts/31_complete_manuscript_results.py --config config/results_completion.yaml
```

The runner rebuilds the outputs committed in the locked research design: four model families, four historical architectures, three reference-state constructions, raw/normalised/signed Shapley attribution, issuer-cluster inference, common post-audit CDF ranks and categories, switching magnitudes and coverage, randomisation benchmarks, time-shift donors, all 24 applied-consequence comparisons, and three Holm-adjusted intersection-union families.

Raw FiinPro X data remain outside the repository. The runner reads `data/processed/accrual_panel.csv.gz`, writes CSV tables to `artifacts/manuscript_results`, and records a SHA-256 manifest. Manuscript values must be regenerated from this manifest rather than inferred from legacy aggregate tables.
