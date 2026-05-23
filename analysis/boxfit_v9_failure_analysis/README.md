# BoxFit v9 Review Package

This folder contains a compact review package for why `rtv4_hgnetv2_m_cotton_fromscratch_v9_896_boxfit` underperformed the previous v6 93-line.

Files:

- `boxfit_v9_failure_analysis.md`: human-readable diagnosis and likely causes.
- `per_class_metrics_boxfit_v9.csv`: all class-level metrics and class counts.
- `class_distribution_balanced_stratified.csv`: train/val/test class distribution.
- `epoch_metrics_v9_boxfit.csv`: epoch-level aggregate metrics parsed from v9 `log.txt`.
- `run_summary_boxfit_v9.csv`: aggregate comparison among v6, v2.2 finetune, and v9.

Raw JSON/log snapshots are stored under `analysis/remote_results_raw/`, excluding checkpoint binaries.
