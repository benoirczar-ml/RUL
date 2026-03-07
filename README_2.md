# README_2 - Technical Appendix & Engineering Notes

This document is the technical counterpart to `README.md`.
It includes exact artifacts, iteration logic, and practical constraints.

## 1) Best Known Checkpoint (Global)
Artifact:
- `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101/hierarchical_ensemble_metrics.csv`

Metrics:
- `macro_rmse = 17.0443`
- `worst_fd_rmse = 23.4693`

Per-FD:
- `FD001 = 10.7157`
- `FD002 = 22.3907`
- `FD003 = 11.6014`
- `FD004 = 23.4693`

## 2) Main System Topology
- Global branch: 5-seed ensemble (`trial_003`, seeds `42..46`)
- FD13 specialist:
  - `models/tuning_fd13_stage_allpoints_multiseed_loop1/trial_001/seed_42`
- FD24 specialist:
  - `models/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun/trial_002/seed_43`
- Per-FD blend weights (best run):
  - `FD001=0.00`
  - `FD002=0.10`
  - `FD003=0.00`
  - `FD004=0.43`

## 3) Progression Snapshot
Source table:
- `docs/tables/summary_results.csv`

Key milestones:
- baseline: `46.2155`
- single model best: `20.8845`
- early hierarchical: `17.2135`
- current best hierarchical: `17.0443`
- variance meta-stacker experiment: `17.5447`

Interpretation:
- Most gain came from architecture/pipeline structuring + specialists, not one-off model swaps.
- Variance-aware meta-stacker (loop1) improved validation shape but overfit on full test.

## 4) FD004 v4 Loop Status
Artifact:
- `outputs/tuning_fd004_v4_tail_longwindow_multiseed_loop1/hybrid_multifd_multiseed_agg.csv`

Best by rank (`mean + 0.7*std + 1.5*worst_fd`):
- `trial_007`
- `val_selection_mean = 15.2900`
- `val_selection_std = 0.2387`

Cross-ensemble tests with FD004 override:
- `trial_007/seed_42`: `macro 17.1083`
- `trial_007/seed_43`: `macro 17.0905`
- `trial_007/seed_44`: `macro 17.1592`

All were worse than `17.0443`.

## 5) Deep Ensemble + Variance Meta-Stacker Loop (Latest)
New script:
- `scripts/meta_stacker_multifd_variance.py`

Output:
- `outputs/meta_stacker_variance_loop1/meta_stacker_variance_metrics.csv`
- `outputs/meta_stacker_variance_loop1/meta_stacker_variance_state.json`

Result:
- `macro = 17.5447`
- `worst_fd = 23.4783`

Why not promoted:
- worse macro than best hierarchical by `+0.5005`
- worst-FD essentially unchanged

## 6) Visualization Assets
Generated from:
- `scripts/generate_readme_assets.py`

Figures:
- `docs/figures/rmse_progression.png`
- `docs/figures/per_fd_comparison.png`
- `docs/figures/fd004_pred_vs_true.png`
- `docs/figures/fd004_error_hist.png`
- `docs/figures/fd004_uncertainty_vs_error.png`

Tables:
- `docs/tables/summary_results.csv`
- `docs/tables/best_per_fd.csv`
- `docs/tables/ablation_lift_summary.csv`
- `docs/tables/fd004_best_errors.csv`
- `docs/tables/fd004_uncertainty_vs_error.csv`

## 7) Known Bottlenecks
- FD002 and FD004 still dominate the global macro.
- FD004-only validation gains do not reliably transfer to full-system blend.
- Additional model complexity near plateau often improves val but not test macro.

## 8) What Could Still Move the Needle (Without Physics-Informed Block)
1. FD002 v4 specialist loop (longer context + stronger tail/late penalties + strict multi-seed ranking).
2. Better specialist routing / gating for FD24 split into FD002 vs FD004 at inference time.
3. More conservative stacker with stronger fold-based validation and fallback-on-baseline per FD.

## 9) Practical Stop Condition
At this stage, expected gains are marginal relative to added complexity/runtime.
Repository is left in a reproducible state with complete logs and visuals.
