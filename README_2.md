# RUL Project - Technical Runbook (In Progress)

Technical appendix to `README.md`.
Current status: iterative development, target `<=12.5` not reached yet.

## 1) Problem Setup
- Dataset: NASA C-MAPSS (`FD001..FD004`)
- Task: Remaining Useful Life regression
- Evaluation point: last cycle per test unit
- Primary KPI: macro RMSE over `FD001..FD004`
- Reliability KPIs:
  - worst-FD RMSE
  - tail RMSE / miss-rate in low-RUL region

## 2) Current Best Result
Artifact:
- `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101/hierarchical_ensemble_metrics.csv`

Metrics:
- `macro_rmse = 17.0443`
- `worst_fd_rmse = 23.4693`

Per-FD RMSE:
- `FD001=10.7157`
- `FD002=22.3907`
- `FD003=11.6014`
- `FD004=23.4693`

## 3) Current Best System Architecture
1. Global branch:
 - 5-seed median ensemble from `trial_003`
 - source: `models/seed_ensemble_global_trial003_loop2/trial_003_seed42..46`
2. FD13 specialist (`FD001/FD003`):
 - `models/tuning_fd13_stage_allpoints_multiseed_loop1/trial_001/seed_42`
3. FD24 specialist (`FD002/FD004`):
 - `models/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun/trial_002/seed_43`
4. Per-FD blend weights:
 - `FD001=0.00`
 - `FD002=0.10`
 - `FD003=0.00`
 - `FD004=0.43`

## 4) Latest Improvement Loop (What Was Executed)
Workflow:
1. FD24 rerun with long windows (`seq_len` up to 120) and stronger late/tail penalties.
2. Ranking with stability-aware score:
 - `val_rank_score = mean + 0.7 * std + 1.5 * worst_fd`
3. Cross-ensemble on top FD24 checkpoints.
4. Dense blend search (`weight-grid-size=101`).

Outcome:
- Previous best: `17.1132`
- New best: `17.0443`
- Delta: `-0.0689` macro RMSE

Comparison artifact:
- `outputs/comparison_fd24v3full_cross.csv`

## 5) Full Trajectory From LSTM (What Worked vs What Did Not)
Recommended public framing:
- `README.md`: only key milestones and current best.
- `README_2.md` + `docs/WORKLOG.md`: full path including dead branches.

Milestone path:
1. Early LSTM restart baseline:
 - global level around `~46` macro RMSE (historical start point)
2. Architecture exploration phase (no stable global breakthrough):
 - TCN / Transformer / Mamba / MoE / regime variants
 - local wins on selected FD, but no robust macro improvement
3. Methodology and feature pipeline fixes:
 - large jump to `~21` macro RMSE
4. Stability-first selection and specialist blending:
 - step to `17.xx`
5. Latest FD24-focused loop:
 - new best `17.0443`

Dead branches policy:
- Keep in repository history and `WORKLOG` (important engineering evidence).
- Do not overload the recruiter-facing main page with low-signal branch details.

## 6) Repro Commands (Latest Best Path)

### 5.1 FD24 multiseed rerun
```bash
python scripts/tune_hybrid_multifd_multiseed.py \
  --config config/tune_hybrid_multifd_fd24_stage_allpoints_v3.json \
  --seeds 42,43,44 \
  --max-trials 9 \
  --eval-test-top-k 3 \
  --std-penalty 0.7 \
  --worst-fd-penalty 1.5 \
  --shuffle-seed 20260307 \
  --output-dir outputs/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun \
  --models-root models/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun
```

### 5.2 Best cross-ensemble
```bash
python scripts/hierarchical_ensemble_multifd.py \
  --global-model-dirs "models/seed_ensemble_global_trial003_loop2/trial_003_seed42,models/seed_ensemble_global_trial003_loop2/trial_003_seed43,models/seed_ensemble_global_trial003_loop2/trial_003_seed44,models/seed_ensemble_global_trial003_loop2/trial_003_seed45,models/seed_ensemble_global_trial003_loop2/trial_003_seed46" \
  --fd13-model-dir models/tuning_fd13_stage_allpoints_multiseed_loop1/trial_001/seed_42 \
  --fd24-model-dir models/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun/trial_002/seed_43 \
  --fds FD001,FD002,FD003,FD004 \
  --weight-grid-size 101 \
  --output-dir outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101 \
  --device cuda
```

## 7) Runtime Constraints (WSL/CUDA)
- Environment: WSL2 + RTX 4060 Laptop GPU
- Memory-sensitive regime: heavy FD24 runs with `seq_len=120`
- Stable config used in latest loop:
 - `.wslconfig`: `memory=22GB`, `swap=16GB`
- Prior failure mode:
 - Linux OOM kill on Python process when memory envelope was too tight

## 8) Key Files
- Training:
 - `train_hybrid_multifd.py`
 - `scripts/tune_hybrid_multifd_multiseed.py`
- Ensembling:
 - `scripts/seed_ensemble_multifd.py`
 - `scripts/hierarchical_ensemble_multifd.py`
- Pipeline core:
 - `src/rul_pipeline/*`
- Experiment history:
 - `docs/WORKLOG.md`

## 9) Next Iteration
- Complete remaining FD24 v3 trials (`004-009`) under current memory setup.
- Re-rank + re-run cross-ensemble.
- Target for next loop: push global RMSE from `17.04` toward `<=16.x`, primarily by reducing FD004 error.
