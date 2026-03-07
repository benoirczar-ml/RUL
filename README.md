# Remaining Useful Life (NASA C-MAPSS)

## Story
Ten projekt to historia zejscia z poziomu "dziala, ale za slabo" do poziomu realnego postepu inzynierskiego.

Punkt startowy byl okolo `46-48` RMSE macro.  
Sama zmiana architektury (TCN/Transformer/Mamba/MoE/multitask) nie dala trwalego przelamania.

Przelom pojawil sie po:
- naprawie metodologii cech (usuniecie leakage i cechy kauzalne),
- ostrzejszej selekcji modeli pod domain shift (FD002/FD004),
- przejsciu z "jednego modelu" na system: globalny model + specjalisci + blend per FD.

Efekt: zejscie do `17.xx` RMSE macro.

## What This Repository Demonstrates
- pelny pipeline prognostyki RUL dla `FD001..FD004`,
- walidacje i ranking ukierunkowane na stabilnosc, nie tylko pojedynczy lucky run,
- decyzje modelowe oparte o metryki operacyjne dla trudnych warunkow.

## Results (Hard Numbers)
| Milestone | Macro RMSE | Worst-FD RMSE | Artifact |
|---|---:|---:|---|
| Pre-fix baseline | 46.2155 | - | `outputs/tuning_hybrid_multifd_v3_selection/hybrid_multifd_tuning.csv` |
| After pipeline fixes | 21.0303 | - | `outputs/tuning_hybrid_multifd_v8_causal_features_baseline/hybrid_multifd_tuning.csv` |
| After encoder/stability tuning | 20.8845 | - | `outputs/tuning_hybrid_multifd_v11_encoder_mix/hybrid_multifd_tuning.csv` |
| Previous best ensemble | 17.1132 | 23.7392 | `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3fast_t2s42_w101/hierarchical_ensemble_metrics.csv` |
| **Current best (latest)** | **17.0443** | **23.4693** | `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101/hierarchical_ensemble_metrics.csv` |

Global reduction vs baseline:
- `46.2155 -> 17.0443` (ok. `63.1%` mniej bledu macro).

## Current Best System
- Global branch: 5-seed median ensemble (`trial_003`, seeds 42..46)
- FD13 specialist: `models/tuning_fd13_stage_allpoints_multiseed_loop1/trial_001/seed_42`
- FD24 specialist: `models/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun/trial_002/seed_43`
- Blend weights:
  - `FD001=0.00`
  - `FD002=0.10`
  - `FD003=0.00`
  - `FD004=0.43`

Per-FD RMSE (best run):
- `FD001=10.7157`
- `FD002=22.3907`
- `FD003=11.6014`
- `FD004=23.4693`

## Project Status
- Project is **not finished**.
- Interim target `<25` is achieved.
- Final target `<=12.5` is still open.
- Main bottleneck remains `FD004` / FD24 regime.

## Where To Start
- Technical runbook: `README_2.md`
- Iteration log: `docs/WORKLOG.md`
- Latest run comparison: `outputs/comparison_fd24v3full_cross.csv`
- Full experiment path (including dead branches): `README_2.md` + `docs/WORKLOG.md`
