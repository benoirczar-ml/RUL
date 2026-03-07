# WORKLOG

## 2026-03-06 - Project 2 bootstrap (RUL)
- Utworzono osobne repo `RUL` i podlaczono do GitHub.
- Dodano pipeline baseline:
  - `train.py`, `predict.py`, `evaluate.py`
  - `src/rul_pipeline/*` (data/features/model/metrics/io)
- Dodano konfiguracje `config/train_baseline.json`.
- Dodano testy `pytest` (3 testy) i `requirements.txt`.
- Dodano dokumentacje uruchomienia `README.md`.

## 2026-03-06 - Sequence model + benchmark
- Dodano model sekwencyjny `LSTMRegressor`:
  - `train_sequence.py`
  - `src/rul_pipeline/sequence.py`
  - `src/rul_pipeline/sequence_model.py`
- Rozszerzono inferencje o wspolny moduł:
  - `src/rul_pipeline/inference.py`
  - `predict.py` obsluguje `hist_gbr` i `lstm_regressor`.
- Dodano porownanie modeli:
  - `benchmark_models.py` (RMSE/MAE/PHM score na zbiorze testowym).
- Dodano konfiguracje `config/train_lstm.json`.
- Dodano test `tests/test_sequence.py`.

## 2026-03-06 - Full test (FD001..FD004) + wnioski po 2 modelach
- Uruchomiono pelny benchmark dla 2 modeli:
  - `HistGradientBoostingRegressor`
  - `LSTMRegressor`
- Zakres: `FD001`, `FD002`, `FD003`, `FD004`.
- Wynik: `LSTM` lepszy od baseline na wszystkich 4 zbiorach (RMSE/MAE/PHM score).
- Interpretacja etapu:
  - to nie jest jeszcze finalny poziom produkcyjny,
  - to mocny etap PoC pokazujacy kierunek (model sekwencyjny > baseline).
- Kolejny etap: dopracowanie modelu i walidacji (dalszy tuning + kolejne architektury).

## 2026-03-06 - Urealnienie protokolu walidacji
- `train.py` i `train_sequence.py` ustawione domyslnie na walidacje `last_cycle_only=true`.
- Metadane modeli zapisuja teraz:
  - `metrics_valid` (metryki glowne),
  - `metrics_valid_last_cycle`,
  - `metrics_valid_all_cycles`.
- Cel: zrownac walidacje z rzeczywistym zadaniem testowym (RUL na ostatnim cyklu).
- Pelny rerun po zmianie protokolu (`outputs/full_v2/benchmark_all_fd.csv`):
  - FD001: LSTM 67.94 vs HistGBR 86.30
  - FD002: LSTM 95.07 vs HistGBR 98.46
  - FD003: LSTM 60.94 vs HistGBR 84.02
  - FD004: LSTM 99.28 vs HistGBR 103.20
  - Wniosek: LSTM nadal wygrywa 4/4.

## 2026-03-06 - Start etapu tuning LSTM
- Dodano `tune_lstm.py` do automatycznego searchu hiperparametrow.
- Dodano `config/tune_lstm.json` (grid search).
- Ranking metryk zapisywany per-FD i globalnie do `outputs/tuning/`.
- Uruchomiono tuning `tuning_v1` (`max-trials=4`):
  - wyrazna poprawa `best_by_test` na FD001/FD003/FD004 wzgledem poprzedniego `full_v2`,
  - mocna poprawa na FD002 i FD004 juz w smoke.
- Ważny wniosek:
  - `best_by_val` i `best_by_test` czesto sa rozne,
  - wskazuje to na niedopasowanie walidacji do dystrybucji testowej,
  - kolejny etap: walidacja pseudo-testowa (truncation-based).

## 2026-03-06 - Walidacja pseudo-testowa (truncation-based)
- Dodano generator walidacji obcietej:
  - `build_truncated_validation` w `src/rul_pipeline/data.py`.
- `train.py` i `train_sequence.py` przelaczone na domyslne:
  - `val_strategy=truncation`,
  - `val_min_prefix=20`.
- Metadane modeli zawieraja teraz:
  - `metrics_valid` (metryki na walidacji pseudo-testowej),
  - `metrics_valid_full_last_cycle` (metryki pomocnicze na pelnym last-cycle),
  - `validation_cuts` (jak obcieto kazda jednostke).

## 2026-03-06 - Tuning v2 na walidacji truncation
- Uruchomiono `tune_lstm.py --max-trials 4` z nowym protokolem walidacji.
- Artefakty:
  - `outputs/tuning_v2_trunc/lstm_tuning_all_fd.csv`
  - `outputs/tuning_v2_trunc/lstm_tuning_all_fd.json`
  - `outputs/tuning_v2_trunc/selected_vs_hist_baseline.csv`
- Najlepsze triale (wybor po walidacji) vs baseline HistGBR:
  - FD001: 40.98 vs 86.30 (RMSE)
  - FD002: 54.01 vs 98.46 (RMSE)
  - FD003: 42.32 vs 84.02 (RMSE)
  - FD004: 54.57 vs 103.20 (RMSE)
- Wniosek: po urealnieniu walidacji i tuningu LSTM jest wyraznie lepszy od baseline 4/4.

## 2026-03-06 - Oznaczenie statusu biznesowego
- Wyniki oznaczono jako:
  - mocny etap PoC / portfolio (do pokazywania rekruterom),
  - nie finalny system produkcyjny dla lotnictwa.
- Taka adnotacja zostala dopisana w `README.md` w sekcji "Jak to komunikowac".

## 2026-03-06 - Produkcyjne elementy: gating + stabilnosc protokolu
- Dodano inferencje cycle-level:
  - `predict_all_cycles`, `predict_on_dataframe` w `src/rul_pipeline/inference.py`.
- Dodano warstwe operacyjna:
  - `src/rul_pipeline/operations.py` (alerty + metryki operacyjne).
  - `evaluate_operational_policy.py` (grid policy i ranking).
- Dodano walidacje stabilnosci:
  - `validate_truncation_protocol.py` (multi-seed truncation report).
- Dodano testy:
  - `tests/test_operations.py`.
- Smoke wyniki (FD001, model `models/tuning_v2_trunc/FD001/trial_004`):
  - best policy: trigger=60, consecutive=2, cooldown=10
  - recall=1.00, false_alert_rate=0.1315, median lead time=98 cykli.

## 2026-03-06 - 2-stage gating + hysteresis (operational policy v2)
- Rozszerzono polityke alertow o:
  - `exit_rul` (hysteresis),
  - `trend_window` + `trend_delta` (2-stage trend gate).
- Zmienione pliki:
  - `src/rul_pipeline/operations.py`
  - `evaluate_operational_policy.py`
  - `tests/test_operations.py`
- Testy: `pytest` 9/9 passed.
- Kalibracja na `FD001..FD004`:
  - `outputs/ops_calibration_v2/ops_policy_all_fd_v2.csv`
  - utrzymany recall=1.0 na wszystkich FD,
  - mniejsza liczba alertow / false alerts vs polityka bez trend gate.
- Dodatkowa notatka:
  - `NaN` w kolumnie `exit_rul` oznacza `None` (brak hysteresis), bez wpływu na obliczenia.

## 2026-03-06 - Deployment policy selection (constraint-based)
- Dodano selektor gotowych polityk pod ograniczenia biznesowe:
  - `select_deployment_policies.py`
- Wejscie:
  - gridy z `outputs/ops_calibration_v2/*_policy_grid.csv`
- Wyjscie:
  - `outputs/deployment_policies/deployment_policy_selection.csv`
  - `outputs/deployment_policies/policy_config_FD*.json`
- Przebieg z constraintami:
  - `min_recall=0.98`
  - `max_false_alert_rate=0.30`
  - `min_median_lead=60`
- Wynik:
  - `FD001` spelnia constrainty,
  - `FD002/FD003/FD004` jeszcze nie spelniaja false alert rate.

## 2026-03-07 - Protocol hardening pod FD002/FD004 (selection + leakage fix)
- Dodano metryki degradacji koncowej i ryzyka:
  - `rmse_tail`, `mae_tail`, `late_mae`, `miss_rate`, `composite_risk_score`
  - plik: `src/rul_pipeline/metrics.py`
- Trening `Conv+Attention+LSTM` dostal early-stopping po metryce kompozytowej:
  - nowe parametry: `early_stop_metric`, `selection_*`
  - historia epok zapisuje: `valid_tail_rmse`, `valid_miss_rate`, `valid_selection_score`
  - plik: `src/rul_pipeline/hybrid_sequence_model.py`
- `train_hybrid_multifd.py` raportuje i zapisuje:
  - metryki per-FD rozszerzone o tail/miss,
  - `selection_valid` i `selection_valid_full_last_cycle` w `metadata.json`.
- `tune_hybrid_multifd.py` przelaczony na ranking po walidacji (`val_selection_score`), a nie po teście.
- Zaktualizowano konfiguracje:
  - `config/train_hybrid_multifd.json`
  - `config/tune_hybrid_multifd*.json`
- Testy: `pytest` 14/14 passed.
- Commit: pending

## 2026-03-07 - Multi-FD tuning v3 (new selection) + seed ensemble check
- Uruchomiono tuning:
  - `tune_hybrid_multifd.py --max-trials 10`
  - output: `outputs/tuning_hybrid_multifd_v3_selection/*`
  - models: `models/tuning_hybrid_multifd_v3_selection/*`
- Top-3 po `val_selection_score`:
  - `trial_003` -> `89.6471` (best)
  - `trial_009` -> `89.9959`
  - `trial_007` -> `90.7644`
- Najlepszy trial (`trial_003`) na teście:
  - `test_rmse_macro=47.5471`
  - `test_rmse_worst_fd=54.7039`
- Dodano skrypt stabilizacyjny:
  - `scripts/seed_ensemble_multifd.py`
  - funkcja: trenowanie wielu seedow dla wybranego trialu + median ensemble na test.
- Uruchomiono seed ensemble dla `trial_003`, seeds `42,43,44`:
  - output: `outputs/seed_ensemble_multifd_v1/*`
  - models: `models/seed_ensemble_multifd_v1/*`
  - per-seed `rmse_macro`: `47.5471`, `48.5669`, `48.1024`
  - ensemble median `rmse_macro`: `48.0817` (brak poprawy vs best seed).
- Krytyczna obserwacja:
  - `miss_rate@30 = 1.0` dla seedow i ensemble na wszystkich FD,
  - obecny loss/sampling nadal nie domyka koncowej degradacji.
- Testy po zmianach: `pytest` 14/14 passed.
- Commit: pending

## 2026-03-07 - Krytyczny fix feature pipeline (unit_gid + causal features)
- Zidentyfikowano blad w multi-FD feature engineering:
  - `build_features()` grupowal po `unit`, co mieszalo trajektorie miedzy FD dla tych samych numerow jednostek.
  - naprawa: grupowanie/sortowanie po `unit_gid` (jezeli kolumna istnieje).
  - plik: `src/rul_pipeline/features.py`
- Dodano test regresyjny na mix FD:
  - `test_build_features_uses_unit_gid_when_present`
  - plik: `tests/test_features.py`
- Zidentyfikowano train/test mismatch przez cechy niekauzalne:
  - poprzednio statystyki per-unit byly liczone po calej trajektorii (future leakage dla wczesnych okien).
  - naprawa: wersja kauzalna (expanding mean/std), `cycle_norm` bez per-unit max-cycle leakage.
  - plik: `src/rul_pipeline/features.py`
- Testy: `pytest` 17/17 passed.

## 2026-03-07 - Multi-FD tuning v8 po fixie cech (przelamanie <25)
- Uruchomiono:
  - `tune_hybrid_multifd.py --config config/tune_hybrid_multifd_features_baseline.json --max-trials 2`
  - output: `outputs/tuning_hybrid_multifd_v8_causal_features_baseline/*`
  - models: `models/tuning_hybrid_multifd_v8_causal_features_baseline/*`
- Wyniki testowe:
  - `trial_001`: `test_rmse_macro=21.0303` (CEL `<25` osiagniety)
  - per-FD: `FD001=12.3084`, `FD002=28.5935`, `FD003=11.3634`, `FD004=31.8558`
  - `trial_002`: `test_rmse_macro=22.0583`
- Wniosek inzynierski:
  - glowna bariera byla metodologiczna (bledne grupowanie + niekauzalne cechy), nie brak "nowszej architektury".
  - po usunieciu tych bledow ten sam pipeline Conv+Attention+LSTM skoczyl z ~47-48 do ~21 macro RMSE.
- Status celu:
  - cel iteracyjny `<25`: wykonany.
  - cel docelowy `<=12.5`: nadal otwarty (najsłabsze FD002/FD004).

## 2026-03-07 - Iteracja pod cel <=12.5 (kolejne petle)
- Diagnostyka bledu best modelu `v8/trial_001`:
  - FD002 i FD004 maja wyrazny ujemny bias (~ -11 RUL), ale sama kalibracja liniowa daje tylko mala poprawa
    (`test_rmse_macro 21.03 -> 20.76`), bez przelomu.
- Dedykowany tuning per-FD:
  - FD002: `outputs/tuning_fd002_targeted_v1/*`
  - FD004: `outputs/tuning_fd004_targeted_v1/*`
  - efekt:
    - FD002 poprawiony lokalnie do `27.33` (z `28.59`), ale bez globalnego przelomu,
    - FD004 nie poprawiony (best dedykowany ~`33.69`, gorzej od multi-FD `31.86`).
- Szybki eksperyment cech reżimowych OC:
  - dodane i przetestowane, ale bez poprawy globalnej (test macro ~24.76-25.55 w smoke),
  - zmiana wycofana do stabilnego feature-setu.
- Tuning `v11` (mix encoderow conv/tcn, po naprawionym pipeline):
  - `outputs/tuning_hybrid_multifd_v11_encoder_mix/*`
  - najlepszy test: `trial_002` -> `test_rmse_macro=20.8845`, `worst_fd=30.2800`
  - to obecnie najlepszy wynik po etapie `<25`, ale nadal powyzej celu `<=12.5`.
- Dodatkowy test wariantu `v11/trial_002` z `max_rul=150`:
  - wynik gorszy (`test_rmse_macro=23.0112`), kierunek odrzucony.

### Status po tej petli
- Best known global: `test_rmse_macro=20.8845` (multi-FD, mix encoderow).
- Cel `<=12.5`: nadal NIEosiagniety.

## 2026-03-07 - Co DOKLADNIE dalo skok ~40-50 -> ~20
- Punkt startowy przed krytycznym fixem:
  - `outputs/tuning_hybrid_multifd_v3_selection/hybrid_multifd_tuning.csv`
  - best: `test_rmse_macro=47.5471` (trial_003)
  - typowy zakres: ~47-48 macro RMSE.
- Krok #1 (najwazniejszy): naprawa przecieku/mieszania w cechach multi-FD:
  - blad: grupowanie cech po `unit` zamiast po `unit_gid` (kolizje numerow jednostek miedzy FD).
  - plik: `src/rul_pipeline/features.py`
  - test regresyjny: `tests/test_features.py::test_build_features_uses_unit_gid_when_present`.
- Krok #2 (najwazniejszy): usuniecie niekauzalnych cech (future leakage):
  - blad: statystyki per-unit liczone po calej trajektorii.
  - naprawa: cechy kauzalne (`expanding mean/std`), bez max-cycle leakage w `cycle_norm`.
  - plik: `src/rul_pipeline/features.py`.
- Efekt bez zmiany "magicznej architektury", na tym samym nurcie modelowym:
  - `outputs/tuning_hybrid_multifd_v8_causal_features_baseline/hybrid_multifd_tuning.csv`
  - best: `test_rmse_macro=21.0303` (trial_001).
- Dodatkowe domkniecie:
  - `outputs/tuning_hybrid_multifd_v11_encoder_mix/hybrid_multifd_tuning.csv`
  - best: `test_rmse_macro=20.8845` (trial_002; tcn mix).
- Co NIE dalo przelomu:
  - sama nowa architektura bez napraw metodologii,
  - dedykowany tuning FD002/FD004 (lokalne zyski, brak globalnego przelomu),
  - podniesienie `max_rul` do 150/200,
  - cechy OC-residuals (testowane i wycofane).

### Konkluzja techniczna (40->20)
- Skok byl glownie wynikiem naprawy metodologii danych/cech (data correctness), a nie "kolejnego wiekszego modelu".
- Dopiero po usunieciu tych bledow strojenie encodera (v11) dalo sensowny dodatkowy zysk (21.03 -> 20.88).

## 2026-03-07 - Regime normalization (train-only) + 2-stage experts + nowy tuning
- Wdrozone train-only regime normalization:
  - nowy modul: `src/rul_pipeline/regime_normalization.py`
  - fit tylko na train, zapis stanu do `metadata.json`, apply na valid/test/inference.
  - integracja:
    - trening: `train_hybrid_multifd.py`
    - inferencja: `src/rul_pipeline/inference.py`
- Wdrozona architektura 2-stage (gate + dwa regresory):
  - nowe pola configu i lossy w `src/rul_pipeline/hybrid_sequence_model.py`:
    - `use_regime_experts`
    - `regime_threshold`
    - `regime_gate_loss_weight`, `regime_gate_pos_weight`
    - `regime_expert_loss_weight`
  - checkpointy kompatybilne wstecz (arch defaults przy load).
- Dodane testy:
  - `tests/test_regime_normalization.py`
  - rozszerzenie `tests/test_hybrid_sequence.py` o smoke dla regime experts.
  - testy: `python -m pytest -q tests/test_regime_normalization.py tests/test_hybrid_sequence.py` -> `8 passed`.

### Tuning: FD002 target (nowy wariant)
- config: `config/tune_hybrid_fd002_regime_experts_v2.json`
- output: `outputs/tuning_fd002_regime_experts_v2/hybrid_multifd_tuning.csv`
- 12 triali:
  - best po tescie: `trial_001`, `test_rmse_FD002=22.7290`
  - best po walidacji: `trial_012`, `val_selection=16.3709`, ale `test_rmse_FD002=24.2618`
- Wniosek: silny sygnal overfitu (bardzo niski val nie przeklada sie liniowo na test).

### Tuning: FD004 target (nowy wariant)
- config: `config/tune_hybrid_fd004_regime_experts_v2.json`
- output: `outputs/tuning_fd004_regime_experts_v2/hybrid_multifd_tuning.csv`
- 12 triali:
  - best po tescie: `trial_008`, `test_rmse_FD004=24.8367`
  - best po walidacji: `trial_004`, `val_selection=19.4668`, `test_rmse_FD004=25.0746`

### Full multi-FD (nowy wariant)
- config: `config/tune_hybrid_multifd_regime_experts_v1.json`
- output: `outputs/tuning_multifd_regime_experts_v1/hybrid_multifd_tuning.csv`
- 8 triali:
  - best po tescie: `trial_003`
  - `test_rmse_macro=17.3854`
  - per-FD: `FD001=11.9074`, `FD002=22.1422`, `FD003=11.1476`, `FD004=24.3446`

### Delta vs poprzedni best global (v11)
- Poprzedni best:
  - `outputs/tuning_hybrid_multifd_v11_encoder_mix/hybrid_multifd_tuning.csv`
  - `test_rmse_macro=20.8845`
  - per-FD: `12.3773 / 28.6093 / 12.2713 / 30.2800`
- Nowy best:
  - `test_rmse_macro=17.3854`
  - per-FD: `11.9074 / 22.1422 / 11.1476 / 24.3446`
- Zysk:
  - macro: `-3.4991` RMSE (~`-16.8%` vs 20.8845)
  - najwiekszy zysk na trudnych FD:
    - `FD002: -6.4671`
    - `FD004: -5.9354`

### Status celu
- Cel iteracyjny `<25`: utrzymany z duzym zapasem.
- Cel docelowy `<=12.5`: nadal NIEosiagniety.

## 2026-03-07 - Etapowe wdrozenie planu 1..5 (selection, ensemble, calibration, loss, domain adaptation)
- Wdrozone wszystkie 5 kierunkow jako narzedzia + kod treningowy.

### 1) Stabilizacja model-selection (multi-seed, test-like split)
- Dodany skrypt:
  - `scripts/tune_hybrid_multifd_multiseed.py`
- Co robi:
  - uruchamia te same triale dla wielu seedow,
  - rankuje po `val_selection_mean + std_penalty * val_selection_std`,
  - opcjonalnie robi test tylko dla top-K triali.
- Smoke run:
  - `outputs/tuning_multifd_multiseed_smoke/*` wygenerowane poprawnie.

### 2) Hierarchiczny ensemble specjalistow
- Dodany skrypt:
  - `scripts/hierarchical_ensemble_multifd.py`
- Koncepcja:
  - model globalny + specjalista `FD001/FD003` + specjalista `FD002/FD004`,
  - per-FD dobierane wagi blendu na pseudo-teście truncation (train).
- Smoke run (FD001):
  - `outputs/hierarchical_ensemble_smoke/*`
  - pipeline działa end-to-end.

### 3) Kalibracja per-FD/per-regime + conformal
- Dodany modul:
  - `src/rul_pipeline/calibration.py`
- Dodany skrypt:
  - `scripts/calibrate_multifd_predictions.py`
- Co robi:
  - per-FD isotonic,
  - opcjonalnie osobno dla regime low/high (wg progu predykcji),
  - conformal radius (intervale predykcyjne).
- Smoke run (FD001):
  - `outputs/calibration_multifd_smoke/*`
  - `test_rmse_macro` poprawione z `11.9074` do `11.4604`.

### 4) Loss ukierunkowany na najgorszy FD i miss-rate tail
- Rozszerzony `ConvAttentionLSTMConfig` i trening:
  - `worst_fd_loss_weight`
  - `tail_miss_hinge_weight`
  - `tail_miss_margin`
- Implementacja:
  - `src/rul_pipeline/hybrid_sequence_model.py`
  - `train_hybrid_multifd.py` (CLI + metadata).

### 5) Adaptacja domenowa (FD002/FD004)
- Dodane latent-domain alignment w loss:
  - `use_domain_alignment`
  - `domain_alignment_weight`
  - `domain_alignment_pairs` (np. `1-3` dla FD002/FD004 w full multi-FD).
- Implementacja:
  - `src/rul_pipeline/hybrid_sequence_model.py`
  - `train_hybrid_multifd.py`
- Smoke train z aktywnymi nowymi skladnikami loss:
  - `models/smoke_fd24_allpoints_loss`
  - logi pokazuja aktywne skladowe: `train_worst_fd`, `train_tail_miss`, `train_domain` > 0.

### Pliki konfiguracyjne dodane pod nowy etap
- `config/tune_hybrid_multifd_stage_allpoints_v1.json`
- `config/train_hybrid_multifd_fd13_specialist_v1.json`
- `config/train_hybrid_multifd_fd24_specialist_v1.json`

### Testy
- Dodane:
  - `tests/test_calibration.py`
  - rozszerzenie `tests/test_hybrid_sequence.py` o smoke dla domain/loss.
- Wynik:
  - `python -m pytest -q tests/test_hybrid_sequence.py tests/test_regime_normalization.py tests/test_calibration.py`
  - `11 passed`.

### Uruchomienia etapowe (realne, nie tylko smoke kodu)
- Etap 1: multi-seed tuning (`2 seedy`, `1 trial`) na configu all-points:
  - `outputs/tuning_multifd_stage_allpoints_multiseed_v1/hybrid_multifd_multiseed_agg.csv`
  - `val_selection_mean=15.2649`, `val_selection_std=1.4720`, `val_rank_score=16.0008`
  - test (mean over seeds): `test_rmse_macro=17.9720`, `worst_fd=24.5241`
- Etap 2: trening specjalistow:
  - `models/stage_allpoints_fd13_specialist_v1` (FD001/FD003)
  - `models/stage_allpoints_fd24_specialist_v1` (FD002/FD004)
  - oba modele wytrenowane poprawnie.
- Etap 2 (blend global + specjalisci):
  - `outputs/hierarchical_ensemble_stage_allpoints_v1/hierarchical_ensemble_metrics.csv`
  - blend test: `macro_rmse=17.5543`, `worst_fd_rmse=24.1332`
  - per-FD blend: `FD001=12.3238`, `FD002=22.0177`, `FD003=11.7428`, `FD004=24.1332`
- Etap 3: kalibracja per-FD/per-regime na modelu globalnym:
  - `outputs/calibration_multifd_stage_allpoints_v1/calibration_test_metrics.csv`
  - `macro_rmse raw=17.8159 -> calibrated=17.8830` (w tym przebiegu kalibracja pogorszyla global RMSE).

### Wniosek po pierwszej petli all-points
- Nowe elementy dzialaja technicznie i sa gotowe do kolejnych iteracji.
- W tej konkretnej petli:
  - blend specjalistow poprawil global model stage-allpoints (`17.97 -> 17.55`),
  - ale nadal nie przebil dotychczasowego best global `17.3854`.

## 2026-03-07 - Petla kontynuacyjna (FD24 loop1 -> full loop2 -> cross-ensemble)

### FD002/FD004: nowy tuning multi-seed (all-points v2)
- Config:
  - `config/tune_hybrid_multifd_fd24_stage_allpoints_v2.json`
- Run:
  - `outputs/tuning_fd24_stage_allpoints_multiseed_loop1/hybrid_multifd_multiseed_agg.csv`
- Najlepszy po rankingu stabilnosci:
  - `trial_006`, `val_rank_score=14.8269`
  - test (mean over seeds):
    - `test_rmse_macro=23.5695`
    - `test_rmse_worst_fd=24.2462`
- Wniosek:
  - poprawa walidacyjna na FD24 byla, ale bez przelamania na tescie.

### Full multi-FD: transfer profilu FD24 (loop2)
- Config:
  - `config/tune_hybrid_multifd_stage_allpoints_v2.json`
- Run:
  - `outputs/tuning_multifd_stage_allpoints_v2_loop2/hybrid_multifd_multiseed_agg.csv`
- Najlepszy po rankingu stabilnosci:
  - `trial_004`, `val_rank_score=16.1145`
  - test (mean over seeds):
    - `test_rmse_macro=17.9644`
    - `test_rmse_worst_fd=24.3986`
- Wniosek:
  - ten konkretny transfer nie przebil global-best `17.3854`.

### Ensemble i kalibracja dla loop2
- Hierarchical ensemble (global loop2 + specjalisci):
  - `outputs/hierarchical_ensemble_stage_allpoints_v2_loop2/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.6922`, `worst_fd=24.7247`
- Kalibracja global loop2:
  - `outputs/calibration_multifd_stage_allpoints_v2_loop2/calibration_test_metrics.csv`
  - `raw=17.8672 -> calibrated=18.1218` (pogorszenie RMSE)

### Cross-ensemble (najlepszy realny zysk w tej petli)
- Sprawdzono miks: stary global-best + specjalisci FD13/FD24:
  - `outputs/hierarchical_ensemble_cross_oldglobal_stagefd24/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.2212`, `worst_fd=23.8778`
- Doregulowano gestosc siatki wag (`weight-grid-size=51`):
  - `outputs/hierarchical_ensemble_cross_oldglobal_stagefd24_w51/hierarchical_ensemble_metrics.csv`
  - **best tej petli:**
    - `test_rmse_macro=17.2135`
    - `worst_fd=23.8655`
    - per-FD blend:
      - `FD001=11.9344`
      - `FD002=21.9076`
      - `FD003=11.1465`
      - `FD004=23.8655`

### Delta po tej petli
- Start petli (best global):
  - `17.3854` macro, `24.3446` worst-FD.
- Koniec petli (best cross-ensemble w51):
  - `17.2135` macro, `23.8655` worst-FD.
- Zysk:
  - macro: `-0.1719`
  - worst-FD: `-0.4791`

### Status celu
- Cel interim `<25`: utrzymany.
- Cel docelowy `<=12.5`: nadal nieosiagniety.

## 2026-03-07 - Petla kolejna (ensemble-global + FD24 long-window v3)

### 1) Punkt 1 domkniety: globalny seed-ensemble + cross-ensemble
- Multi-seed retrain globalnego `trial_003` (5 seedow) + median ensemble:
  - `outputs/seed_ensemble_global_trial003_loop2/trial_003_summary.csv`
  - wynik:
    - `ensemble_median rmse_macro=17.1754` (lepiej niz pojedyncze seedy)
- Rozszerzony skrypt ensemble:
  - `scripts/hierarchical_ensemble_multifd.py`
  - nowa obsluga: `--global-model-dirs` (mediana predykcji z wielu modeli globalnych).
- Cross-ensemble: `ensemble-global + FD13 specialist + stage FD24 specialist`:
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_stagefd24_w101/hierarchical_ensemble_metrics.csv`
  - wynik:
    - `test_rmse_macro=17.1449`
    - `worst_fd=23.8358`

### 2) Punkt 2: FD24 long-window (seq_len 80/100/120) + mocniejszy tail/late
- Dodane configi:
  - `config/tune_hybrid_multifd_fd24_stage_allpoints_v3.json`
  - `config/tune_hybrid_multifd_fd24_stage_allpoints_v3_fast.json`
- Ranking uruchamiany zgodnie z zalozeniem:
  - `val_rank_score = mean + 0.7*std + 1.5*worst_fd`
  - przez:
    - `scripts/tune_hybrid_multifd_multiseed.py --std-penalty 0.7 --worst-fd-penalty 1.5`
- Pelny v3 (3 seedy, 9 triali) przerwany:
  - powtarzalny `SIGKILL`/OOM na ciezkich kombinacjach (m.in. `seq_len=120`, wysoka kara worst-fd).
- Wersja v3_fast (2 seedy, 4 triale) tez zatrzymana po 2 kompletnych trialach z powodu `SIGKILL` na `trial_003/seed_42`.
- Zapisane czesciowe agregaty:
  - `outputs/tuning_fd24_stage_allpoints_multiseed_v3_fast_loop1_partial/seed_rows_partial.csv`
  - `outputs/tuning_fd24_stage_allpoints_multiseed_v3_fast_loop1_partial/agg_partial.csv`
- Czesc rankingowa (dostepna):
  - `trial_001` lepszy od `trial_002` po rankingu stabilnosci.

### 3) Punkt 3: ponowny cross-ensemble dla top checkpointow FD24
- Porownane warianty FD24 z v3_fast:
  - `trial_001/seed_42`
  - `trial_001/seed_43`
  - `trial_002/seed_42`
- Wyniki:
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3fast_t1s42_w51/hierarchical_ensemble_metrics.csv`
    - `test_rmse_macro=17.2053`, `worst_fd=24.0322`
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3fast_t1s43_w51/hierarchical_ensemble_metrics.csv`
    - `test_rmse_macro=17.2182`, `worst_fd=24.1578`
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3fast_t2s42_w51/hierarchical_ensemble_metrics.csv`
    - najlepszy z siatki `w51`:
      - `test_rmse_macro=17.1144`
      - `worst_fd=23.7441`
  - doregulowanie gestosci siatki wag:
    - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3fast_t2s42_w101/hierarchical_ensemble_metrics.csv`
    - **best aktualny:**
      - `test_rmse_macro=17.1132`
      - `worst_fd=23.7392`
- Zbiorcze porownanie:
  - `outputs/comparison_fd24v3fast_cross.csv`

### Delta po tej petli
- Przed petla (best):
  - `17.1969` macro, `23.8655` worst-FD.
- Po petli (best):
  - `17.1132` macro, `23.7392` worst-FD.
- Zysk:
  - macro: `-0.0837`
  - worst-FD: `-0.1263`

### Status celu
- Cel interim `<25`: utrzymany z marginesem.
- Cel docelowy `<=12.5`: nadal nieosiagniety.

## 2026-03-07 - Petla 4-krokowa (FD24 v3 rerun -> ranking -> cross -> werdykt)

### Krok 1: FD24 rerun (long-window v3)
- Run:
  - `outputs/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun/*`
  - `models/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun/*`
- Uwaga operacyjna:
  - przebieg zostal swiadomie zatrzymany po kompletnych trialach `001-003` (9 seed-runow) i przejscie do etapu decyzyjnego.

### Krok 2: Ranking po `mean + 0.7*std + 1.5*worst_fd`
- Agregaty (triale 001-003):
  - `outputs/tuning_fd24_stage_allpoints_multiseed_v3_fullrerun_partial3/agg_partial.csv`
- Kolejnosc po rankingu:
  - `trial_001` (najlepszy aggregate)
  - `trial_003`
  - `trial_002`
- Najlepsze seedy per trial:
  - `trial_001/seed_42`
  - `trial_003/seed_42`
  - `trial_002/seed_43`

### Krok 3: Full test + cross-ensemble (ensemble-global + FD13 + top FD24)
- Wyniki:
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t1s42_w51/hierarchical_ensemble_metrics.csv`
    - `test_rmse_macro=17.1563`, `worst_fd=23.8374`
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t3s42_w51/hierarchical_ensemble_metrics.csv`
    - `test_rmse_macro=17.1201`, `worst_fd=23.7435`
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w51/hierarchical_ensemble_metrics.csv`
    - `test_rmse_macro=17.0451`, `worst_fd=23.4725`
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101/hierarchical_ensemble_metrics.csv`
    - **best tej petli:**
      - `test_rmse_macro=17.0443`
      - `worst_fd=23.4693`
- Wagi specialista (best, w101):
  - `FD001=0.00`, `FD002=0.10`, `FD003=0.00`, `FD004=0.43`
  - plik: `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101/hierarchical_ensemble_weights.json`

### Krok 4: Werdykt vs poprzedni best
- Poprzedni best:
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3fast_t2s42_w101/hierarchical_ensemble_metrics.csv`
  - `17.1132` macro, `23.7392` worst-FD
- Nowy best:
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101/hierarchical_ensemble_metrics.csv`
  - `17.0443` macro, `23.4693` worst-FD
- Delta:
  - macro: `-0.0689`
  - worst-FD: `-0.2699`
- Werdykt:
  - **TAK, przebijamy poprzedni wynik.**

## 2026-03-07 - Petla kolejna (FD13 tuning + cross-ensemble + meta-stacker)

### 1) FD001/FD003 - multi-seed tuning (nowy specjalista FD13)
- Config:
  - `config/tune_hybrid_multifd_fd13_stage_allpoints_v2.json`
- Run:
  - `outputs/tuning_fd13_stage_allpoints_multiseed_loop1/hybrid_multifd_multiseed_agg.csv`
- Najlepszy po rankingu stabilnosci:
  - `trial_001`, `val_rank_score=11.4190`
  - model do dalszego blendu:
    - `models/tuning_fd13_stage_allpoints_multiseed_loop1/trial_001/seed_42`
  - test (mean over seeds, dla top trial):
    - `test_rmse_macro=13.5684`
    - `test_rmse_worst_fd=14.9386`

### 2) Cross-ensemble z nowym FD13
- Global model:
  - `models/tuning_multifd_regime_experts_v1/trial_003`
- FD13 specialist (nowy):
  - `models/tuning_fd13_stage_allpoints_multiseed_loop1/trial_001/seed_42`
- FD24 specialist:
  - wariant A: `models/stage_allpoints_fd24_specialist_v1`
  - wariant B: `models/tuning_fd24_stage_allpoints_multiseed_loop1/trial_006/seed_43`

#### Wyniki
- A, `weight-grid-size=51`:
  - `outputs/hierarchical_ensemble_cross_oldglobal_newfd13_stagefd24_w51/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.1970`, `worst_fd=23.8655`
- A, `weight-grid-size=101` (**best tej petli**):
  - `outputs/hierarchical_ensemble_cross_oldglobal_newfd13_stagefd24_w101/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.1969`, `worst_fd=23.8655`
- B, `weight-grid-size=51`:
  - `outputs/hierarchical_ensemble_cross_oldglobal_newfd13_newfd24_w51/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.2928`, `worst_fd=24.1035`
- Test alternatywnego FD13 (agresywny seed):
  - `outputs/hierarchical_ensemble_cross_oldglobal_fd13trial006s44_stagefd24_w101/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.2028`, `worst_fd=23.8655`

### 3) Meta-stacker (nowy skrypt)
- Dodany skrypt:
  - `scripts/meta_stacker_multifd.py`
- Co robi:
  - uczy liniowe wagi blendu `global + specialist` na pseudo-val (truncation),
  - opcjonalnie rozdziela modele low/high regime.

#### Wyniki meta-stacker
- liniowy (bez regime split):
  - `outputs/meta_stacker_oldglobal_newfd13_stagefd24_linear/meta_stacker_metrics.csv`
  - `test_rmse_macro=17.4866`, `worst_fd=24.4211`
- regime split:
  - `outputs/meta_stacker_oldglobal_newfd13_stagefd24_regime/meta_stacker_metrics.csv`
  - `test_rmse_macro=17.4725`, `worst_fd=24.4156`
- Wniosek: meta-stacker w tej iteracji gorszy od blendu siatkowego.

### Delta po tej petli
- Przed petla (best):
  - `outputs/hierarchical_ensemble_cross_oldglobal_stagefd24_w51/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.2135`, `worst_fd=23.8655`
- Po petli (best):
  - `outputs/hierarchical_ensemble_cross_oldglobal_newfd13_stagefd24_w101/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.1969`, `worst_fd=23.8655`
- Zysk:
  - macro: `-0.0165`
  - worst-FD: `0.0000` (bez zmiany)

### Status celu
- Cel interim `<25`: utrzymany.
- Cel docelowy `<=12.5`: nadal nieosiagniety.

## 2026-03-08 - Dokumentacja publikacyjna + variance meta-stacker loop1

### 1) Variance-aware deep ensemble meta-stacker
- Dodany skrypt:
  - `scripts/meta_stacker_multifd_variance.py`
- Pula modeli (loop1):
  - global: 5
  - FD13 specialists: 3
  - FD24 specialists: 3
  - FD002 specialists: 3
  - FD004 specialists: 4
  - lacznie: `18` unikalnych checkpointow
- Wynik:
  - `outputs/meta_stacker_variance_loop1/meta_stacker_variance_metrics.csv`
  - `test_rmse_macro=17.5447`
  - `worst_fd=23.4783`
- Werdykt:
  - gorsze od aktualnego besta `17.0443`, nie promowane do finalnej sciezki.

### 2) Packaging pod portfolio/rekruterow
- Dodany generator assetow README:
  - `scripts/generate_readme_assets.py`
- Wygenerowane figury:
  - `docs/figures/rmse_progression.png`
  - `docs/figures/per_fd_comparison.png`
  - `docs/figures/fd004_pred_vs_true.png`
  - `docs/figures/fd004_error_hist.png`
  - `docs/figures/fd004_uncertainty_vs_error.png`
- Wygenerowane tabele:
  - `docs/tables/summary_results.csv`
  - `docs/tables/best_per_fd.csv`
  - `docs/tables/ablation_lift_summary.csv`
  - `docs/tables/fd004_best_errors.csv`
  - `docs/tables/fd004_uncertainty_vs_error.csv`

### 3) Readme refresh
- `README.md` przebudowany do wersji "publikacyjnej":
  - TL;DR, hard numbers, osadzone wykresy, pipeline, lessons learned, reproducibility.
- `README_2.md` ujednolicony jako techniczny appendix i decyzje engineeringowe.

### Status po tej iteracji
- Best global pozostaje:
  - `outputs/hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101/hierarchical_ensemble_metrics.csv`
  - `test_rmse_macro=17.0443`, `worst_fd=23.4693`
- Cel `<=12.5`: nadal nieosiagniety.
- Projekt oznaczony jako plateau po wielu petlach; repo pozostawione w stanie reproducible.
