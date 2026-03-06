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
