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
