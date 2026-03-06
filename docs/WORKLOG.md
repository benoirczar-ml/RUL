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
