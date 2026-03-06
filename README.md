# Project 2: Remaining Useful Life (C-MAPSS)

Projekt portfolio pod rekrutacje ML/AI: prognozowanie RUL silnikow turbowentylatorowych (NASA C-MAPSS).

## Zakres
- Dane: `RULdata/CMAPSSData/train_FDxxx.txt`, `test_FDxxx.txt`, `RUL_FDxxx.txt`
- Baseline: `HistGradientBoostingRegressor` (tabular, cycle-level)
- Model sekwencyjny: `LSTMRegressor` (okna czasowe)
- Feature engineering:
  - ustawienia operacyjne + sensory
  - `cycle_norm`
  - delty 1-krokowe dla wszystkich sensorow
- CLI:
  - `train.py`
  - `predict.py`
  - `evaluate.py`
- Metryki:
  - RMSE
  - MAE
  - PHM score (asymetryczna kara za opoznione przewidywania)

## Struktura
- `src/rul_pipeline/data.py` - wczytywanie C-MAPSS i target RUL
- `src/rul_pipeline/features.py` - cechy tabular i delty
- `src/rul_pipeline/modeling.py` - trening/inferencja modelu
- `src/rul_pipeline/sequence.py` - budowa okien sekwencyjnych + standaryzacja
- `src/rul_pipeline/sequence_model.py` - model LSTM i petla treningowa
- `src/rul_pipeline/inference.py` - wspolna inferencja dla wszystkich modeli
- `src/rul_pipeline/metrics.py` - metryki RUL
- `config/train_baseline.json` - domyslna konfiguracja treningu
- `tests/` - testy jednostkowe

## Instalacja
```bash
python -m pip install -r requirements.txt
```

## Trening (FD001)
```bash
python train.py
```

Przyklad nadpisania parametrow:
```bash
python train.py --fd FD002 --max-rul 130 --max-iter 500
```

## Trening sekwencyjny (LSTM)
```bash
python train_sequence.py
```

Szybki smoke:
```bash
python train_sequence.py --fd FD001 --epochs 3 --sample-step 2 --model-dir models/lstm_smoke_fd001
```

## Predykcja testu
1. Wybierz katalog modelu, np. `models/hist_gbr_FD001_YYYYMMDD_HHMMSS`
```bash
python predict.py \
  --model-dir models/hist_gbr_FD001_YYYYMMDD_HHMMSS \
  --fd FD001 \
  --output-csv outputs/predictions_fd001.csv
```
`predict.py` dziala zarowno dla `hist_gbr`, jak i `lstm_regressor`.

## Ewaluacja na `RUL_FDxxx.txt`
```bash
python evaluate.py \
  --predictions-csv outputs/predictions_fd001.csv \
  --fd FD001 \
  --output-json outputs/metrics_fd001.json
```

## Benchmark: baseline vs LSTM
```bash
python benchmark_models.py \
  --fd FD001 \
  --model-dirs models/hist_gbr_FD001_YYYYMMDD_HHMMSS models/lstm_FD001_YYYYMMDD_HHMMSS \
  --output-csv outputs/benchmark_fd001.csv \
  --output-json outputs/benchmark_fd001.json
```

## Status po 2 modelach (aktualny etap)
- Sprawdzone modele:
  - `HistGradientBoostingRegressor` (baseline)
  - `LSTMRegressor` (sekwencyjny)
- Pelny test wykonany na `FD001..FD004`.
- W tym etapie `LSTM` wygrywa z baseline na wszystkich 4 zbiorach.
- To jest etap **PoC/portfolio**, nie finalny system produkcyjny lotniczy.
- Model bedzie dalej dopracowywany (kolejne architektury, tuning, ostrzejsza walidacja).

## Testy
```bash
python -m pytest -q
```

## Uwagi o danych
- Pipeline zaklada standard C-MAPSS: 26 kolumn (`unit`, `cycle`, 3 operacyjne, 21 sensorow).
- W lokalnym mirrorze liczba jednostek dla FD004 moze sie roznic od starego opisu tekstowego; kod opiera sie na faktycznych plikach.
