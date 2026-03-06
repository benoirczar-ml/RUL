# Project 2: Remaining Useful Life (C-MAPSS)

Projekt portfolio pod rekrutacje ML/AI: prognozowanie RUL silnikow turbowentylatorowych (NASA C-MAPSS).

## Zakres
- Dane: `RULdata/CMAPSSData/train_FDxxx.txt`, `test_FDxxx.txt`, `RUL_FDxxx.txt`
- Baseline: `HistGradientBoostingRegressor` (tabular, cycle-level)
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

## Predykcja testu
1. Wybierz katalog modelu, np. `models/hist_gbr_FD001_YYYYMMDD_HHMMSS`
```bash
python predict.py \
  --model-dir models/hist_gbr_FD001_YYYYMMDD_HHMMSS \
  --fd FD001 \
  --output-csv outputs/predictions_fd001.csv
```

## Ewaluacja na `RUL_FDxxx.txt`
```bash
python evaluate.py \
  --predictions-csv outputs/predictions_fd001.csv \
  --fd FD001 \
  --output-json outputs/metrics_fd001.json
```

## Testy
```bash
python -m pytest -q
```

## Uwagi o danych
- Pipeline zaklada standard C-MAPSS: 26 kolumn (`unit`, `cycle`, 3 operacyjne, 21 sensorow).
- W lokalnym mirrorze liczba jednostek dla FD004 moze sie roznic od starego opisu tekstowego; kod opiera sie na faktycznych plikach.
