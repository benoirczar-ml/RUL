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
Domyslnie walidacja jest pseudo-testowa (`val_strategy=truncation`): kazda jednostka walidacyjna jest obcinana przed awaria, a metryki liczone sa na punkcie odciecia.

Przyklad nadpisania parametrow:
```bash
python train.py --fd FD002 --max-rul 130 --max-iter 500
```

## Trening sekwencyjny (LSTM)
```bash
python train_sequence.py
```
Domyslnie early stopping i metryki sa liczone na walidacji pseudo-testowej (`val_strategy=truncation`).

Szybki smoke:
```bash
python train_sequence.py --fd FD001 --epochs 3 --sample-step 2 --model-dir models/lstm_smoke_fd001
```

GPU runtime toggles (dla CUDA):
```bash
python train_sequence.py \
  --fd FD001 \
  --device cuda \
  --pin-memory \
  --non-blocking \
  --use-amp \
  --enable-tf32 \
  --cudnn-benchmark
```

Szybki smoke/benchmark sciezki GPU transfer + AMP/TF32/cuDNN:
```bash
python scripts/gpu_runtime_smoke.py --device auto --output-json outputs/gpu_runtime_smoke.json
```

## Trening hybrydowy (Conv + Attention + LSTM)
```bash
python train_hybrid_sequence.py
```
Ten model laczy:
- `Conv1D`: lokalne wzorce czasowe,
- `MultiHeadAttention`: zaleznosci miedzy krokami sekwencji,
- `LSTM`: trend degradacji i pamiec czasowa.

Szybki smoke:
```bash
python train_hybrid_sequence.py \
  --fd FD001 \
  --epochs 2 \
  --sample-step 3 \
  --val-fraction 0.1 \
  --device cuda \
  --model-dir models/caelstm_smoke_fd001
```

## Predykcja testu
1. Wybierz katalog modelu, np. `models/hist_gbr_FD001_YYYYMMDD_HHMMSS`
```bash
python predict.py \
  --model-dir models/hist_gbr_FD001_YYYYMMDD_HHMMSS \
  --fd FD001 \
  --output-csv outputs/predictions_fd001.csv
```
`predict.py` dziala dla `hist_gbr`, `lstm_regressor` i `conv_attn_lstm_regressor`.

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

## Tuning LSTM
```bash
python tune_lstm.py --config config/tune_lstm.json --max-trials 6
```
Wyniki zapisywane sa do `outputs/tuning/` (ranking per FD + plik zbiorczy).
Aktualny przebieg: `outputs/tuning_v1/`.
Po wdrozeniu walidacji `truncation`:
- przebieg: `outputs/tuning_v2_trunc/`
- porownanie najlepszych triali (wybor po walidacji) vs baseline:
  - FD001: 40.98 vs 86.30 (RMSE)
  - FD002: 54.01 vs 98.46 (RMSE)
  - FD003: 42.32 vs 84.02 (RMSE)
  - FD004: 54.57 vs 103.20 (RMSE)
  - tabela: `outputs/tuning_v2_trunc/selected_vs_hist_baseline.csv`

## Operacyjna polityka alarmow (gating)
Symulacja alertowania na predykcjach cycle-level:
```bash
python evaluate_operational_policy.py \
  --model-dir models/tuning_v2_trunc/FD001/trial_004 \
  --fd FD001 \
  --split train \
  --trigger-ruls 60,80,100,120,140 \
  --consecutives 1,2 \
  --cooldowns 0,5,10 \
  --min-lead 5 \
  --max-lead 120 \
  --output-csv outputs/ops_policy_fd001_v2.csv \
  --output-json outputs/ops_policy_fd001_v2.json
```

Przyklad najlepszego wariantu (FD001 train, model `trial_004`):
- policy: `trigger_rul=60`, `consecutive=2`, `cooldown=10`
- recall: `1.00`
- false alerts: `129`
- false alert rate: `0.1315`
- median lead time: `98` cykli

### 2-stage gate + hysteresis
Mozna wlaczyc dodatkowe bramki:
- `exit_rul` (hysteresis: wyjscie ze stanu alarmu),
- `trend_window` + `trend_delta` (wymagany spadek predykcji RUL).

Przyklad:
```bash
python evaluate_operational_policy.py \
  --model-dir models/tuning_v2_trunc/FD001/trial_004 \
  --fd FD001 \
  --split train \
  --trigger-ruls 80,100,120,140 \
  --exit-ruls none,140,160 \
  --consecutives 2,3 \
  --cooldowns 10,20 \
  --trend-windows 0,5 \
  --trend-deltas 0,5 \
  --min-lead 5 \
  --max-lead 120 \
  --output-csv outputs/ops_policy_fd001_hybrid.csv \
  --output-json outputs/ops_policy_fd001_hybrid.json
```

Uwaga:
- w CSV wartosc `NaN` w kolumnie `exit_rul` oznacza `None` (czyli brak hysteresis), to nie jest blad.

Kalibracja V2 (4xFD, best LSTM per FD):
- plik zbiorczy: `outputs/ops_calibration_v2/ops_policy_all_fd_v2.csv`
- dla wszystkich FD utrzymany `recall=1.0`,
- trend gate (`trend_window=5`) obniza laczna liczbe alertow i false alerts vs wersja bez trendu.

## Selection pod constrainty biznesowe
Automatyczny wybór polityki pod targety (np. `recall>=0.98`, `false_alert_rate<=0.30`, `median_lead>=60`):
```bash
python select_deployment_policies.py \
  --policy-dir outputs/ops_calibration_v2 \
  --min-recall 0.98 \
  --max-false-alert-rate 0.30 \
  --min-median-lead 60 \
  --output-dir outputs/deployment_policies
```

Artefakty:
- `outputs/deployment_policies/deployment_policy_selection.csv`
- `outputs/deployment_policies/policy_config_FD001.json` ... `policy_config_FD004.json`

Wynik dla obecnych gridow:
- `FD001`: spelnia constrainty,
- `FD002`, `FD003`, `FD004`: jeszcze nie spelniaja progu false alert rate.

## Stabilnosc protokolu walidacji truncation
```bash
python validate_truncation_protocol.py \
  --model-dir models/tuning_v2_trunc/FD001/trial_004 \
  --fd FD001 \
  --cut-seeds 11,22,33,44,55 \
  --output-csv outputs/trunc_stability_fd001.csv \
  --output-json outputs/trunc_stability_fd001.json
```
To raportuje rozrzut metryk po wielu seedach obciecia trajektorii.

## Status po 2 modelach (aktualny etap)
- Sprawdzone modele:
  - `HistGradientBoostingRegressor` (baseline)
  - `LSTMRegressor` (sekwencyjny)
- Pelny test wykonany na `FD001..FD004`.
- W tym etapie `LSTM` wygrywa z baseline na wszystkich 4 zbiorach.
- To jest etap **PoC/portfolio**, nie finalny system produkcyjny lotniczy.
- Model bedzie dalej dopracowywany (kolejne architektury, tuning, ostrzejsza walidacja).
- Aktualny punkt odniesienia (full_v2, walidacja `last_cycle_only=true`):
  - FD001: LSTM RMSE `67.94` vs HistGBR `86.30`
  - FD002: LSTM RMSE `95.07` vs HistGBR `98.46`
  - FD003: LSTM RMSE `60.94` vs HistGBR `84.02`
  - FD004: LSTM RMSE `99.28` vs HistGBR `103.20`
  - Wynik: LSTM lepszy na wszystkich 4 zbiorach.
- Uwaga metodologiczna: tuning pokazal, ze `best_by_val` i `best_by_test` czesto sie rozjezdzaja, wiec kolejny krok to walidacja jeszcze bardziej zblizona do testu (symulacja obcietych trajektorii).
  - (wdrozone) walidacja pseudo-testowa oparta o obciete trajektorie (`truncation`).

## Jak to komunikowac
- Projekt mozna bezpiecznie pokazywac rekruterom i zespolom ML jako mocny PoC end-to-end.
- Nie nalezy opisywac tego etapu jako gotowy system produkcyjny dla lotnictwa.
- Wlasciwa narracja: wysoka poprawa modelowa + poprawna metodologia + kolejne kroki do hardeningu.

## Testy
```bash
python -m pytest -q
```

## Uwagi o danych
- Pipeline zaklada standard C-MAPSS: 26 kolumn (`unit`, `cycle`, 3 operacyjne, 21 sensorow).
- W lokalnym mirrorze liczba jednostek dla FD004 moze sie roznic od starego opisu tekstowego; kod opiera sie na faktycznych plikach.
