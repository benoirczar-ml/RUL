from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]

import sys

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.data import add_train_rul, build_truncated_validation, load_rul_targets, load_split, select_last_cycle_rows
from rul_pipeline.inference import _predict_on_dataframe, predict_last_cycle
from rul_pipeline.io_utils import read_json, write_json
from rul_pipeline.metrics import mae, miss_rate, phm_score, rmse, rmse_tail


def _resolve(path_arg: str) -> Path:
    p = Path(path_arg)
    return (ROOT / p).resolve() if not p.is_absolute() else p


def _parse_csv_paths(raw: str | None) -> list[Path]:
    if raw is None:
        return []
    out = []
    for token in raw.split(","):
        tok = token.strip()
        if tok:
            out.append(_resolve(tok))
    return out


def _parse_fds(raw: str) -> list[str]:
    fds = [x.strip().upper() for x in raw.split(",") if x.strip()]
    valid = {"FD001", "FD002", "FD003", "FD004"}
    if not fds:
        raise ValueError("fds cannot be empty.")
    for fd in fds:
        if fd not in valid:
            raise ValueError(f"Unsupported FD={fd}. Allowed: {sorted(valid)}")
    return list(dict.fromkeys(fds))


def _predict_last_on_df(model_dir: Path, fd: str, df: pd.DataFrame, device: str) -> pd.DataFrame:
    metadata = read_json(model_dir / "metadata.json")
    pred_all = _predict_on_dataframe(
        model_dir=model_dir,
        metadata=metadata,
        data_df=df,
        device=device,
        dataset_fd=fd,
    )
    idx = pred_all.groupby("unit")["cycle"].idxmax().sort_values()
    return pred_all.loc[idx, ["unit", "pred_rul"]].sort_values("unit").reset_index(drop=True)


def _safe_name(model_dir: Path) -> str:
    tail = str(model_dir).replace(str(ROOT), "").strip("/")
    return tail.replace("/", "__")


def _predict_matrix_on_df(
    model_dirs: list[Path],
    fd: str,
    df: pd.DataFrame,
    device: str,
) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    used_dirs: list[Path] = []
    for md in model_dirs:
        try:
            pred = _predict_last_on_df(model_dir=md, fd=fd, df=df, device=device)
        except Exception:
            continue
        if pred["pred_rul"].isna().any():
            continue
        col = f"pred__{_safe_name(md)}"
        cur = pred.rename(columns={"pred_rul": col})
        frames.append(cur)
        used_dirs.append(md)
    if not frames:
        raise RuntimeError(f"No usable models for FD {fd}.")
    merged = frames[0]
    for fr in frames[1:]:
        merged = merged.merge(fr, on="unit", how="inner")
    return merged.sort_values("unit").reset_index(drop=True), used_dirs


def _predict_matrix_on_test(
    model_dirs: list[Path],
    data_dir: Path,
    fd: str,
    device: str,
) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    used_dirs: list[Path] = []
    for md in model_dirs:
        try:
            pred, _ = predict_last_cycle(model_dir=md, data_dir=data_dir, fd=fd, device=device)
        except Exception:
            continue
        if pred["pred_rul"].isna().any():
            continue
        col = f"pred__{_safe_name(md)}"
        cur = pred.rename(columns={"pred_rul": col})
        frames.append(cur)
        used_dirs.append(md)
    if not frames:
        raise RuntimeError(f"No usable test models for FD {fd}.")
    merged = frames[0]
    for fr in frames[1:]:
        merged = merged.merge(fr, on="unit", how="inner")
    return merged.sort_values("unit").reset_index(drop=True), used_dirs


def _group_indices(used_dirs: list[Path], groups: dict[str, set[Path]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for gname, gset in groups.items():
        out[gname] = [i for i, md in enumerate(used_dirs) if md in gset]
    return out


def _take_stats(mat: np.ndarray, idxs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not idxs:
        n = mat.shape[0]
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    sub = mat[:, idxs]
    return np.mean(sub, axis=1), np.std(sub, axis=1)


def _build_features(pred_matrix: np.ndarray, group_idxs: dict[str, list[int]], tail_threshold: float) -> tuple[np.ndarray, list[str]]:
    all_mean = np.mean(pred_matrix, axis=1)
    all_std = np.std(pred_matrix, axis=1)
    all_min = np.min(pred_matrix, axis=1)
    all_max = np.max(pred_matrix, axis=1)
    all_p10 = np.quantile(pred_matrix, 0.10, axis=1)
    all_p50 = np.quantile(pred_matrix, 0.50, axis=1)
    all_p90 = np.quantile(pred_matrix, 0.90, axis=1)
    all_spread = all_max - all_min
    below_tail_frac = np.mean(pred_matrix <= float(tail_threshold), axis=1)

    g_mean, g_std = _take_stats(pred_matrix, group_idxs.get("global", []))
    f13_mean, f13_std = _take_stats(pred_matrix, group_idxs.get("fd13", []))
    f24_mean, f24_std = _take_stats(pred_matrix, group_idxs.get("fd24", []))
    f2_mean, f2_std = _take_stats(pred_matrix, group_idxs.get("fd002", []))
    f4_mean, f4_std = _take_stats(pred_matrix, group_idxs.get("fd004", []))

    feats = np.column_stack(
        [
            all_mean,
            all_std,
            all_min,
            all_max,
            all_p10,
            all_p50,
            all_p90,
            all_spread,
            below_tail_frac,
            g_mean,
            g_std,
            f13_mean,
            f13_std,
            f24_mean,
            f24_std,
            f2_mean,
            f2_std,
            f4_mean,
            f4_std,
        ]
    ).astype(np.float64)
    names = [
        "all_mean",
        "all_std",
        "all_min",
        "all_max",
        "all_p10",
        "all_p50",
        "all_p90",
        "all_spread",
        "below_tail_frac",
        "global_mean",
        "global_std",
        "fd13_mean",
        "fd13_std",
        "fd24_mean",
        "fd24_std",
        "fd002_mean",
        "fd002_std",
        "fd004_mean",
        "fd004_std",
    ]
    return feats, names


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, tail_threshold: float) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
        "tail_rmse": rmse_tail(y_true, y_pred, rul_threshold=tail_threshold),
        "miss_rate": miss_rate(y_true, y_pred, rul_threshold=tail_threshold),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variance-aware meta-stacker over deep ensemble for multi-FD RUL.")
    p.add_argument("--global-model-dirs", required=True, help="Comma-separated global model dirs.")
    p.add_argument("--fd13-model-dirs", default="", help="Comma-separated FD13 specialist dirs.")
    p.add_argument("--fd24-model-dirs", default="", help="Comma-separated FD24 specialist dirs.")
    p.add_argument("--fd002-model-dirs", default="", help="Comma-separated FD002-only specialist dirs.")
    p.add_argument("--fd004-model-dirs", default="", help="Comma-separated FD004-only specialist dirs.")
    p.add_argument("--data-dir", default="RULdata/CMAPSSData")
    p.add_argument("--fds", default="FD001,FD002,FD003,FD004")
    p.add_argument("--val-min-prefix", type=int, default=20)
    p.add_argument("--val-seed", type=int, default=42)
    p.add_argument("--tail-threshold", type=float, default=30.0)
    p.add_argument("--xgb-n-estimators", type=int, default=240)
    p.add_argument("--xgb-max-depth", type=int, default=3)
    p.add_argument("--xgb-learning-rate", type=float, default=0.04)
    p.add_argument("--xgb-min-child-weight", type=float, default=5.0)
    p.add_argument("--xgb-subsample", type=float, default=0.9)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    p.add_argument("--xgb-reg-alpha", type=float, default=0.8)
    p.add_argument("--xgb-reg-lambda", type=float, default=8.0)
    p.add_argument("--xgb-random-state", type=int, default=42)
    p.add_argument("--fallback-margin", type=float, default=0.995, help="Require model val RMSE <= baseline * margin.")
    p.add_argument("--output-dir", default="outputs/meta_stacker_multifd_variance")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fds = _parse_fds(args.fds)

    global_dirs = _parse_csv_paths(args.global_model_dirs)
    fd13_dirs = _parse_csv_paths(args.fd13_model_dirs)
    fd24_dirs = _parse_csv_paths(args.fd24_model_dirs)
    fd002_dirs = _parse_csv_paths(args.fd002_model_dirs)
    fd004_dirs = _parse_csv_paths(args.fd004_model_dirs)
    if not global_dirs:
        raise ValueError("global_model_dirs cannot be empty.")

    max_rul = int(read_json(global_dirs[0] / "metadata.json").get("max_rul", 125))

    rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "global_model_dirs": [str(p) for p in global_dirs],
        "fd13_model_dirs": [str(p) for p in fd13_dirs],
        "fd24_model_dirs": [str(p) for p in fd24_dirs],
        "fd002_model_dirs": [str(p) for p in fd002_dirs],
        "fd004_model_dirs": [str(p) for p in fd004_dirs],
        "fds": fds,
        "tail_threshold": float(args.tail_threshold),
        "xgb": {
            "n_estimators": int(args.xgb_n_estimators),
            "max_depth": int(args.xgb_max_depth),
            "learning_rate": float(args.xgb_learning_rate),
            "min_child_weight": float(args.xgb_min_child_weight),
            "subsample": float(args.xgb_subsample),
            "colsample_bytree": float(args.xgb_colsample_bytree),
            "reg_alpha": float(args.xgb_reg_alpha),
            "reg_lambda": float(args.xgb_reg_lambda),
            "random_state": int(args.xgb_random_state),
        },
        "by_fd": {},
    }

    for fd in fds:
        # Per-FD deep ensemble pool.
        pool: list[Path] = []
        pool.extend(global_dirs)
        if fd in {"FD001", "FD003"}:
            pool.extend(fd13_dirs)
        if fd in {"FD002", "FD004"}:
            pool.extend(fd24_dirs)
        if fd == "FD002":
            pool.extend(fd002_dirs)
        if fd == "FD004":
            pool.extend(fd004_dirs)
        # Keep order and drop duplicates.
        seen: set[Path] = set()
        pool_unique = []
        for p in pool:
            if p not in seen:
                pool_unique.append(p)
                seen.add(p)

        train_raw = load_split(data_dir, fd_id=fd, split="train")
        train_with_target = add_train_rul(train_raw, max_rul=max_rul)
        val_obs, val_cuts = build_truncated_validation(
            train_with_target,
            min_prefix_cycles=int(args.val_min_prefix),
            random_state=int(args.val_seed) + 31 * (int(fd[-1]) - 1),
        )

        val_preds_df, used_val_dirs = _predict_matrix_on_df(model_dirs=pool_unique, fd=fd, df=val_obs, device=args.device)
        test_preds_df, used_test_dirs = _predict_matrix_on_test(model_dirs=pool_unique, data_dir=data_dir, fd=fd, device=args.device)
        used_dirs = [md for md in used_val_dirs if md in set(used_test_dirs)]
        if not used_dirs:
            raise RuntimeError(f"No intersecting usable models for FD {fd}.")

        # Recompute using intersected model set for exact alignment.
        val_preds_df, _ = _predict_matrix_on_df(model_dirs=used_dirs, fd=fd, df=val_obs, device=args.device)
        test_preds_df, _ = _predict_matrix_on_test(model_dirs=used_dirs, data_dir=data_dir, fd=fd, device=args.device)

        pred_cols = [c for c in val_preds_df.columns if c.startswith("pred__")]
        if len(pred_cols) < 2:
            raise RuntimeError(f"FD {fd}: need at least 2 usable models, got {len(pred_cols)}.")

        group_map: dict[str, set[Path]] = {
            "global": set(global_dirs),
            "fd13": set(fd13_dirs),
            "fd24": set(fd24_dirs),
            "fd002": set(fd002_dirs),
            "fd004": set(fd004_dirs),
        }
        idx_map = _group_indices(used_dirs=used_dirs, groups=group_map)

        merged_val = val_cuts[["unit", "true_rul_at_cut"]].merge(val_preds_df, on="unit", how="inner").sort_values("unit").reset_index(drop=True)
        y_val = merged_val["true_rul_at_cut"].to_numpy(dtype=np.float64)
        x_val_raw = merged_val[pred_cols].to_numpy(dtype=np.float64)
        x_val, feature_cols = _build_features(pred_matrix=x_val_raw, group_idxs=idx_map, tail_threshold=float(args.tail_threshold))

        test_last_units = select_last_cycle_rows(load_split(data_dir, fd_id=fd, split="test"))["unit"].astype(int).to_numpy()
        gt_rul = load_rul_targets(data_dir, fd_id=fd)["rul"].to_numpy(dtype=np.float64)
        gt_df = pd.DataFrame({"unit": test_last_units, "true_rul": gt_rul})
        merged_test = gt_df.merge(test_preds_df, on="unit", how="inner").sort_values("unit").reset_index(drop=True)
        y_test = merged_test["true_rul"].to_numpy(dtype=np.float64)
        x_test_raw = merged_test[pred_cols].to_numpy(dtype=np.float64)
        x_test, _ = _build_features(pred_matrix=x_test_raw, group_idxs=idx_map, tail_threshold=float(args.tail_threshold))

        baseline_val = x_val[:, 0]
        baseline_test = x_test[:, 0]

        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(args.xgb_n_estimators),
            max_depth=int(args.xgb_max_depth),
            learning_rate=float(args.xgb_learning_rate),
            min_child_weight=float(args.xgb_min_child_weight),
            subsample=float(args.xgb_subsample),
            colsample_bytree=float(args.xgb_colsample_bytree),
            reg_alpha=float(args.xgb_reg_alpha),
            reg_lambda=float(args.xgb_reg_lambda),
            random_state=int(args.xgb_random_state) + int(fd[-1]),
            tree_method="hist",
            n_jobs=0,
        )
        xgb.fit(x_val, y_val)
        pred_val_model = np.clip(xgb.predict(x_val).astype(np.float64), 0.0, float(max_rul))
        pred_test_model = np.clip(xgb.predict(x_test).astype(np.float64), 0.0, float(max_rul))

        val_rmse_model = float(np.sqrt(np.mean((y_val - pred_val_model) ** 2)))
        val_rmse_baseline = float(np.sqrt(np.mean((y_val - baseline_val) ** 2)))
        use_model = bool(val_rmse_model <= val_rmse_baseline * float(args.fallback_margin))
        pred_val_final = pred_val_model if use_model else baseline_val
        pred_test_final = pred_test_model if use_model else baseline_test

        met_global = _evaluate(y_test, x_test[:, 9], tail_threshold=float(args.tail_threshold))  # global_mean
        met_baseline = _evaluate(y_test, baseline_test, tail_threshold=float(args.tail_threshold))
        met_meta = _evaluate(y_test, pred_test_final, tail_threshold=float(args.tail_threshold))
        met_special = _evaluate(
            y_test,
            x_test[:, 13] if fd in {"FD002", "FD004"} else x_test[:, 11],  # fd24_mean or fd13_mean
            tail_threshold=float(args.tail_threshold),
        )

        rows.append(
            {
                "fd": fd,
                "n_models": int(len(pred_cols)),
                "val_rmse_baseline_mean": val_rmse_baseline,
                "val_rmse_model": val_rmse_model,
                "use_xgb_model": int(use_model),
                "test_rmse_global_mean": float(met_global["rmse"]),
                "test_rmse_special_mean": float(met_special["rmse"]),
                "test_rmse_baseline_mean": float(met_baseline["rmse"]),
                "test_rmse_meta": float(met_meta["rmse"]),
                "test_tail_rmse_meta": float(met_meta["tail_rmse"]),
                "test_miss_rate_meta": float(met_meta["miss_rate"]),
            }
        )

        state["by_fd"][fd] = {
            "used_model_dirs": [str(p) for p in used_dirs],
            "feature_cols": feature_cols,
            "group_counts": {k: int(len(v)) for k, v in idx_map.items()},
            "val_rmse_baseline_mean": val_rmse_baseline,
            "val_rmse_model": val_rmse_model,
            "use_xgb_model": bool(use_model),
            "fallback_margin": float(args.fallback_margin),
            "test_metrics_meta": met_meta,
            "test_metrics_baseline_mean": met_baseline,
            "test_metrics_global_mean": met_global,
            "test_metrics_special_mean": met_special,
        }

        pred_out = merged_test[["unit"]].copy()
        pred_out["pred_rul"] = pred_test_final.astype(np.float64)
        pred_out.to_csv(output_dir / f"pred_{fd}_meta_variance.csv", index=False)

    df = pd.DataFrame(rows).sort_values("fd").reset_index(drop=True)
    macro = float(df["test_rmse_meta"].mean())
    worst = float(df["test_rmse_meta"].max())
    tail_macro = float(df["test_tail_rmse_meta"].mean())

    df.to_csv(output_dir / "meta_stacker_variance_metrics.csv", index=False)
    state["macro_rmse_meta"] = macro
    state["worst_fd_rmse_meta"] = worst
    state["macro_tail_rmse_meta"] = tail_macro
    write_json(output_dir / "meta_stacker_variance_state.json", state)

    print(f"Saved: {output_dir / 'meta_stacker_variance_metrics.csv'}")
    print(f"Saved: {output_dir / 'meta_stacker_variance_state.json'}")
    print(
        f"Meta-variance summary: test_rmse_macro={macro:.4f} "
        f"test_tail_rmse_macro={tail_macro:.4f} worst_fd_rmse={worst:.4f}"
    )


if __name__ == "__main__":
    main()
