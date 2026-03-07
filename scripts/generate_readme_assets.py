from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "RULdata" / "CMAPSSData"
FIG_DIR = ROOT / "docs" / "figures"
TAB_DIR = ROOT / "docs" / "tables"


def _load_tuning_best(path: Path, stage_name: str) -> dict[str, float | str]:
    df = pd.read_csv(path)
    best = df.sort_values("test_rmse_macro", ascending=True).iloc[0]
    return {
        "stage": stage_name,
        "macro_rmse": float(best["test_rmse_macro"]),
        "fd001_rmse": float(best.get("test_rmse_FD001", np.nan)),
        "fd002_rmse": float(best.get("test_rmse_FD002", np.nan)),
        "fd003_rmse": float(best.get("test_rmse_FD003", np.nan)),
        "fd004_rmse": float(best.get("test_rmse_FD004", np.nan)),
        "worst_fd_rmse": float(best.get("test_rmse_worst_fd", np.nan)),
        "artifact": str(path.relative_to(ROOT)),
    }


def _load_hierarchical(path: Path, stage_name: str) -> dict[str, float | str]:
    df = pd.read_csv(path).sort_values("fd")
    m = {row["fd"]: float(row["test_rmse_blend"]) for _, row in df.iterrows()}
    return {
        "stage": stage_name,
        "macro_rmse": float(df["test_rmse_blend"].mean()),
        "fd001_rmse": float(m.get("FD001", np.nan)),
        "fd002_rmse": float(m.get("FD002", np.nan)),
        "fd003_rmse": float(m.get("FD003", np.nan)),
        "fd004_rmse": float(m.get("FD004", np.nan)),
        "worst_fd_rmse": float(df["test_rmse_blend"].max()),
        "artifact": str(path.relative_to(ROOT)),
    }


def _load_meta_variance(path: Path, stage_name: str) -> dict[str, float | str]:
    df = pd.read_csv(path).sort_values("fd")
    m = {row["fd"]: float(row["test_rmse_meta"]) for _, row in df.iterrows()}
    return {
        "stage": stage_name,
        "macro_rmse": float(df["test_rmse_meta"].mean()),
        "fd001_rmse": float(m.get("FD001", np.nan)),
        "fd002_rmse": float(m.get("FD002", np.nan)),
        "fd003_rmse": float(m.get("FD003", np.nan)),
        "fd004_rmse": float(m.get("FD004", np.nan)),
        "worst_fd_rmse": float(df["test_rmse_meta"].max()),
        "artifact": str(path.relative_to(ROOT)),
    }


def _load_truth(fd: str) -> pd.DataFrame:
    rul_path = DATA_DIR / f"RUL_{fd}.txt"
    test_path = DATA_DIR / f"test_{fd}.txt"
    rul = pd.read_csv(rul_path, header=None, names=["rul"], sep=r"\s+", engine="python")
    test = pd.read_csv(test_path, header=None, sep=r"\s+", engine="python")
    test = test.drop(columns=[c for c in test.columns if test[c].isna().all()])
    test.columns = ["unit", "cycle"] + [f"x{i}" for i in range(1, test.shape[1] - 1)]
    last_units = test.groupby("unit", as_index=False)["cycle"].max().sort_values("unit")
    gt = pd.DataFrame({"unit": last_units["unit"].astype(int), "true_rul": rul["rul"].astype(float)})
    return gt.sort_values("unit").reset_index(drop=True)


def _save_progression_plot(summary: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    y = summary["macro_rmse"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, y, color=["#6e6e6e", "#7ba4ff", "#3f7fdb", "#1b4da8", "#c07b39"])
    plt.xticks(x, summary["stage"], rotation=20, ha="right")
    plt.ylabel("Macro RMSE (lower is better)")
    plt.title("RUL Progression Across Iterations")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for b, val in zip(bars, y):
        plt.text(b.get_x() + b.get_width() / 2, val + 0.2, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rmse_progression.png", dpi=150)
    plt.close()


def _save_per_fd_plot(summary: pd.DataFrame) -> None:
    plot_df = summary[summary["fd001_rmse"].notna()].copy()
    labels = ["FD001", "FD002", "FD003", "FD004"]
    cols = ["fd001_rmse", "fd002_rmse", "fd003_rmse", "fd004_rmse"]
    x = np.arange(len(labels))
    width = 0.25

    stages = plot_df["stage"].tolist()
    vals = [plot_df[c].to_numpy(dtype=float) for c in cols]

    plt.figure(figsize=(10, 5))
    for i, (stage, offset) in enumerate(zip(stages[-3:], [-width, 0.0, width])):
        stage_row = plot_df[plot_df["stage"] == stage].iloc[0]
        y = [float(stage_row[c]) for c in cols]
        plt.bar(x + offset, y, width=width, label=stage)
    plt.xticks(x, labels)
    plt.ylabel("RMSE")
    plt.title("Per-FD RMSE Comparison (Recent Stages)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "per_fd_comparison.png", dpi=150)
    plt.close()


def _save_fd004_scatter_and_hist(best_pred_path: Path) -> None:
    gt = _load_truth("FD004")
    pred = pd.read_csv(best_pred_path).rename(columns={"pred_rul": "pred_rul"})
    df = gt.merge(pred, on="unit", how="inner").sort_values("unit").reset_index(drop=True)
    df["err"] = df["pred_rul"] - df["true_rul"]
    df["abs_err"] = df["err"].abs()
    df.to_csv(TAB_DIR / "fd004_best_errors.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(df["true_rul"], df["pred_rul"], s=12, alpha=0.7, color="#1f77b4")
    low, high = 0.0, max(df["true_rul"].max(), df["pred_rul"].max()) + 5.0
    plt.plot([low, high], [low, high], "--", color="black", linewidth=1)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("FD004: Predicted vs True (Best Global System)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fd004_pred_vs_true.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(df["err"], bins=25, color="#2a9d8f", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Prediction Error (pred - true)")
    plt.ylabel("Count")
    plt.title("FD004 Error Distribution (Best Global System)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fd004_error_hist.png", dpi=150)
    plt.close()


def _save_uncertainty_scatter() -> None:
    gt = _load_truth("FD004")
    pred_paths = [
        ROOT / "outputs" / "hierarchical_ensemble_cross_ensglobal_newfd004v4_t7s42_w101_cuda" / "pred_FD004_blend.csv",
        ROOT / "outputs" / "hierarchical_ensemble_cross_ensglobal_newfd004v4_t7s43_w101_cuda" / "pred_FD004_blend.csv",
        ROOT / "outputs" / "hierarchical_ensemble_cross_ensglobal_newfd004v4_t7s44_w101_cuda" / "pred_FD004_blend.csv",
    ]
    frames = []
    for i, p in enumerate(pred_paths):
        cur = pd.read_csv(p).rename(columns={"pred_rul": f"p{i}"})
        frames.append(cur)
    merged = frames[0]
    for fr in frames[1:]:
        merged = merged.merge(fr, on="unit", how="inner")
    merged = gt.merge(merged, on="unit", how="inner").sort_values("unit").reset_index(drop=True)
    pred_cols = ["p0", "p1", "p2"]
    arr = merged[pred_cols].to_numpy(dtype=float)
    merged["pred_mean"] = arr.mean(axis=1)
    merged["pred_std"] = arr.std(axis=1)
    merged["abs_err_mean"] = (merged["pred_mean"] - merged["true_rul"]).abs()
    merged.to_csv(TAB_DIR / "fd004_uncertainty_vs_error.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(merged["pred_std"], merged["abs_err_mean"], s=14, alpha=0.7, color="#e76f51")
    z = np.polyfit(merged["pred_std"], merged["abs_err_mean"], 1)
    xx = np.linspace(merged["pred_std"].min(), merged["pred_std"].max(), 100)
    plt.plot(xx, z[0] * xx + z[1], color="black", linewidth=1.2, linestyle="--")
    plt.xlabel("Ensemble Std (FD004 specialists)")
    plt.ylabel("|Error| of mean prediction")
    plt.title("FD004 Uncertainty vs Error")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fd004_uncertainty_vs_error.png", dpi=150)
    plt.close()


def _save_ablation_table(summary: pd.DataFrame) -> None:
    base = float(summary.iloc[0]["macro_rmse"])
    rows = []
    for _, row in summary.iterrows():
        cur = float(row["macro_rmse"])
        rows.append(
            {
                "stage": row["stage"],
                "macro_rmse": cur,
                "delta_vs_baseline": cur - base,
                "improvement_vs_baseline_pct": (1.0 - cur / base) * 100.0,
            }
        )
    pd.DataFrame(rows).to_csv(TAB_DIR / "ablation_lift_summary.csv", index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _load_tuning_best(
        ROOT / "outputs" / "tuning_hybrid_multifd_v3_selection" / "hybrid_multifd_tuning.csv",
        "Simple Baseline",
    )
    single = _load_tuning_best(
        ROOT / "outputs" / "tuning_hybrid_multifd_v11_encoder_mix" / "hybrid_multifd_tuning.csv",
        "Single Model Best",
    )
    hierarchical_prev = _load_hierarchical(
        ROOT / "outputs" / "hierarchical_ensemble_cross_oldglobal_stagefd24_w51" / "hierarchical_ensemble_metrics.csv",
        "Hierarchical (early)",
    )
    hierarchical_best = _load_hierarchical(
        ROOT
        / "outputs"
        / "hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101"
        / "hierarchical_ensemble_metrics.csv",
        "Hierarchical Best",
    )
    meta_variance = _load_meta_variance(
        ROOT / "outputs" / "meta_stacker_variance_loop1" / "meta_stacker_variance_metrics.csv",
        "Meta-Stacker (variance, experiment)",
    )

    summary = pd.DataFrame([baseline, single, hierarchical_prev, hierarchical_best, meta_variance])
    summary["delta_vs_baseline"] = summary["macro_rmse"] - float(summary.iloc[0]["macro_rmse"])
    summary["improvement_vs_baseline_pct"] = (1.0 - summary["macro_rmse"] / float(summary.iloc[0]["macro_rmse"])) * 100.0
    summary.to_csv(TAB_DIR / "summary_results.csv", index=False)

    best_per_fd = summary[summary["stage"] == "Hierarchical Best"][
        ["fd001_rmse", "fd002_rmse", "fd003_rmse", "fd004_rmse", "macro_rmse", "worst_fd_rmse", "artifact"]
    ]
    best_per_fd.to_csv(TAB_DIR / "best_per_fd.csv", index=False)

    _save_progression_plot(summary)
    _save_per_fd_plot(summary)
    _save_fd004_scatter_and_hist(
        ROOT
        / "outputs"
        / "hierarchical_ensemble_cross_ensglobal_newfd13_fd24v3full_t2s43_w101"
        / "pred_FD004_blend.csv"
    )
    _save_uncertainty_scatter()
    _save_ablation_table(summary)
    print("Saved tables to:", TAB_DIR)
    print("Saved figures to:", FIG_DIR)


if __name__ == "__main__":
    main()
