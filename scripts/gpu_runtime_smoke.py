#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.sequence_model import LSTMRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke/benchmark checks for CUDA data and training runtime path.")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--num-samples", type=int, default=8192, help="Synthetic sample count.")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length.")
    parser.add_argument("--n-features", type=int, default=24, help="Feature count.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--output-json", default="outputs/gpu_runtime_smoke.json", help="Where to save JSON results.")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_dataset(num_samples: int, seq_len: int, n_features: int) -> TensorDataset:
    x = torch.randn(num_samples, seq_len, n_features, dtype=torch.float32)
    y = torch.randn(num_samples, dtype=torch.float32)
    return TensorDataset(x, y)


def benchmark_transfer(
    ds: TensorDataset,
    device: str,
    batch_size: int,
    pin_memory: bool,
    non_blocking: bool,
) -> float:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    start = time.perf_counter()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=non_blocking)
        yb = yb.to(device, non_blocking=non_blocking)
        _ = xb, yb
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - start


def run_amp_step(ds: TensorDataset, device: str, batch_size: int) -> bool:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    model = LSTMRegressor(input_size=ds.tensors[0].shape[-1], hidden_size=64, num_layers=1, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    model.train()
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            pred = model(xb)
            loss = criterion(pred, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        if steps >= 5:
            break

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return True


def run_tf32_check(device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    _ = a @ b
    torch.cuda.synchronize()
    return True


def run_cudnn_benchmark_check(ds: TensorDataset, device: str, batch_size: int) -> bool:
    if not device.startswith("cuda"):
        return False
    torch.backends.cudnn.benchmark = True
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    model = LSTMRegressor(input_size=ds.tensors[0].shape[-1], hidden_size=32, num_layers=1, dropout=0.0).to(device)
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            _ = model(xb)
            break
    torch.cuda.synchronize()
    return True


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ds = make_dataset(args.num_samples, args.seq_len, args.n_features)

    results: dict[str, object] = {
        "torch_version": torch.__version__,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.startswith("cuda"):
        results["cuda_name"] = torch.cuda.get_device_name(0)

    if not device.startswith("cuda"):
        results["note"] = "CUDA not available on selected device, GPU smoke checks skipped."
        output_path = (ROOT / args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return

    t_no_pin = benchmark_transfer(ds, device, args.batch_size, pin_memory=False, non_blocking=False)
    t_pin_nb = benchmark_transfer(ds, device, args.batch_size, pin_memory=True, non_blocking=True)

    amp_ok = run_amp_step(ds, device, args.batch_size)
    tf32_ok = run_tf32_check(device)
    cudnn_ok = run_cudnn_benchmark_check(ds, device, args.batch_size)

    improvement = ((t_no_pin - t_pin_nb) / t_no_pin * 100.0) if t_no_pin > 0 else 0.0
    results.update(
        {
            "transfer_seconds_no_pin_blocking": round(t_no_pin, 6),
            "transfer_seconds_pin_non_blocking": round(t_pin_nb, 6),
            "transfer_improvement_percent": round(improvement, 2),
            "amp_smoke_ok": amp_ok,
            "tf32_smoke_ok": tf32_ok,
            "cudnn_benchmark_smoke_ok": cudnn_ok,
        }
    )

    output_path = (ROOT / args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
