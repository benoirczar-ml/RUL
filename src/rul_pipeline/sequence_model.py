from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 12
    batch_size: int = 256
    patience: int = 3
    random_state: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    non_blocking: bool = True
    use_amp: bool = True
    enable_tf32: bool = True
    cudnn_benchmark: bool = True


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.regressor(h_last).squeeze(-1)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def _configure_gpu_runtime(cfg: LSTMConfig, resolved_device: str) -> None:
    if not _is_cuda_device(resolved_device):
        return
    if cfg.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except RuntimeError:
            pass
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


def train_lstm_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    cfg: LSTMConfig,
    device: str = "auto",
) -> tuple[LSTMRegressor, list[dict[str, float]], str]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    resolved_device = resolve_device(device)
    use_cuda = _is_cuda_device(resolved_device)
    _configure_gpu_runtime(cfg, resolved_device)
    model = LSTMRegressor(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(resolved_device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": max(0, int(cfg.num_workers)),
        "pin_memory": bool(cfg.pin_memory and use_cuda),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_ds,
        **loader_kwargs,
    )

    copy_non_blocking = bool(cfg.non_blocking and use_cuda)
    x_valid_t = torch.from_numpy(x_valid).float().to(resolved_device, non_blocking=copy_non_blocking)
    y_valid_np = y_valid.astype(np.float32)
    use_amp = bool(cfg.use_amp and use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = None
    best_rmse = float("inf")
    no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(resolved_device, non_blocking=copy_non_blocking)
            yb = yb.to(resolved_device, non_blocking=copy_non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            y_valid_pred = model(x_valid_t).detach().cpu().numpy()
        valid_rmse = float(np.sqrt(np.mean((y_valid_np - y_valid_pred) ** 2)))
        valid_mae = float(np.mean(np.abs(y_valid_np - y_valid_pred)))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "valid_rmse": valid_rmse,
                "valid_mae": valid_mae,
            }
        )

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, resolved_device


def predict_lstm(
    model: LSTMRegressor,
    x: np.ndarray,
    batch_size: int,
    device: str,
    non_blocking: bool = False,
    pin_memory: bool | None = None,
) -> np.ndarray:
    model.eval()
    use_cuda = _is_cuda_device(device)
    copy_non_blocking = bool(non_blocking and use_cuda)
    use_pin_memory = bool(use_cuda if pin_memory is None else (pin_memory and use_cuda))
    x_t = torch.from_numpy(x).float()
    ds = TensorDataset(x_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=use_pin_memory)

    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=copy_non_blocking)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def save_lstm_checkpoint(model: LSTMRegressor, path: str) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "arch": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
        },
    }
    torch.save(payload, path)


def load_lstm_checkpoint(path: str, device: str = "auto") -> tuple[LSTMRegressor, str]:
    resolved_device = resolve_device(device)
    try:
        ckpt = torch.load(path, map_location=resolved_device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=resolved_device)
    arch = ckpt["arch"]
    model = LSTMRegressor(
        input_size=int(arch["input_size"]),
        hidden_size=int(arch["hidden_size"]),
        num_layers=int(arch["num_layers"]),
        dropout=float(arch["dropout"]),
    ).to(resolved_device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, resolved_device
