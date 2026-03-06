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
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    x_valid_t = torch.from_numpy(x_valid).float().to(resolved_device)
    y_valid_np = y_valid.astype(np.float32)

    best_state = None
    best_rmse = float("inf")
    no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(resolved_device)
            yb = yb.to(resolved_device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
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


def predict_lstm(model: LSTMRegressor, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    x_t = torch.from_numpy(x).float()
    ds = TensorDataset(x_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
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
