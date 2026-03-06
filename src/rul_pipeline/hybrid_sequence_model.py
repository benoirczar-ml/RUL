from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass
class ConvAttentionLSTMConfig:
    input_size: int
    conv_channels: int = 96
    kernel_size: int = 5
    attention_heads: int = 4
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.2
    num_fd_heads: int = 4
    loss_name: str = "huber_asymmetric"
    huber_delta: float = 10.0
    late_error_weight: float = 0.35
    late_error_margin: float = 0.0
    emphasize_failure: bool = True
    failure_rul_threshold: int = 30
    failure_weight: float = 2.0
    sampling_strategy: str = "auto"
    fd_balance_power: float = 1.0
    warmup_mse_epochs: int = 0
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
    log_every_epoch: bool = True


class _RegressorHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class ConvAttentionLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        conv_channels: int,
        kernel_size: int,
        attention_heads: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_fd_heads: int = 1,
    ) -> None:
        super().__init__()
        if conv_channels % attention_heads != 0:
            raise ValueError(
                f"conv_channels must be divisible by attention_heads, got {conv_channels} and {attention_heads}"
            )
        if num_fd_heads < 1:
            raise ValueError("num_fd_heads must be >= 1.")

        self.input_size = input_size
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_fd_heads = num_fd_heads

        self.temporal_conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=conv_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(conv_channels)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.shared_head = _RegressorHead(hidden_size)
        self.fd_heads = nn.ModuleList([_RegressorHead(hidden_size) for _ in range(num_fd_heads)])

    def forward(self, x: torch.Tensor, fd_idx: torch.Tensor | None = None) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = torch.relu(self.temporal_conv(h))
        h = h.transpose(1, 2)

        h_attn, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_norm(h + self.attn_dropout(h_attn))

        h_lstm, _ = self.lstm(h)
        h_last = h_lstm[:, -1, :]

        if self.num_fd_heads == 1:
            return self.shared_head(h_last)

        if fd_idx is None:
            raise ValueError("fd_idx is required when num_fd_heads > 1.")
        if fd_idx.ndim == 0:
            fd_idx = fd_idx.view(1).repeat(h_last.shape[0])
        if fd_idx.shape[0] != h_last.shape[0]:
            raise ValueError(f"fd_idx length mismatch: {fd_idx.shape[0]} vs batch {h_last.shape[0]}")

        out = self.shared_head(h_last)
        for idx in torch.unique(fd_idx):
            head_id = int(idx.item())
            if head_id < 0 or head_id >= self.num_fd_heads:
                raise ValueError(f"fd_idx={head_id} out of range [0, {self.num_fd_heads - 1}]")
            mask = fd_idx == idx
            if mask.any():
                out[mask] = self.fd_heads[head_id](h_last[mask])
        return out


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def _configure_gpu_runtime(cfg: ConvAttentionLSTMConfig, resolved_device: str) -> None:
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


def _asymmetric_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float,
    late_error_weight: float,
    late_error_margin: float,
) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    base = 0.5 * quad**2 + delta * lin
    late = (err > late_error_margin).to(base.dtype)
    weights = 1.0 + late_error_weight * late
    return (base * weights).mean()


def _compute_loss(
    cfg: ConvAttentionLSTMConfig,
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str | None = None,
) -> torch.Tensor:
    active_loss = cfg.loss_name if loss_name is None else str(loss_name)
    if active_loss == "mse":
        return nn.functional.mse_loss(pred, target)
    if active_loss == "huber_asymmetric":
        return _asymmetric_huber_loss(
            pred=pred,
            target=target,
            delta=float(cfg.huber_delta),
            late_error_weight=float(cfg.late_error_weight),
            late_error_margin=float(cfg.late_error_margin),
        )
    raise ValueError(f"Unsupported loss_name={active_loss}")


def _resolve_sampling_strategy(cfg: ConvAttentionLSTMConfig) -> str:
    strategy = str(cfg.sampling_strategy).lower()
    if strategy == "auto":
        return "failure_weighted" if cfg.emphasize_failure else "shuffle"
    return strategy


def _build_train_loader(
    x_train: np.ndarray,
    y_train: np.ndarray,
    fd_train: np.ndarray | None,
    cfg: ConvAttentionLSTMConfig,
    use_cuda: bool,
) -> DataLoader:
    x_t = torch.from_numpy(x_train).float()
    y_t = torch.from_numpy(y_train).float()
    if fd_train is not None:
        fd_t = torch.from_numpy(fd_train.astype(np.int64))
        train_ds = TensorDataset(x_t, y_t, fd_t)
    else:
        train_ds = TensorDataset(x_t, y_t)

    loader_kwargs: dict[str, object] = {
        "batch_size": cfg.batch_size,
        "drop_last": False,
        "num_workers": max(0, int(cfg.num_workers)),
        "pin_memory": bool(cfg.pin_memory and use_cuda),
    }
    if int(loader_kwargs["num_workers"]) > 0:
        loader_kwargs["persistent_workers"] = True

    strategy = _resolve_sampling_strategy(cfg)
    if strategy == "shuffle":
        loader_kwargs["shuffle"] = True
        return DataLoader(train_ds, **loader_kwargs)

    if strategy not in {"failure_weighted", "balanced_fd_failure"}:
        raise ValueError(f"Unsupported sampling_strategy={cfg.sampling_strategy}")

    weights = np.ones_like(y_train, dtype=np.float64)
    if strategy == "balanced_fd_failure":
        if fd_train is None:
            raise ValueError("fd_train is required for balanced_fd_failure sampling_strategy.")
        unique_fd, counts_fd = np.unique(fd_train.astype(np.int64), return_counts=True)
        balance_power = max(0.0, float(cfg.fd_balance_power))
        for fd_id, fd_count in zip(unique_fd, counts_fd):
            if fd_count > 0:
                weights[fd_train == fd_id] *= 1.0 / (float(fd_count) ** balance_power)

    if cfg.emphasize_failure:
        weights[y_train <= float(cfg.failure_rul_threshold)] *= float(cfg.failure_weight)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=int(len(weights)),
        replacement=True,
    )
    loader_kwargs["sampler"] = sampler

    return DataLoader(train_ds, **loader_kwargs)


def train_conv_attention_lstm_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    cfg: ConvAttentionLSTMConfig,
    device: str = "auto",
    fd_train: np.ndarray | None = None,
    fd_valid: np.ndarray | None = None,
) -> tuple[ConvAttentionLSTMRegressor, list[dict[str, float]], str]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    if cfg.num_fd_heads > 1 and (fd_train is None or fd_valid is None):
        raise ValueError("fd_train and fd_valid are required when num_fd_heads > 1.")

    resolved_device = resolve_device(device)
    use_cuda = _is_cuda_device(resolved_device)
    _configure_gpu_runtime(cfg, resolved_device)

    model = ConvAttentionLSTMRegressor(
        input_size=cfg.input_size,
        conv_channels=cfg.conv_channels,
        kernel_size=cfg.kernel_size,
        attention_heads=cfg.attention_heads,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        num_fd_heads=cfg.num_fd_heads,
    ).to(resolved_device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_loader = _build_train_loader(
        x_train=x_train,
        y_train=y_train,
        fd_train=fd_train,
        cfg=cfg,
        use_cuda=use_cuda,
    )

    copy_non_blocking = bool(cfg.non_blocking and use_cuda)
    x_valid_t = torch.from_numpy(x_valid).float().to(resolved_device, non_blocking=copy_non_blocking)
    y_valid_np = y_valid.astype(np.float32)
    fd_valid_t = None
    if fd_valid is not None:
        fd_valid_t = torch.from_numpy(fd_valid.astype(np.int64)).to(resolved_device, non_blocking=copy_non_blocking)

    use_amp = bool(cfg.use_amp and use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = None
    best_rmse = float("inf")
    no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        epoch_loss_name = "mse" if epoch <= max(0, int(cfg.warmup_mse_epochs)) else cfg.loss_name
        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, fb = batch
            else:
                xb, yb = batch
                fb = None

            xb = xb.to(resolved_device, non_blocking=copy_non_blocking)
            yb = yb.to(resolved_device, non_blocking=copy_non_blocking)
            if fb is not None:
                fb = fb.to(resolved_device, non_blocking=copy_non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(xb, fd_idx=fb)
                loss = _compute_loss(cfg, pred, yb, loss_name=epoch_loss_name)
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
            y_valid_pred = model(x_valid_t, fd_idx=fd_valid_t).detach().cpu().numpy()

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

        improved = valid_rmse < best_rmse
        if improved:
            best_rmse = valid_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if cfg.log_every_epoch:
            print(
                f"epoch {epoch:03d}/{cfg.epochs:03d} "
                f"train_loss={train_loss:.6f} "
                f"loss={epoch_loss_name} "
                f"valid_rmse={valid_rmse:.6f} "
                f"valid_mae={valid_mae:.6f} "
                f"best_rmse={best_rmse:.6f} "
                f"patience={no_improve}/{cfg.patience} "
                f"improved={'yes' if improved else 'no'}",
                flush=True,
            )

        if no_improve >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, resolved_device


def predict_conv_attention_lstm(
    model: ConvAttentionLSTMRegressor,
    x: np.ndarray,
    batch_size: int,
    device: str,
    non_blocking: bool = False,
    pin_memory: bool | None = None,
    fd_idx: int | np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    use_cuda = _is_cuda_device(device)
    copy_non_blocking = bool(non_blocking and use_cuda)
    use_pin_memory = bool(use_cuda if pin_memory is None else (pin_memory and use_cuda))

    x_t = torch.from_numpy(x).float()
    if model.num_fd_heads > 1:
        if fd_idx is None:
            raise ValueError("fd_idx is required for predict_conv_attention_lstm when num_fd_heads > 1.")
        if isinstance(fd_idx, np.ndarray):
            if len(fd_idx) != len(x):
                raise ValueError(f"fd_idx length mismatch: {len(fd_idx)} vs {len(x)}")
            fd_arr = fd_idx.astype(np.int64)
        else:
            fd_arr = np.full(len(x), int(fd_idx), dtype=np.int64)
        fd_t = torch.from_numpy(fd_arr)
        ds = TensorDataset(x_t, fd_t)
    else:
        ds = TensorDataset(x_t)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=use_pin_memory)

    preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                xb, fb = batch
                fb = fb.to(device, non_blocking=copy_non_blocking)
            else:
                (xb,) = batch
                fb = None
            xb = xb.to(device, non_blocking=copy_non_blocking)
            preds.append(model(xb, fd_idx=fb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def save_conv_attention_lstm_checkpoint(model: ConvAttentionLSTMRegressor, path: str) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "arch": {
            "input_size": model.input_size,
            "conv_channels": model.conv_channels,
            "kernel_size": model.kernel_size,
            "attention_heads": model.attention_heads,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "num_fd_heads": model.num_fd_heads,
        },
    }
    torch.save(payload, path)


def load_conv_attention_lstm_checkpoint(path: str, device: str = "auto") -> tuple[ConvAttentionLSTMRegressor, str]:
    resolved_device = resolve_device(device)
    try:
        ckpt = torch.load(path, map_location=resolved_device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=resolved_device)
    arch = ckpt["arch"]
    state_dict = ckpt["state_dict"]
    # Backward compatibility for older checkpoints that used a single "regressor" head.
    if "num_fd_heads" not in arch:
        arch["num_fd_heads"] = 1
    if any(k.startswith("regressor.") for k in state_dict.keys()):
        remapped: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("regressor."):
                remapped[k.replace("regressor.", "shared_head.net.", 1)] = v
            else:
                remapped[k] = v
        # For old single-head checkpoints, mirror shared head weights to fd_heads[0].
        head_pairs = [
            ("shared_head.net.0.weight", "fd_heads.0.net.0.weight"),
            ("shared_head.net.0.bias", "fd_heads.0.net.0.bias"),
            ("shared_head.net.2.weight", "fd_heads.0.net.2.weight"),
            ("shared_head.net.2.bias", "fd_heads.0.net.2.bias"),
        ]
        for src, dst in head_pairs:
            if src in remapped and dst not in remapped:
                remapped[dst] = remapped[src].clone()
        state_dict = remapped

    model = ConvAttentionLSTMRegressor(
        input_size=int(arch["input_size"]),
        conv_channels=int(arch["conv_channels"]),
        kernel_size=int(arch["kernel_size"]),
        attention_heads=int(arch["attention_heads"]),
        hidden_size=int(arch["hidden_size"]),
        num_layers=int(arch["num_layers"]),
        dropout=float(arch["dropout"]),
        num_fd_heads=int(arch.get("num_fd_heads", 1)),
    ).to(resolved_device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, resolved_device
