from __future__ import annotations

import numpy as np
import torch

from rul_pipeline.hybrid_sequence_model import (
    ConvAttentionLSTMConfig,
    ConvAttentionLSTMRegressor,
    predict_conv_attention_lstm,
    train_conv_attention_lstm_regressor,
)


def test_hybrid_forward_shape() -> None:
    model = ConvAttentionLSTMRegressor(
        input_size=8,
        conv_channels=16,
        kernel_size=3,
        attention_heads=4,
        hidden_size=12,
        num_layers=1,
        dropout=0.1,
    )
    x = torch.randn(5, 10, 8)
    y = model(x)
    assert y.shape == (5,)


def test_hybrid_train_predict_smoke() -> None:
    rng = np.random.default_rng(123)
    x_train = rng.normal(size=(64, 12, 8)).astype(np.float32)
    y_train = rng.normal(size=(64,)).astype(np.float32)
    x_valid = rng.normal(size=(16, 12, 8)).astype(np.float32)
    y_valid = rng.normal(size=(16,)).astype(np.float32)

    cfg = ConvAttentionLSTMConfig(
        input_size=8,
        conv_channels=16,
        kernel_size=3,
        attention_heads=4,
        hidden_size=12,
        num_layers=1,
        dropout=0.1,
        epochs=2,
        batch_size=16,
        patience=2,
        use_amp=False,
        pin_memory=False,
        non_blocking=False,
    )
    model, history, resolved_device = train_conv_attention_lstm_regressor(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        cfg=cfg,
        device="cpu",
    )
    preds = predict_conv_attention_lstm(model, x_valid, batch_size=8, device=resolved_device)

    assert len(history) >= 1
    assert preds.shape == (16,)
