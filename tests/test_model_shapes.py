from __future__ import annotations

import torch

from qubridge.models.hybrid_net import QuBridgeModel


def test_forward_shape() -> None:
    model = QuBridgeModel(input_dim=6, n_classes=3, n_qubits=4, n_q_layers=1, hidden_dims=(12,))
    x = torch.randn(7, 6)
    out = model(x)
    assert out.shape == (7, 3)
