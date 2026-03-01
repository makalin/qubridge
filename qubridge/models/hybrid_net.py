from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from qubridge.models.classical import ClassicalFeatureExtractor
from qubridge.models.quantum_layer import QuantumLayer


class QuBridgeModel(nn.Module):
    """Hybrid classical-quantum neural network for classification."""

    def __init__(
        self,
        input_dim: int = 256,
        n_classes: int = 2,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.1,
        backend: str = "default.qubit",
    ) -> None:
        super().__init__()

        self.feature_extractor = ClassicalFeatureExtractor(
            input_dim=input_dim,
            output_dim=n_qubits,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            backend=backend,
        )
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        classical_features = self.feature_extractor(x)
        quantum_features = self.quantum_layer(classical_features)
        return quantum_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        features = self.forward_features(x)
        return self.classifier(features)
