from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


class QuantumLayer(nn.Module):
    """PennyLane-backed quantum layer exposed as a torch.nn.Module."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        backend: str = "default.qubit",
        diff_method: str = "parameter-shift",
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        device = qml.device(backend, wires=n_qubits)

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_q_layers, n_qubits, 3)}
        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
