from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class ClassicalFeatureExtractor(nn.Module):
    """Maps classical input vectors into a compact representation for quantum encoding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []

        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
