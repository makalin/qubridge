from __future__ import annotations

import torch

from qubridge.models.hybrid_net import QuBridgeModel


def main() -> None:
    torch.manual_seed(0)

    model = QuBridgeModel(
        input_dim=8,
        n_classes=2,
        n_qubits=4,
        n_q_layers=2,
        hidden_dims=(16, 8),
        dropout=0.0,
    )

    batch = torch.randn(5, 8)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)

    print("Logits:")
    print(logits)
    print("Probabilities:")
    print(probs)


if __name__ == "__main__":
    main()
