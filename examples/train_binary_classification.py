from __future__ import annotations

from qubridge.data.synthetic import make_binary_moons
from qubridge.train import TrainConfig, train_and_evaluate


def main() -> None:
    data = make_binary_moons(n_samples=1000, noise=0.15)

    config = TrainConfig(
        input_dim=2,
        n_classes=2,
        n_qubits=4,
        n_q_layers=2,
        hidden_dims=(32, 16),
        epochs=10,
        batch_size=32,
        lr=0.01,
    )

    result = train_and_evaluate(data, config)
    metrics = result["test_metrics"]

    print(f"Final test loss: {metrics['loss']:.4f}")
    print(f"Final test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
