from __future__ import annotations

from qubridge.data.real import make_breast_cancer_split
from qubridge.train import TrainConfig, train_and_evaluate


def main() -> None:
    data = make_breast_cancer_split(random_state=42)

    config = TrainConfig(
        input_dim=data.X_train.shape[1],
        n_classes=2,
        n_qubits=4,
        n_q_layers=2,
        hidden_dims=(64, 32),
        epochs=10,
        batch_size=32,
        lr=0.005,
    )

    result = train_and_evaluate(data, config)
    metrics = result["test_metrics"]
    print(f"Breast cancer test loss: {metrics['loss']:.4f}")
    print(f"Breast cancer test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
