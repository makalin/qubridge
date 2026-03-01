from __future__ import annotations

import torch
from torch import nn

from qubridge.data.synthetic import make_binary_moons
from qubridge.train import TrainConfig, build_model, load_checkpoint, save_checkpoint, train_and_evaluate


def test_single_batch_optimization_reduces_loss() -> None:
    torch.manual_seed(7)
    data = make_binary_moons(n_samples=200, noise=0.15, random_state=7)

    config = TrainConfig(
        input_dim=2,
        n_classes=2,
        n_qubits=2,
        n_q_layers=1,
        hidden_dims=(8,),
        epochs=1,
        batch_size=16,
        lr=0.05,
        device="cpu",
    )

    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    x = data.X_train[:16]
    y = data.y_train[:16]

    with torch.no_grad():
        baseline = criterion(model(x), y).item()

    for _ in range(10):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = criterion(model(x), y).item()

    assert final_loss < baseline


def test_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(11)
    config = TrainConfig(
        input_dim=2,
        n_classes=2,
        n_qubits=2,
        n_q_layers=1,
        hidden_dims=(8,),
        epochs=1,
        batch_size=8,
        lr=0.01,
        device="cpu",
    )

    model = build_model(config)
    x = torch.randn(5, 2)
    before = model(x)

    ckpt = tmp_path / "model.pt"
    save_checkpoint(model, config, ckpt)

    loaded_model, loaded_config = load_checkpoint(ckpt)
    after = loaded_model(x)

    assert loaded_config == config
    assert torch.allclose(before, after)


def test_training_is_reproducible_with_seed() -> None:
    data = make_binary_moons(n_samples=300, noise=0.2, random_state=23)
    config = TrainConfig(
        input_dim=2,
        n_classes=2,
        n_qubits=2,
        n_q_layers=1,
        hidden_dims=(8,),
        epochs=3,
        batch_size=16,
        lr=0.01,
        device="cpu",
    )

    torch.manual_seed(23)
    run_a = train_and_evaluate(data, config)

    torch.manual_seed(23)
    run_b = train_and_evaluate(data, config)

    assert run_a["history"]["val_loss"] == run_b["history"]["val_loss"]
    assert run_a["history"]["val_acc"] == run_b["history"]["val_acc"]


def test_quantum_path_has_gradients() -> None:
    torch.manual_seed(13)
    config = TrainConfig(
        input_dim=2,
        n_classes=2,
        n_qubits=2,
        n_q_layers=1,
        hidden_dims=(8,),
        epochs=1,
        batch_size=4,
        lr=0.01,
        device="cpu",
    )

    model = build_model(config)
    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(g is not None for g in grads)
