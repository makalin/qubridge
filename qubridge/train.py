from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from qubridge.data.synthetic import SplitData
from qubridge.models.hybrid_net import QuBridgeModel


@dataclass
class TrainConfig:
    input_dim: int = 2
    n_classes: int = 2
    n_qubits: int = 4
    n_q_layers: int = 2
    hidden_dims: tuple[int, ...] = (32, 16)
    dropout: float = 0.0
    lr: float = 0.01
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    device: str = "cpu"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in payload.items() if k in valid_keys}

        if "hidden_dims" in filtered:
            hidden_dims = filtered["hidden_dims"]
            if isinstance(hidden_dims, list):
                filtered["hidden_dims"] = tuple(int(v) for v in hidden_dims)
            elif isinstance(hidden_dims, tuple):
                filtered["hidden_dims"] = tuple(int(v) for v in hidden_dims)

        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_train_config(path: str | Path) -> TrainConfig:
    source = Path(path)
    suffix = source.suffix.lower()
    raw = source.read_text(encoding="utf-8")

    if suffix == ".json":
        payload = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML config files") from exc
        payload = yaml.safe_load(raw)
    else:
        raise ValueError(f"Unsupported config file extension: {suffix}")

    if not isinstance(payload, dict):
        raise ValueError("Train config file must decode to a JSON/YAML object")

    return TrainConfig.from_dict(payload)


def _make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, data_loader: DataLoader, device: str) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y).sum().item()
            total += x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def fit(
    model: nn.Module,
    data: SplitData,
    config: TrainConfig,
    logger: logging.Logger | None = None,
    writer: Any | None = None,
) -> dict[str, list[float]]:
    train_loader = _make_loader(data.X_train, data.y_train, config.batch_size, shuffle=True)
    val_loader = _make_loader(data.X_val, data.y_val, config.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    device = config.device
    model.to(device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in tqdm(range(config.epochs), desc="Training", leave=False):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        step = epoch + 1
        if logger is not None:
            logger.info(
                "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                step,
                train_loss,
                train_acc,
                val_metrics["loss"],
                val_metrics["accuracy"],
            )

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, step)
            writer.add_scalar("train/accuracy", train_acc, step)
            writer.add_scalar("val/loss", val_metrics["loss"], step)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], step)

    return history


def save_checkpoint(model: nn.Module, config: TrainConfig, path: str | Path) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config.to_dict(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[QuBridgeModel, TrainConfig]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    config = TrainConfig.from_dict(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    return model, config


def build_model(config: TrainConfig) -> QuBridgeModel:
    return QuBridgeModel(
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        n_qubits=config.n_qubits,
        n_q_layers=config.n_q_layers,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )


def test(model: nn.Module, data: SplitData, config: TrainConfig) -> dict[str, float]:
    test_loader = _make_loader(data.X_test, data.y_test, config.batch_size, shuffle=False)
    model.to(config.device)
    return evaluate(model, test_loader, config.device)


def train_and_evaluate(
    data: SplitData,
    config: TrainConfig,
    logger: logging.Logger | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    model = build_model(config)
    history = fit(model, data, config, logger=logger, writer=writer)
    test_metrics = test(model, data, config)

    if logger is not None:
        logger.info("test_loss=%.4f test_acc=%.4f", test_metrics["loss"], test_metrics["accuracy"])

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
    }
