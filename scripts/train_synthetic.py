#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch

from qubridge.data.synthetic import make_binary_moons
from qubridge.train import TrainConfig, load_train_config, save_checkpoint, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QuBridge on synthetic moons data")

    parser.add_argument("--config", type=Path, default=None, help="Path to JSON or YAML config file")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--n-q-layers", type=int, default=None)
    parser.add_argument("--hidden-dims", type=str, default=None, help="Comma-separated dims, e.g. 32,16")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/qubridge_moons.pt"))

    parser.add_argument("--history-out", type=Path, default=Path("outputs/history.json"))
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def parse_hidden_dims(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = [segment.strip() for segment in raw.split(",") if segment.strip()]
    if not values:
        return None
    return tuple(int(v) for v in values)


def configure_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("qubridge.train")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    config = load_train_config(args.config) if args.config else TrainConfig()

    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["lr"] = args.learning_rate
    if args.n_qubits is not None:
        overrides["n_qubits"] = args.n_qubits
    if args.n_q_layers is not None:
        overrides["n_q_layers"] = args.n_q_layers
    if args.device is not None:
        overrides["device"] = args.device

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    if hidden_dims is not None:
        overrides["hidden_dims"] = hidden_dims

    merged = {**asdict(config), **overrides}
    return TrainConfig.from_dict(merged)


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.log_level)

    torch.manual_seed(args.seed)

    data = make_binary_moons(
        n_samples=args.samples,
        noise=args.noise,
        random_state=args.seed,
    )

    config = resolve_config(args)
    config.input_dim = data.X_train.shape[1]

    writer = None
    if args.tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(args.tensorboard_dir))

    logger.info("training_config=%s", config.to_dict())
    result = train_and_evaluate(data, config, logger=logger, writer=writer)

    if writer is not None:
        writer.close()

    save_checkpoint(result["model"], config, args.out)

    args.history_out.parent.mkdir(parents=True, exist_ok=True)
    args.history_out.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")

    metrics = result["test_metrics"]
    logger.info("checkpoint_saved=%s", args.out)
    logger.info("history_saved=%s", args.history_out)
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Checkpoint saved to: {args.out}")
    print(f"History saved to: {args.history_out}")


if __name__ == "__main__":
    main()
