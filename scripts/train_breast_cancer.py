#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from qubridge.data.real import make_breast_cancer_split
from qubridge.train import TrainConfig, load_train_config, save_checkpoint, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QuBridge on breast cancer dataset")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--n-q-layers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/qubridge_breast_cancer.pt"))
    parser.add_argument("--history-out", type=Path, default=Path("outputs/history_breast_cancer.json"))
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def configure_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("qubridge.train.breast")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.log_level)

    torch.manual_seed(args.seed)

    data = make_breast_cancer_split(random_state=args.seed)

    config = load_train_config(args.config) if args.config else TrainConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.lr = args.learning_rate
    if args.n_qubits is not None:
        config.n_qubits = args.n_qubits
    if args.n_q_layers is not None:
        config.n_q_layers = args.n_q_layers
    if args.device is not None:
        config.device = args.device

    config.input_dim = data.X_train.shape[1]
    config.n_classes = 2

    logger.info("training_config=%s", config.to_dict())
    result = train_and_evaluate(data, config, logger=logger)

    save_checkpoint(result["model"], config, args.out)
    args.history_out.parent.mkdir(parents=True, exist_ok=True)
    args.history_out.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")

    metrics = result["test_metrics"]
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
