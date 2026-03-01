#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qubridge.train import load_checkpoint


def parse_features(raw: str) -> torch.Tensor:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return torch.tensor([values], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from a saved QuBridge checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Comma-separated feature vector, e.g. '0.1,1.2,-0.4,...'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, config = load_checkpoint(args.checkpoint)
    model.eval()

    x = parse_features(args.features)
    if x.shape[1] != config.input_dim:
        raise ValueError(f"Expected {config.input_dim} features, got {x.shape[1]}")

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())

    print(f"Predicted class: {pred}")
    print(f"Probabilities: {probs.squeeze(0).tolist()}")


if __name__ == "__main__":
    main()
