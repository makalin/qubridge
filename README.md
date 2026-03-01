# QuBridge: Hybrid Classical-Quantum Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-QML-yellow.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

QuBridge is a reference implementation of a hybrid classical-quantum learning pipeline.
It combines a classical PyTorch feature extractor with a PennyLane-powered parameterized quantum circuit (PQC)
for end-to-end differentiable classification.

## Features

- Hybrid PyTorch + PennyLane model (`QuBridgeModel`).
- JSON/YAML config-driven training (`TrainConfig`).
- Structured epoch logging and optional TensorBoard metrics.
- Checkpoint save/load utilities.
- Synthetic and real dataset pipelines (sklearn breast cancer).
- CI, lint/type tooling, contribution templates, and security policy.

## Project Structure

```text
qubridge/
├── qubridge/
│   ├── data/
│   │   ├── real.py
│   │   └── synthetic.py
│   ├── models/
│   │   ├── classical.py
│   │   ├── hybrid_net.py
│   │   └── quantum_layer.py
│   └── train.py
├── scripts/
│   ├── predict.py
│   ├── train_breast_cancer.py
│   └── train_synthetic.py
├── examples/
│   ├── basic_forward.py
│   ├── train_binary_classification.py
│   └── train_breast_cancer.py
├── tests/
├── configs/
│   ├── train_breast_cancer.yaml
│   ├── train_synthetic.json
│   └── train_synthetic.yaml
├── docs/
│   └── PERFORMANCE.md
├── .github/
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Installation

```bash
git clone https://github.com/makalin/qubridge.git
cd qubridge
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# dev setup: pip install -r requirements-dev.txt
```

## Quick Start

### Basic forward pass

```bash
python examples/basic_forward.py
```

### Train on synthetic moons

```bash
python scripts/train_synthetic.py --config configs/train_synthetic.yaml
```

### Train on real dataset (breast cancer)

```bash
python scripts/train_breast_cancer.py --config configs/train_breast_cancer.yaml
```

### Enable TensorBoard

```bash
python scripts/train_synthetic.py --config configs/train_synthetic.json --tensorboard-dir logs/tensorboard
tensorboard --logdir logs/tensorboard
```

### Inference from a saved checkpoint

```bash
python scripts/predict.py --checkpoint checkpoints/qubridge_breast_cancer.pt --features "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0"
```

## Quality and CI

```bash
ruff check .
ruff format .
mypy qubridge
pytest
```

GitHub Actions runs lint, mypy, and tests on Python 3.10 and 3.11.

## Documentation and Policies

- Performance guidance: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- Contribution process: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security reporting: [SECURITY.md](SECURITY.md)

## Architecture Deep Dive

[The Quantum Leap in AI: Building Hybrid Classical-Quantum Neural Networks](https://medium.com/@makalin/the-quantum-leap-in-ai-building-hybrid-classical-quantum-neural-networks-324dacce5c83)

## License

MIT License. See [LICENSE](LICENSE).
