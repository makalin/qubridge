# QuBridge: Hybrid Classical-Quantum Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-QML-yellow.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**QuBridge** is an experimental machine learning pipeline that fuses traditional deep learning with quantum computing. By leveraging PyTorch for classical feature extraction and PennyLane for Parameterized Quantum Circuits (PQCs), this project demonstrates how to build and train hybrid neural networks for advanced pattern recognition.

---

## 📖 The Concept

As classical deep learning models approach their computational limits, Quantum Machine Learning (QML) offers a new frontier. QuBridge does not rely entirely on a quantum computer. Instead, it uses a highly efficient hybrid architecture:

1.  **Classical Feature Extraction:** A traditional PyTorch neural network ingests complex data and compresses it into a refined feature vector.
2.  **Quantum Classification:** These extracted features are embedded into a quantum state using a Parameterized Quantum Circuit (PQC), which acts as the final highly-dimensional classifier.



## 🚀 Key Features

* **Seamless Integration:** Treats quantum circuits as standard PyTorch layers.
* **Quantum Data Embedding:** Implements efficient angle and amplitude embedding to translate classical bits into quantum qubits.
* **Parameter-Shift Training:** Utilizes the parameter-shift rule to calculate precise gradients for the quantum circuit without collapsing the quantum state during backpropagation.



## 🛠️ Tech Stack

* **Python:** Core orchestration and logic.
* **PyTorch:** Classical neural network layers, loss functions, and optimization.
* **PennyLane:** Quantum circuit construction, data embedding, and QNode integration.

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/makalin/qubridge.git](https://github.com/makalin/qubridge.git)
cd qubridge
pip install -r requirements.txt

```

## 💻 Quick Start

Here is a basic example of how to initialize and run the hybrid model:

```python
import torch
import pennylane as qml
from models.hybrid_net import QuBridgeModel

# Initialize the hybrid model (e.g., 4 classical features out, 4 qubits)
model = QuBridgeModel(n_qubits=4)

# Dummy input data (e.g., a batch of 10 images flattened)
input_data = torch.randn(10, 256) 

# Forward pass through classical PyTorch layers into the Quantum Circuit
predictions = model(input_data)

print(predictions)

```

## 🧠 Architecture Deep Dive

For a detailed breakdown of the math, the data embedding techniques, and how the gradients are calculated across the classical-quantum boundary, please read our technical essay: [The Quantum Leap in AI: Building Hybrid Classical-Quantum Neural Networks](https://www.google.com/search?q=link-to-your-essay-here).

## 👨‍💻 Author

**Mehmet T. AKALIN**

* GitHub: [@makalin](https://github.com/makalin)
* Company: [Digital Vision](https://dv.com.tr)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
