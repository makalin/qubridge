from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qubridge.data.synthetic import SplitData


def make_breast_cancer_split(
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """Create train/val/test tensors for sklearn breast cancer dataset."""
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_train,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def as_tensor(array: np.ndarray, is_target: bool = False) -> torch.Tensor:
        if is_target:
            return torch.tensor(array, dtype=torch.long)
        return torch.tensor(array, dtype=torch.float32)

    return SplitData(
        X_train=as_tensor(X_train),
        y_train=as_tensor(y_train, is_target=True),
        X_val=as_tensor(X_val),
        y_val=as_tensor(y_val, is_target=True),
        X_test=as_tensor(X_test),
        y_test=as_tensor(y_test, is_target=True),
    )
