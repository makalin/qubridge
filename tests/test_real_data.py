from __future__ import annotations

from qubridge.data.real import make_breast_cancer_split


def test_breast_cancer_split_shapes() -> None:
    split = make_breast_cancer_split(random_state=1)

    assert split.X_train.ndim == 2
    assert split.X_val.ndim == 2
    assert split.X_test.ndim == 2
    assert split.X_train.shape[1] == 30
    assert split.X_val.shape[1] == 30
    assert split.X_test.shape[1] == 30
    assert split.y_train.ndim == 1
