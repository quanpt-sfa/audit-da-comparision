import numpy as np

from audit_da.signal import classify_draws


def test_classification_is_exhaustive():
    pre = np.array([[0.10, 0.10, -0.10, 0.02, 0.10]])
    post = np.array([[0.03, 0.12, 0.05, 0.02, -0.20]])
    states = classify_draws(pre, post, 0.01)
    total = sum(value.astype(int) for value in states.values())
    assert np.all(total == 1)
    assert states["normalization"][0, 0]
    assert states["deterioration"][0, 1]
    assert states["overshoot"][0, 2]
