"""
FAIO RobotWalker policy.

This file is part of the problem statement. The official evaluator computes
actions with exactly the `act` function below. Your submission is a
`submission.npz` file with the 8 arrays listed in SHAPES.
"""

import numpy as np

SHAPES = {
    "W1": (18, 64), "b1": (64,),
    "W2": (64, 64), "b2": (64,),
    "W3": (64, 6),  "b3": (6,),
    "obs_mean": (18,), "obs_std": (18,),
}
MAX_FILE_BYTES = 1024 * 1024  # 1 MiB (a real file is ~25 KB)


def act(obs, w):
    """Deterministic policy: 18 observations -> 6 actions in [-1, 1]."""
    x = (obs - w["obs_mean"]) / w["obs_std"]
    x = np.clip(x, -10.0, 10.0)
    x = np.tanh(x @ w["W1"] + w["b1"])
    x = np.tanh(x @ w["W2"] + w["b2"])
    a = x @ w["W3"] + w["b3"]
    return np.clip(a, -1.0, 1.0)


def load_submission(path):
    """Load and validate a submission.npz. Raises ValueError with a clear message if invalid."""
    import os
    if not os.path.isfile(path):
        raise ValueError(f"file not found: {path}")
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise ValueError(f"file is larger than {MAX_FILE_BYTES // 1024 // 1024} MiB")
    try:
        data = np.load(path)
    except Exception as e:
        raise ValueError(f"not a valid .npz file ({e})")
    arrays = {name: data[name] for name in data.files}
    if set(arrays) != set(SHAPES):
        raise ValueError(f"expected arrays {sorted(SHAPES)}, got {sorted(arrays)}")
    w = {}
    for name, shape in SHAPES.items():
        arr = arrays[name]
        if arr.dtype != np.float32:
            raise ValueError(f"{name}: dtype must be float32, got {arr.dtype}")
        if arr.shape != shape:
            raise ValueError(f"{name}: shape must be {shape}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: contains NaN or Inf")
        w[name] = np.ascontiguousarray(arr, dtype=np.float64)  # fixed layout -> reproducible arithmetic
    if w["obs_std"].min() <= 0:
        raise ValueError("obs_std: all values must be positive")
    return w


def save_submission(path, W1, b1, W2, b2, W3, b3, obs_mean=None, obs_std=None):
    """Save arrays as a valid submission.npz. Weights use (in, out) layout: action = x @ W + b."""
    if obs_mean is None:
        obs_mean = np.zeros(18)
    if obs_std is None:
        obs_std = np.ones(18)
    arrays = dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, obs_mean=obs_mean, obs_std=obs_std)
    arrays = {k: np.asarray(v, dtype=np.float32).reshape(SHAPES[k]) for k, v in arrays.items()}
    np.savez_compressed(path, **arrays)
    load_submission(path)  # fail early if something is wrong
