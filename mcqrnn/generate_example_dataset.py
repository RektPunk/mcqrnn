import numpy as np


def _sincx(x: np.ndarray) -> np.ndarray:
    """
    Return sincx
    Args:
        x (np.ndarray): input x
    Returns:
        (np.ndarray): sinc(x)
    """
    _x_pi = np.pi * x
    return np.sin(_x_pi) / _x_pi


def generate_example(
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset
    Args:
        x (int): number of samples
    Returns:
        Tuple[float32]: x, y
    """
    samples = np.random.uniform(low=-1, high=1, size=n_samples)
    reshaped_samples = np.reshape(samples, newshape=(n_samples, 1))
    sincx_samples = _sincx(samples)
    eps = np.random.normal(loc=0, scale=0.1 * np.exp(1 - samples))
    target = sincx_samples + eps
    return reshaped_samples.astype("float32"), target.astype("float32")
