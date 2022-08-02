import numpy as np
import pytest
from mcqrnn.generate_example_dataset import _sincx, generate_example


def test_sincx():
    input_array = np.array([0.1, 0.5, 1, 1.5, 2, 1 / np.pi])
    output_array = np.array(
        [0.98363164, 0.63661977, 0.0, -0.21220659, -0.0, 0.84147098]
    )
    assert all(np.round(_sincx(input_array), 8) == output_array)


def test_generate_dataset():
    x, y = generate_example(10)
    assert x.shape[0] == 10
    assert y.shape[0] == 10
