import numpy as np

from mcqrnn.data_transform import DataTransformer
from mcqrnn.generate_example_dataset import generate_example
from mcqrnn.models import Mcqrnn

test_taus = np.array([i / 10 for i in range(1, 10)])


def test_Mcqrnn():
    x_train, y_train = generate_example(10)
    data_transformer = DataTransformer(
        x=x_train,
        y=y_train,
        taus=test_taus,
    )
    mcqrnn_module = Mcqrnn(
        out_features=3,
        dense_features=3,
    )

    tests = []
    for test_tau in test_taus:
        x_tmp, tau_tmp = data_transformer.transform(x_train, np.array([test_tau]))
        tests.append(mcqrnn_module(x_tmp, tau_tmp))

    assert np.all(np.diff(np.concatenate(tests, axis=1)) >= 0)
