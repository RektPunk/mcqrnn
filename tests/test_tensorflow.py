import numpy as np
import tensorflow as tf

from mcqrnn.tensorflow import MCQRNNRegressor


def test_Mcqrnn():
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    ### train data
    n = 1000
    input_dim = 1
    x_data = np.random.uniform(-1, 1, (n, input_dim))

    # np.sinc(x) is defined as sin(pi*x) / (pi*x)
    sincx = np.sinc(x_data)
    Z = sincx.reshape(n, input_dim)
    # heteroscedastic noise
    ep = np.random.normal(0, 0.1 * np.exp(1 - x_data)).reshape(n, 1)
    y_data = Z + ep

    tau_vec = np.arange(0.1, 1.0, 0.1)

    mcqrnn = MCQRNNRegressor(tau=tau_vec, out_features=3, dense_features=3, epochs=100)

    mcqrnn.fit(x_data, y_data)
    predictions = mcqrnn.predict(x_data)
    assert np.all(np.diff(predictions, axis=1) >= 0)
