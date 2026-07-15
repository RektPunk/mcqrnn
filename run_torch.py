import matplotlib.pyplot as plt
import numpy as np
import torch

from mcqrnn.torch import MCQRNNRegressor

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

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

    ### test data
    x_test_data = np.random.uniform(-1, 1, (n, input_dim))
    sincx_test = np.sinc(x_test_data)
    y_test = sincx_test.reshape(n, input_dim) + np.random.normal(
        0, 0.1 * np.exp(1 - x_test_data)
    ).reshape(n, 1)

    ###
    tau_vec = np.arange(0.1, 1.0, 0.1)

    mcqrnn_regressor = MCQRNNRegressor(
        tau=tau_vec,
        out_features=20,
        dense_features=20,
        lr=0.01,
        epochs=5000,
    )

    mcqrnn_regressor.fit(x_data, y_data)

    y_pred = mcqrnn_regressor.predict(x_test_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_test_data, y_test, color="gray", alpha=0.3, label="Test Data")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(tau_vec)))
    for i in range(y_pred.shape[1]):
        plt.scatter(x_test_data, y_pred[:, i], color=colors[i], s=5)

    plt.title("MCQRNN (Torch)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
