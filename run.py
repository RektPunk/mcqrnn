import tensorflow as tf
import numpy as np
from mcqrnn import generate_example, Mcqrnn, DataTransformer


x_train, y_train = generate_example(10)
taus = np.array([0.1, 0.5, 0.9])

data_transformer = DataTransformer(
    x=x_train,
    taus=taus,
    y=y_train,
)

x_train_transform, y_train_transform, taus_transform = data_transformer()

mcqrnn_module = Mcqrnn(
    input_features=3,
    dense_features=3,
    activation=tf.nn.relu,
)
