from typing import Callable
import numpy as np
import tensorflow as tf
from mcqrnn.layers import (
    McqrnnInputDense,
    McqrnnDense,
    McqrnnOutputDense,
)


class Mcqrnn(tf.keras.Model):
    """
    Mcqrnn simple structure
    Note that the middle of dense network can be modified with McqrnnDense
    Args:
        out_features (int): the number of nodes in first hidden layer
        dense_features (int): the number of nodes in hidden layer
        activation (Callable): activation function e.g. tf.nn.relu or tf.nn.sigmoid
    Methods:
        call:
            Return dense layer with input activation and features
    """

    def __init__(
        self,
        out_features: int,
        dense_features: int,
        activation: Callable = tf.nn.relu,
        **kwargs,
    ):
        super(Mcqrnn, self).__init__(**kwargs)
        self.input_dense = McqrnnInputDense(
            out_features=out_features,
            activation=activation,
        )
        self.dense = McqrnnDense(
            dense_features=dense_features,
            activation=activation,
        )
        self.output_dense = McqrnnOutputDense()

    def call(
        self,
        inputs: np.ndarray,
        tau: np.ndarray,
    ):
        x = self.input_dense(inputs, tau)
        x = self.dense(x)
        outputs = self.output_dense(x)
        return outputs
