from typing import Callable, Union
import numpy as np
import tensorflow as tf


class McqrnnInputDense(tf.Module):
    """
    Mcqrnn Input dense network
    Args:
        input_features (int): the number of nodes in first hidden layer
        activation (Callable): activation function e.g. tf.nn.relu or tf.nn.sigmoid
        name (Union[str, None]): name of Module
    Methods:
        __call__:
            Return dense layer with input activation and features
    """

    def __init__(
        self, input_features: int, activation: Callable, name: Union[str, None] = None
    ):
        super().__init__(name=name)
        self.is_built = False
        self.input_features = input_features
        self.activation = activation

    def __call__(
        self,
        x: np.ndarray,
        tau: np.ndarray,
    ):
        if not self.is_built:
            self.w_x = tf.Variable(
                tf.random.normal([x.shape[-1], self.input_features]), name="w_x"
            )
            self.w_tau = tf.Variable(
                tf.random.normal([tau.shape[-1], self.input_features]), name="w_tau"
            )
            self.b = tf.Variable(tf.zeros([self.input_features]), name="b")
            self.is_built = True

        y = tf.matmul(x, self.w_x) + tf.matmul(tau, tf.exp(self.w_tau)) + self.b
        return self.activation(y)


class McqrnnDense(tf.Module):
    """
    Mcqrnn Input dense network
    Args:
        input_features (int): the number of nodes in first hidden layer
        dense_features (int): the number of nodes in hidden layer
        activation (Callable): activation function e.g. tf.nn.relu or tf.nn.sigmoid
        name (Union[str, None]): name of Module
    Methods:
        __call__:
            Return dense layer with input activation and features
    """

    def __init__(
        self, dense_features: int, activation: Callable, name: Union[str, None] = None
    ):
        super().__init__(name=name)
        self.is_built = False
        self.dense_features = dense_features
        self.activation = activation

    def __call__(self, x: np.ndarray):
        if not self.is_built:
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.dense_features]), name="w"
            )
            self.b = tf.Variable(tf.zeros([self.dense_features]), name="b")
            self.is_built = True

        y = tf.matmul(x, tf.exp(self.w)) + self.b
        return self.activation(y)


class McqrnnOutputDense(tf.Module):
    """
    Mcqrnn Output dense network
    Args:
        name (Union[str, None]): name of Module
    Methods:
        __call__:
            Return dense layer with input activation and features
    """

    def __init__(self, name: Union[str, None] = None):
        super().__init__(name=name)
        self.is_built = False

    def __call__(self, x: np.ndarray):
        if not self.is_built:
            self.w = tf.Variable(tf.random.normal([x.shape[-1], 1]), name="w")
            self.b = tf.Variable(tf.zeros([1]), name="b")
            self.is_built = True

        y = tf.matmul(x, tf.exp(self.w)) + self.b
        return y


class Mcqrnn(tf.Module):
    """
    Mcqrnn Simple structure
    Note that the middle of dense network can be modified with McqrnnDense
    Args:
        input name (Union[str, None]): name of Module
    Methods:
        __call__:
            Return dense layer with input activation and features
    """

    def __init__(
        self,
        input_features: int,
        dense_features: int,
        activation: Callable = tf.nn.relu,
        name: Union[str, None] = None,
    ):
        super().__init__(name=name)
        self.input_features = input_features
        self.dense_features = dense_features
        self.activation = activation
        self.input_dense = McqrnnInputDense(
            input_features=self.input_features,
            activation=self.activation,
        )
        self.dense = McqrnnDense(
            dense_features=self.dense_features, activation=self.activation
        )
        self.output_dense = McqrnnOutputDense()

    def __call__(self, x: np.ndarray, tau: np.ndarray):
        x = self.input_dense(x, tau)
        x = self.dense(x)
        x = self.output_dense(x)
        return x
