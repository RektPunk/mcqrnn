from typing import Callable, Union
import numpy as np
import tensorflow as tf


class McqrnnInputDense(tf.keras.layers.Layer):
    """
    Mcqrnn Input dense network
    Args:
        out_features (int): the number of nodes in first hidden layer
        activation (Callable): activation function e.g. tf.nn.relu or tf.nn.sigmoid
    Methods:
        build:
            Set weight shape for first call
        call:
            Return dense layer with input activation and features
    """

    def __init__(
        self,
        out_features: int,
        activation: Callable,
        **kwargs,
    ):
        super(McqrnnInputDense, self).__init__(**kwargs)
        self.out_features = out_features
        self.activation = activation

    def build(
        self,
        input_shape,
    ):
        self.w_inputs = tf.Variable(
            tf.random.normal([input_shape[-1], self.out_features]), name="w_inputs"
        )
        self.w_tau = tf.Variable(tf.random.normal([1, self.out_features]), name="w_tau")
        self.b = tf.Variable(tf.zeros([self.out_features]), name="b")

    def call(self, inputs, tau):
        outputs = (
            tf.matmul(inputs, self.w_inputs)
            + tf.matmul(tau, tf.exp(self.w_tau))
            + self.b
        )
        return self.activation(outputs)


class McqrnnDense(tf.keras.layers.Layer):
    """
    Mcqrnn dense network
    Args:
        dense_features (int): the number of nodes in hidden layer
        activation (Callable): activation function e.g. tf.nn.relu or tf.nn.sigmoid
        name (Union[str, None]): name of Module
    Methods:
        build:
            Set weight shape for first call
        call:
            Return dense layer with activation
    """

    def __init__(
        self,
        dense_features: int,
        activation: Callable,
        **kwargs,
    ):
        super(McqrnnDense, self).__init__(**kwargs)
        self.dense_features = dense_features
        self.activation = activation

    def build(
        self,
        input_shape,
    ):
        self.w = tf.Variable(
            tf.random.normal([input_shape[-1], self.dense_features]), name="w"
        )
        self.b = tf.Variable(tf.zeros([self.dense_features]), name="b")

    def call(
        self,
        inputs: np.ndarray,
    ):
        outputs = tf.matmul(inputs, tf.exp(self.w)) + self.b
        return self.activation(outputs)


class McqrnnOutputDense(tf.keras.layers.Layer):
    """
    Mcqrnn Output dense network
    Methods:
        build:
            Set weight shape for first call
        call:
            Return dense layer with activation
    """

    def __init__(
        self,
        **kwargs,
    ):
        super(McqrnnOutputDense, self).__init__(**kwargs)

    def build(
        self,
        input_shape,
    ):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], 1]), name="w")
        self.b = tf.Variable(tf.zeros([1]), name="b")

    def call(
        self,
        inputs: np.ndarray,
    ):
        outputs = tf.matmul(inputs, tf.exp(self.w)) + self.b
        return outputs
