from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class PositiveConstraint(Constraint):
    """Constraint to keep weights positive."""

    def __init__(self, min_value=1e-7):
        self.min_value = min_value

    def __call__(self, w):
        w = tf.nn.relu(w)
        w = w + self.min_value
        return w

    def get_config(self):
        return {"min_value": self.min_value}


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
        self.w_inputs = self.add_weight(
            name="w_inputs",
            shape=(input_shape[-1], self.out_features),
            initializer="random_normal",
            trainable=True,
        )
        self.w_tau = self.add_weight(
            shape=(1, self.out_features),
            initializer="random_normal",
            constraint=PositiveConstraint(),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.out_features,), initializer="zeros", trainable=True
        )

    def call(self, inputs, tau):
        outputs = tf.matmul(inputs, self.w_inputs) + tf.matmul(tau, self.w_tau) + self.b
        return self.activation(outputs)


class McqrnnDense(tf.keras.layers.Layer):
    """
    Mcqrnn dense network
    Args:
        dense_features (int): the number of nodes in hidden layer
        activation (Callable): activation function e.g. tf.nn.relu or tf.nn.sigmoid
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
        self.w = self.add_weight(
            shape=(input_shape[-1], self.dense_features),
            initializer="random_normal",
            constraint=PositiveConstraint(),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.dense_features,), initializer="zeros", trainable=True
        )

    def call(
        self,
        inputs: np.ndarray,
    ):
        outputs = tf.matmul(inputs, self.w) + self.b
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
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            constraint=PositiveConstraint(),
            trainable=True,
        )
        self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True)

    def call(
        self,
        inputs: np.ndarray,
    ):
        outputs = tf.matmul(inputs, self.w) + self.b
        return outputs
