from typing import Union
import tensorflow as tf
import numpy as np


def tilted_absolute_loss(
    y_true: Union[np.ndarray, tf.Tensor],
    y_pred: Union[np.ndarray, tf.Tensor],
    tau: Union[np.ndarray, tf.Tensor],
) -> tf.Tensor:
    """
    Tilted absolute loss function or check loss
    Args:
        y_true (Union[np.ndarray, tf.Tesnsor]): train target value
        y_pred (Union[np.ndarray, tf.Tesnsor]): pred value
        tau (Union[np.ndarray, tf.Tesnsor]): quantiles
    Return:
        tf.Tensor: tilted absolute loss
    """
    error = y_true - y_pred
    one_tf = tf.cast(1, dtype=tau.dtype)
    tau_tf = tf.cast(tau, dtype=y_pred.dtype)
    loss_tf = tf.math.maximum(tau_tf * error, (tau_tf - one_tf) * error)
    return tf.reduce_mean(loss_tf)
