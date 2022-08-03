from typing import Union
import tensorflow as tf
import numpy as np


class TiltedAbsoluteLoss(tf.keras.losses.Loss):
    """
    Tilted absolute loss function or check loss
    Args:
        y_true (Union[np.ndarray, tf.Tesnsor]): train target value
        y_pred (Union[np.ndarray, tf.Tesnsor]): pred value
        tau (Union[np.ndarray, tf.Tesnsor]): quantiles
    Return:
        tf.Tensor: tilted absolute loss
    """

    def __init__(self, tau: Union[np.ndarray, tf.Tensor], **kwargs):
        super(TiltedAbsoluteLoss, self).__init__(**kwargs)
        self._one = tf.cast(1, dtype=tau.dtype)
        self._tau = tf.cast(tau, dtype=tau.dtype)

    def call(
        self,
        y_true: Union[np.ndarray, tf.Tensor],
        y_pred: Union[np.ndarray, tf.Tensor],
    ) -> tf.Tensor:
        error = y_true - y_pred
        _loss = tf.math.maximum(self._tau * error, (self._tau - self._one) * error)
        return tf.reduce_mean(_loss)
