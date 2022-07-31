from typing import Union, Tuple
import numpy as np


def _mcqrnn_transform(
    x: np.ndarray,
    taus: np.ndarray,
    y: Union[np.ndarray, None] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Transform x, y, taus into the trainable form
    Args:
        x (np.ndarray): input
        y (np.ndarray): output
        taus (np.ndarray): quantiles
    Return
        Tuple[np.ndarray]: transformed x, y
    """
    _len_taus = len(taus)
    _len_x = len(x)
    x_trans = np.repeat(x, _len_taus, axis=0).astype("float32")
    taus_trans = (
        np.tile(taus, _len_x).reshape((_len_x * _len_taus, 1)).astype("float32")
    )
    if y is not None:
        y_trans = np.repeat(y, _len_taus, axis=0).astype("float32")
        return x_trans, y_trans, taus_trans
    else:
        return x_trans, taus_trans


class DataTransformer:
    """
    A class to transform data into trainable form.
    Args:
        x (np.ndarray): input
        taus (np.ndarray): quantiles
        y (Union[np.ndarray, None]): output

    Methods:
        __call__:
            Return Tuple[np.ndarray, ...]:
        transform(input_taus: np.ndarray):
            Return transformed x with given input_taus
    """

    def __init__(
        self,
        x: np.ndarray,
        taus: np.ndarray,
        y: Union[np.ndarray, None] = None,
    ):
        self.x = x
        self.y = y
        self.taus = taus
        self.x_trans, self.y_trans, self.tau_trans = _mcqrnn_transform(
            x=self.x, y=self.y, taus=self.taus
        )

    def __call__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x_trans, self.y_trans, self.tau_trans

    def transform(self, input_taus: np.ndarray) -> np.ndarray:
        input_taus = input_taus.astype("float32")
        return _mcqrnn_transform(
            x=self.x,
            taus=input_taus,
        )
