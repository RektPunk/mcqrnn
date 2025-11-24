import numpy as np


def _data_transform(
    x: np.ndarray,
    taus: np.ndarray,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """
    Transform x, y, taus into the trainable form
    Args:
        x (np.ndarray): input
        y (np.ndarray): output
        taus (np.ndarray): quantiles
    Return
        tuple[np.ndarray]: transformed x, y
    """
    _len_taus = len(taus)
    _len_x = len(x)
    _len_data = _len_x * _len_taus

    x_trans = np.repeat(x, _len_taus, axis=0).astype("float32")
    taus_trans = np.tile(taus, _len_x).reshape((_len_data, 1)).astype("float32")
    if y is not None:
        y_trans = np.repeat(y, _len_taus, axis=0).astype("float32")
        y_trans = y_trans.reshape((_len_data, 1))
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
            Return tuple[np.ndarray, ...]:
        transform(input_taus: np.ndarray):
            Return transformed x with given input_taus
    """

    def __init__(
        self,
        x: np.ndarray,
        taus: np.ndarray,
        y: np.ndarray | None = None,
    ):
        self.x = x
        self.y = y
        self.taus = taus
        self.x_trans, self.y_trans, self.tau_trans = _data_transform(
            x=self.x, y=self.y, taus=self.taus
        )

    def __call__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x_trans, self.y_trans, self.tau_trans

    def transform(self, x: np.ndarray, input_taus: np.ndarray) -> np.ndarray:
        input_taus = input_taus.astype("float32")
        return _data_transform(
            x=x,
            taus=input_taus,
        )
