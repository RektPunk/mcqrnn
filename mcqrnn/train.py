import numpy as np
import tensorflow as tf
from mcqrnn.models import Mcqrnn
from mcqrnn.loss import TiltedAbsoluteLoss


@tf.function
def train_step(
    model: Mcqrnn,
    inputs: np.ndarray,
    output: np.ndarray,
    tau: np.ndarray,
    loss_func: TiltedAbsoluteLoss,
    optimizer: tf.keras.optimizers,
):
    with tf.GradientTape(persistent=True) as tape:
        predicted = model(inputs, tau)
        loss = loss_func(output, predicted)

    grad = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))
    return loss
