import numpy as np
import tensorflow as tf


@tf.function
def train_step(
    model: tf.keras.Model,
    inputs: np.ndarray,
    output: np.ndarray,
    tau: np.ndarray,
    loss_func: tf.keras.losses.Loss,
    optimizer: tf.keras.optimizers,
):
    with tf.GradientTape(persistent=True) as tape:
        predicted = model(inputs, tau)
        loss = loss_func(output, predicted)

    grad = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))
    return loss
