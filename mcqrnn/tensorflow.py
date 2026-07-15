import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from numpy.typing import NDArray


class PositiveConstraint(keras.constraints.Constraint):
    def __init__(self, min_value: float = 1e-7):
        self.min_value = min_value

    def __call__(self, w):
        return tf.maximum(w, self.min_value)


class MCQRNNModel(keras.Model):
    def __init__(self, out_features: int, dense_features: int, **kwargs):
        super().__init__(**kwargs)
        self.x_feature_extractor = keras.Sequential(
            [
                Dense(dense_features, activation="sigmoid"),
                Dense(out_features, activation="sigmoid"),
            ]
        )

        pos_initializer = keras.initializers.RandomUniform(minval=0.01, maxval=0.1)
        self.tau_embedding = Dense(
            out_features,
            use_bias=True,
            kernel_constraint=PositiveConstraint(),
            kernel_initializer=pos_initializer,  # type: ignore
        )
        self.monotone_hidden = Dense(
            dense_features,
            activation="sigmoid",
            kernel_constraint=PositiveConstraint(),
            kernel_initializer=pos_initializer,  # type: ignore
        )
        self.output_dense = Dense(
            1,
            activation="linear",
            kernel_constraint=PositiveConstraint(),
            kernel_initializer=pos_initializer,  # type: ignore
        )

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, tau = inputs
        h_x = self.x_feature_extractor(x)
        h_tau = self.tau_embedding(tau)
        merged = h_x + h_tau
        x_mono = self.monotone_hidden(merged)
        outputs = self.output_dense(x_mono)

        return outputs


class MCQRNNRegressor:
    def __init__(
        self,
        tau: NDArray[np.float64],
        out_features: int = 64,
        dense_features: int = 32,
        lr: float = 0.01,
        epochs: int = 5000,
    ):
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.r = len(tau)
        self.out_features = out_features
        self.dense_features = dense_features
        self.lr = lr
        self.epochs = epochs
        self.model: MCQRNNModel | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "MCQRNNRegressor":
        n_samples = X.shape[0]
        self.model = MCQRNNModel(
            out_features=self.out_features, dense_features=self.dense_features
        )
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        X_tensor = tf.cast(X, tf.float32)
        y_tensor = tf.cast(y, tf.float32)
        if len(y_tensor.shape) == 1:
            y_tensor = tf.expand_dims(y_tensor, axis=-1)

        X_tiled = tf.repeat(X_tensor, repeats=self.r, axis=0)
        y_tiled = tf.repeat(y_tensor, repeats=self.r, axis=0)
        tau_tiled = tf.tile(tf.expand_dims(self.tau, axis=-1), [n_samples, 1])

        assert self.model is not None
        model_local = self.model

        @tf.function
        def train_step(X_batch, y_batch, tau_batch):
            with tf.GradientTape() as tape:
                y_pred = model_local((X_batch, tau_batch), training=True)
                diff_y = y_batch - y_pred
                loss = tf.maximum(
                    tau_batch * diff_y,
                    (tau_batch - 1.0) * diff_y,
                )
                loss = tf.reduce_mean(loss)
                total_loss = loss + sum(model_local.losses)

            gradients = tape.gradient(total_loss, model_local.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_local.trainable_variables))
            return total_loss

        for epoch in range(self.epochs):
            loss_val = train_step(X_tiled, y_tiled, tau_tiled)

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] - Loss: {loss_val.numpy():.4f}"
                )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float32]:
        if self.model is None:
            raise ValueError(
                "The model has not been trained yet. Please call .fit() first."
            )

        n_samples = X.shape[0]
        X_tensor = tf.cast(X, tf.float32)

        X_tiled = tf.repeat(X_tensor, repeats=self.r, axis=0)
        tau_tiled = tf.tile(tf.expand_dims(self.tau, axis=-1), [n_samples, 1])

        predictions_flat = self.model((X_tiled, tau_tiled), training=False)
        predictions = tf.reshape(predictions_flat, [n_samples, self.r])
        return predictions.numpy()
