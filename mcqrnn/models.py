import tensorflow as tf


class McqrnnInputDense(tf.Module):
    def __init__(self, input_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.input_features = input_features

    def __call__(self, x, tau):
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
        return tf.nn.relu(y)


class McqrnnDense(tf.Module):
    def __init__(self, dense_features, activation, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.dense_features = dense_features
        self.activation = activation

    def __call__(self, x):
        # Create variables on first call.
        if not self.is_built:
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.dense_features]), name="w"
            )
            self.b = tf.Variable(tf.zeros([self.dense_features]), name="b")
            self.is_built = True

        y = tf.matmul(x, tf.exp(self.w)) + self.b
        return self.activation(y)


class McqrnnOutputDense(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.is_built = False

    def __call__(self, x):
        # Create variables on first call.
        if not self.is_built:
            self.w = tf.Variable(tf.random.normal([x.shape[-1], 1]), name="w")
            self.b = tf.Variable(tf.zeros([1]), name="b")
            self.is_built = True

        y = tf.matmul(x, tf.exp(self.w)) + self.b
        return y


class Mcqrnn(tf.Module):
    def __init__(
        self, input_features, dense_features, activation=tf.nn.relu, name=None
    ):
        super().__init__(name=name)
        self.input_features = input_features
        self.dense_features = dense_features
        self.activation = activation
        self.input_dense = McqrnnInputDense(input_features=input_features)
        self.dense = McqrnnDense(dense_features=dense_features, activation=activation)
        self.output_dense = McqrnnOutputDense()

    def __call__(self, x, tau):
        x = self.input_dense(x, tau)
        x = self.dense(x)
        x = self.output_dense(x)
        return x
