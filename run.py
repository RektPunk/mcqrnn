import tensorflow as tf
import numpy as np
from mcqrnn import generate_example, train_step
from mcqrnn import (
    TiltedAbsoluteLoss,
    Mcqrnn,
    DataTransformer,
)

# Examples setting
EPOCHS = 1000
LEARNING_RATE = 0.005
N_SAMPLES = 100
TAUS = [0.1, 0.5, 0.9]
OUT_FEATURES = 10
DENSE_FEATURES = 10

x_train, y_train = generate_example(N_SAMPLES)
taus = np.array(TAUS)


data_transformer = DataTransformer(x=x_train, taus=taus, y=y_train,)
x_train_transform, y_train_transform, taus_transform = data_transformer()

mcqrnn_module = Mcqrnn(
    out_features=OUT_FEATURES, dense_features=DENSE_FEATURES, activation=tf.nn.sigmoid,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
tilted_absolute_loss = TiltedAbsoluteLoss(tau=taus_transform)
for epoch in range(EPOCHS):
    train_loss = train_step(
        model=mcqrnn_module,
        inputs=x_train_transform,
        output=y_train_transform,
        tau=taus_transform,
        loss_func=tilted_absolute_loss,
        optimizer=optimizer,
    )
    if epoch % 100 == 0:
        print(epoch, train_loss)

y_predicted = mcqrnn_module(inputs=x_train_transform, tau=taus_transform,)
