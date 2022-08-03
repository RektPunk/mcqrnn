import tensorflow as tf
import numpy as np
from mcqrnn import generate_example, train_step
from mcqrnn import (
    TiltedAbsoluteLoss,
    Mcqrnn,
    DataTransformer,
)

x_train, y_train = generate_example(10)
taus = np.array([0.1, 0.5, 0.9])

data_transformer = DataTransformer(
    x=x_train,
    taus=taus,
    y=y_train,
)

x_train_transform, y_train_transform, taus_transform = data_transformer()

mcqrnn_module = Mcqrnn(
    out_features=10,
    dense_features=10,
    activation=tf.nn.sigmoid,
)

tilted_absolute_loss = TiltedAbsoluteLoss(tau = taus_transform)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005)

EPOCHS = 1000

for epoch in range(EPOCHS):
    train_loss = train_step(
        model = mcqrnn_module, 
        inputs = x_train_transform, 
        output = y_train_transform,
        tau = taus_transform, 
        loss_func = tilted_absolute_loss,
        optimizer = optimizer,
    )
    if epoch % 100 == 0:
        print(epoch, train_loss)


mcqrnn_module(
    inputs = x_train_transform,
    tau = taus_transform,
)
