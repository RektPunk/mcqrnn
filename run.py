import numpy as np
import tensorflow as tf

from mcqrnn import (
    DataTransformer,
    Mcqrnn,
    TiltedAbsoluteLoss,
    generate_example,
    train_step,
)

# Examples setting
EPOCHS = 5000
LEARNING_RATE = 0.15
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
N_SAMPLES = 1000
OUT_FEATURES = 15
DENSE_FEATURES = 15

x_train, y_train = generate_example(N_SAMPLES)
x_test, y_test = generate_example(N_SAMPLES)
taus = np.array(TAUS)
data_transformer = DataTransformer(
    x=x_train,
    taus=taus,
    y=y_train,
)
x_train_transform, y_train_transform, taus_transform = data_transformer()
mcqrnn_regressor = Mcqrnn(
    out_features=OUT_FEATURES,
    dense_features=DENSE_FEATURES,
    activation=tf.nn.sigmoid,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
tilted_absolute_loss = TiltedAbsoluteLoss(tau=taus_transform)
for epoch in range(EPOCHS):
    train_loss = train_step(
        model=mcqrnn_regressor,
        inputs=x_train_transform,
        output=y_train_transform,
        tau=taus_transform,
        loss_func=tilted_absolute_loss,
        optimizer=optimizer,
    )
    if epoch % 1000 == 0:
        print(epoch, train_loss)

x_test_transform, taus_transform = data_transformer.transform(x=x_test, input_taus=taus)
y_test_predicted = mcqrnn_regressor(
    inputs=x_test_transform,
    tau=taus_transform,
)
y_test_predicted_reshaped = y_test_predicted.numpy().reshape(N_SAMPLES, len(TAUS)).T
