import tensorflow as tf
from module.generate_example_dataset import generate_dataset


x_train, y_train = generate_dataset(1000)
print(tf.__version__)
print(x_train.shape)
