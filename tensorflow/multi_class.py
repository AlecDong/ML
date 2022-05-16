import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0


def plot_curve(epochs, hist, list_of_metrics):

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()

def create_model(lr):
    model = tf.keras.models.Sequential()

    # Each picture is a 28 by 28 array
    # Flatten it to a 1D array of size 784
    model.add(layers.Flatten(input_shape=(28, 28)))

    # Hidden layer with 64 units and l2 regularization
    model.add(layers.Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003)))
    model.add(layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003)))
    
    # Dropout regularization
    model.add(layers.Dropout(rate=0.2))

    # Output layer (10 classes)
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr = lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def train_model(model, features, label, epochs, batch_size=None, validation_split=0.1):
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=validation_split)

    return history.epoch, pd.DataFrame(history.history)

lr = 0.003
epochs = 70
batch_size = 2000
validation_split = 0.2

model = create_model(lr)

epochs, hist = train_model(model, x_train_norm, y_train, epochs, batch_size, validation_split)


list_of_metrics_to_plot = ['accuracy', 'loss']
plot_curve(epochs, hist, list_of_metrics_to_plot)


model.evaluate(x = x_test_norm, y = y_test, batch_size = batch_size)