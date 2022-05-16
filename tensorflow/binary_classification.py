import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers
"""import matplotlib.pyplot as plt"""

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle the training set

# Calculate the Z-scores of each column in the training set and
# write those Z-scores into a new pandas DataFrame named train_df_norm.
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std

# Calculate the Z-scores of each column in the test set and
# write those Z-scores into a new pandas DataFrame named test_df_norm.
test_df_mean = test_df.mean()
test_df_std  = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std

# Cutoff at around $265000 (75th percentile)
train_df_norm["median_house_value_is_high"] = train_df_norm["median_house_value"] > 1.0
test_df_norm["median_house_value_is_high"] = test_df_norm["median_house_value"] > 1.0

# Examine some of the values of the normalized training set. Notice that most 
# Z-scores fall between -2 and +2.
print(train_df_norm.head())

feature_columns = []
feature_columns.append(feature_column.numeric_column("median_income"))
feature_columns.append(feature_column.numeric_column("total_rooms"))
feature_columns.append(feature_column.numeric_column("population"))

resolution_in_Zs = 0.3

# Bucket latitude
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])), int(max(train_df_norm['latitude'])), resolution_in_Zs))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)

# Bucket longitude
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])), int(max(train_df_norm['longitude'])), resolution_in_Zs))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, longitude_boundaries)

# Cross (longitude and latitude)
lo_x_la = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feat = tf.feature_column.indicator_column(lo_x_la)
feature_columns.append(crossed_feat)

feature_layer = layers.DenseFeatures(feature_columns)

def create_model(learning_rate, feature_layer, metrics):
    model = tf.keras.models.Sequential()

    model.add(feature_layer)
    model.add(layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003)))
    model.add(layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)

    return model

def train_model(model, dataset, epochs, label_name, batch_size = None, shuffle = True):
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    return history.epoch, pd.DataFrame(history.history)

"""def plot_curve(epochs, hist, list_of_metrics):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()"""

lr = 0.001
epochs = 50
batch_size = 100
label = "median_house_value_is_high"
threshold = 0.5

metrics = [tf.keras.metrics.BinaryAccuracy(name="acc", threshold=threshold), 
            tf.keras.metrics.MeanSquaredError(name="mean_sq_err"),
            tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
            tf.keras.metrics.Recall(thresholds=threshold, name="recall")]

model = create_model(lr, feature_layer, metrics)

epochs, hist = train_model(model, train_df_norm, epochs, label, batch_size)

"""list_of_metrics_to_plot = ['accuracy', "loss", "mean_sq_err", "precision", "recall"]
plot_curve(epochs, hist, list_of_metrics_to_plot)"""

# Evaluate the accuracy of the model
features = {name:np.array(value) for name, value in test_df_norm.items()}
label = np.array(features.pop(label))

model.evaluate(x = features, y = label, batch_size=batch_size)