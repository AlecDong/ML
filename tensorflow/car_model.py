import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers

pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 15
pd.options.display.max_columns = None

# Provide the names for the columns since the CSV file with the data does
# not have a header row.
feature_names = ['symboling', 'normalized-losses', 'make', 'fuel-type',
        'aspiration', 'num-doors', 'body-style', 'drive-wheels',
        'engine-location', 'wheel-base', 'length', 'width', 'height', 'weight',
        'engine-type', 'num-cylinders', 'engine-size', 'fuel-system', 'bore',
        'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
        'highway-mpg', 'price']

car_data = pd.read_csv('https://storage.googleapis.com/mledu-datasets/cars_data.csv',
                        sep=',', names=feature_names, header=None, encoding='latin-1')

car_data = car_data.reindex(np.random.permutation(car_data.index))

numeric_feature_names = ['symboling', 'normalized-losses', 'wheel-base',
        'length', 'width', 'height', 'weight', 'engine-size', 'horsepower',
        'peak-rpm', 'city-mpg', 'highway-mpg', 'bore', 'stroke',
         'compression-ratio']

label = "price"
categorical_feature_names = list(set(feature_names) - set(numeric_feature_names) - set([label]))

for feature_name in numeric_feature_names + [label]:
  car_data[feature_name] = pd.to_numeric(car_data[feature_name], errors='coerce')

car_data = car_data.dropna()

epsilon = 1e-7
car_mean = car_data[numeric_feature_names].mean()
car_std = car_data[numeric_feature_names].std()
car_data[numeric_feature_names] = (car_data[numeric_feature_names] - car_std) / (car_mean + epsilon)

one_hot = pd.get_dummies(car_data[categorical_feature_names])
car_data_1 = pd.concat([one_hot, car_data[numeric_feature_names]], axis=1)
car_data_2 = pd.concat([car_data_1, car_data[label]], axis=1)

car_data_train = car_data_2.sample(frac=0.9, random_state=0)
car_data_test = car_data_2.drop(car_data_train.index)

print(car_data_train.head())

# feature_columns = []

# for feature_name in numeric_feature_names:
#     feature_columns.append(feature_column.numeric_column(feature_name))

# for feature_name in categorical_feature_names:  
#     feature_columns.append(feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list(
#                         feature_name, vocabulary_list=car_data[feature_name].unique())))
                        
# feature_layer = layers.DenseFeatures(feature_columns)

def create_model(lr, features, metrics=["accuracy"]):
    feat_inputs = []
    inputs = []
    for feature in features:
        inp = tf.keras.Input(shape=(1,))
        inputs.append(inp)
        feat_inputs.append(inp)
    
    x = layers.concatenate(feat_inputs)
    x = layers.Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003))(x)
    x = layers.Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003))(x)
    x = layers.Dropout(rate=0.3)(x)
    outputs = layers.Dense(units=1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # model.add(feature_layer)
    # model.add(layers.Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003)))
    # model.add(layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003)))
    # model.add(layers.Dropout(rate=0.2))
    # model.add(layers.Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003)))
    # model.add(layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=0.003)))
    
    # model.add(layers.Dense(units=1, activation="linear"))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr), loss="mse", metrics=metrics)
    return model

def train_model(model, dataset, label_name, epochs, batch_size = None, validation_split=0.1):
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=2)

    return history.epoch, pd.DataFrame(history.history)
    
lr = 0.005
epochs = 500
batch_size = 5
val_split = 0.1
metrics = ['mae', 'mse']

# model = create_model(lr, feature_layer, metrics)
model = create_model(lr, car_data_1.columns, metrics)

epochs, hist = train_model(model, car_data_train, label, epochs, batch_size, val_split)

features = {name:np.array(value) for name, value in car_data_test.items()}
label = np.array(features.pop(label))

model.evaluate(x = features, y = label, batch_size=batch_size)