import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421334343, tf.float64)

print("sport is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("number is a {}-d Tensor".format(tf.rank(number).numpy()))

sports = tf.constant(["Tennis", 'Basketball'], tf.string)
numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

print("sports is a {}-d Tensor".format(tf.rank(sports).numpy()))
print("numbers is a {}-d Tensor".format(tf.rank(numbers).numpy()))

matrix = tf.constant([[123, 12, 32], [23, 32, 22]], tf.int64)
print(tf.rank(matrix).numpy())

images = tf.zeros([10, 256, 256, 3])
print(tf.rank(images).numpy())

row_vector = matrix[1]
column_vector = matrix[:, 2]
scalar = matrix[1,2]
print("`row_vector`: {}".format(row_vector.numpy()))
print("`column_vector`: {}".format(column_vector.numpy()))
print("`scalar`: {}".format(scalar.numpy()))

def func(a, b):
    c = a + b
    d = b - 1
    e = c * d
    return e

a, b = 15, 61

print(func(a, b))

# Making own dense layer
class OurDenseLayer(tf.keras.layers.Layer):
    """
    Network Layer -> sigmoid(XW + b)
    n_output_nodes: number of output nodes (percentage of each class (i.e. if classifying either dog or cat, n_output_nodes = 2))
    input_shape: shape of the input (1, number of features) (each data point used separately)
    x: the input to the layer
    """
    def __init__(self, n_output_nodes):
        super().__init__()
        self.n_output_nodes = n_output_nodes
    
    def build(self, input_shape):
        # d is the number of features
        d = int(input_shape[-1])
        # Weights has size [number of features, number of classes]
        self.W = self.add_weight("weight", shape=[d, self.n_output_nodes])
        # Bias has size [1, number of classes]
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes])
    
    def call(self, x):
        z = tf.matmul(x, self.W) + self.b
        y = tf.sigmoid(z)
        return y

# Testing own dense layer
tf.random.set_seed(1)
layer = OurDenseLayer(3)
layer.build((1, 2))
x_input = tf.constant([[1,2.]])
y = layer.call(x_input)

# Prediction (using randomly initialized weights + biases)
print(y.numpy())

# Using keras dense layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Number of classes
n_output_nodes = 3

# Create the model
model = Sequential()
# Add a dense layer
dense_layer = Dense(n_output_nodes, activation="sigmoid")
model.add(dense_layer)

# Predict the output of an input (using randomly initialized weights + biases)
x_input = tf.constant([[1, 2.]])
model_output = model(x_input).numpy()
print(model_output)

# using tf.keras.model subclass
from tensorflow.keras import Model

class SubclassModel(Model):
    def __init__(self, n_output_nodes):
        super().__init__()

        self.dense_layer = Dense(n_output_nodes, activation="sigmoid")

    def call(self, inputs):
        return self.dense_layer(inputs)

n_output_nodes = 3
model = SubclassModel(n_output_nodes)

x_input = tf.constant([[1,2.]])

print(model.call(x_input).numpy())

# This allows for custom behaviour 
# subclass that can output the input without change:
class IdentityModel(Model):
    def __init__(self, n_output_nodes):
        super().__init__()
        self.dense_layer = Dense(n_output_nodes, activation='sigmoid')

    # If isidentity is true, just output the original input
    def call(self, inputs, isidentity=False):
        if not isidentity:
            return self.dense_layer(inputs)
        return inputs

# testing
n_output_nodes = 3
model = IdentityModel(n_output_nodes)

x_input = tf.constant([[1, 2.]])
out_activate = model.call(x_input)
out_identity = model.call(x_input, True)

print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy()))

# Automatic differentiation can be done with tf.GradientTape
# All forward pass operations are recorded to the gradient tape
# when the tape is called, the tape is played backwards to compute gradient
# it is then discarded unless GradientTape(persistent = True)
x = tf.Variable(3.0)
# tensorflow variables are automatically tracked by gradient tape
with tf.GradientTape() as tape:
    # defined function
    y = x*x

# derivative of x^2 at x=3
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())

# stochastic gradient descent
x = tf.Variable([tf.random.normal([1])])
print("initializing x={}".format(x.numpy()))

learning_rate = 0.01
history = []
# target value
x_f = 4

# try to minimize loss function = (x-x_f)^2
for i in range(500):
    with tf.GradientTape() as tape:
        loss = (x-x_f)**2

    grad = tape.gradient(loss, x)
    new_x = x - learning_rate*grad
    x.assign(new_x)
    history.append(x.numpy()[0])

plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()