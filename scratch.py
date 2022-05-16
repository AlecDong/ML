import torch
from IPython import display
import matplotlib.pyplot as plt
from dataset import load_data_fashion_mnist, show_images, get_fashion_mnist_labels

class Accumulator: 
    """For accumulating sums over `n` variables. A way to store metrics """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] # add to multiple metrics simultaneously

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.X, self.Y, self.fmts = None, None, fmts
    
    def config_axes(self):
        self.axes[0].set_xlabel(self.xlabel)
        self.axes[0].set_ylabel(self.ylabel)
        self.axes[0].set_xscale(self.xscale)
        self.axes[0].set_yscale(self.yscale)
        self.axes[0].set_xlim(self.xlim)
        self.axes[0].set_ylim(self.ylim)
        if self.legend:
            self.axes[0].legend(self.legend)
        self.axes[0].grid()

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        

""" Prediction function (softmax) """
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # broadcasting happens here

""" Network """
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) # Flattens the images into a vector

""" Cross-entropy loss function """
def cross_entropy(y_hat, y):
    # Below, y_hat[range(len(y_hat)), y] is the same as [y_hat[0][y[0]], y_hat[1][y[1]], ..., y_hat[n][y[n]]]
    return -torch.log(y_hat[range(len(y_hat)), y]) # only get the predicted probability for the true values

""" Accuracy """
def accuracy(y_hat, y):
    """ Returns number of correct guesses """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # Checks if it is a matrix
        y_hat = y_hat.argmax(axis=1) # Argmax gives the position of the largest value in each row (axis=1)
    cmp = y_hat.type(y.dtype) == y # Change float into int to make equality work
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval() # use the network to predict (evaluate) from inputs
    metric = Accumulator(2) # Keep number of correct predictions and number of predictions over all batches

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

""" Defining optimization """
def sgd(params, lr, batch_size):
    """
    (list of tensors, num, num) -> NoneType
    Updates the parameters of the linear model
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

""" Training """
def train_epoch(net, train_iter, loss, updater):
    metric = Accumulator(3) ## Sum of training loss, sum of training accuracy, no. of examples
    for X, y in train_iter:
        y_hat = net(X) # Prediction using softmax
        l = loss(y_hat, y) # Loss from prediction
        l.sum().backward() # Calculate gradients
        updater([W, b], lr, batch_size) # Update parameters using stochastic gradient descent
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]  # Return training loss and accuracy
 
if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 28 * 28 # 28 by 28 pixels
    num_outputs = 10 # 10 classes

    ''' Initialize parameters '''
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    lr = 0.1
    num_epochs = 10
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, cross_entropy, sgd)
        test_accuracy = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_accuracy,))
    plt.show()

    n = 6
    for X, y in test_iter:
        break
    true_vals = get_fashion_mnist_labels(y)
    prediction_vals = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(true_vals, prediction_vals)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])