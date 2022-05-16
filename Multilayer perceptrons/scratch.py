import torch
import dataset
import matplotlib.pyplot as plt
from IPython import display

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

""" Training """
def train_epoch(net, train_iter, loss, updater):
    metric = Accumulator(3) ## Sum of training loss, sum of training accuracy, no. of examples
    for X, y in train_iter:
        y_hat = net(X) # Prediction using softmax
        l = loss(y_hat, y) # Loss from prediction
        l.sum().backward() # Calculate gradients
        updater(params, lr, batch_size) # Update parameters using stochastic gradient descent
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]  # Return training loss and accuracy

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

def relu(X): # Activation function
    return torch.max(X, torch.zeros_like(X))

def net(X):
    X = X.reshape((-1, num_inputs)) # Flatten image
    H = relu(X @ W1 + b1) # Same as torch.mm(X, W1) + b1
    return (H @ W2 + b2)

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = dataset.load_data_fashion_mnist(batch_size)

    num_inputs = 28 * 28
    num_outputs = 10
    num_hiddens = 256 # The width of hidden layers is usually in powers of 2

    # Every layer needs weights and biases
    W1 = torch.nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01) # randum numbers with mean 0 and variance 0.01
    b1 = torch.nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = torch.nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01) # randum numbers with mean 0 and variance 0.01
    b2 = torch.nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    loss = torch.nn.CrossEntropyLoss(reduction='none')

    num_epochs = 10
    lr = 0.1
    updater = torch.optim.SGD(params, lr=lr)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        for X, y in train_iter:
            # Compute gradients and update parameters
            y_hat = net(X) # Prediction using softmax
            l = loss(y_hat, y) # Loss from prediction
            updater.zero_grad()
            l.mean().backward() # Calculate gradients
            updater.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        train_metrics = (metric[0] / metric[2], metric[1] / metric[2])
        test_accuracy = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_accuracy,))
    plt.show()