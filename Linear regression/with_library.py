import torch
from torch.utils import data
from torch import nn

def generate_data(w, b, num_examples):
    """
    (tensor, num, num) -> (tensor, tensor)
    Generates data with noise
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

def load_array(data_arrays, batch_size, is_train = True):
    """
    (tuple of tensors, num, boolean) -> Pytorch data iterator
    Loads the data as minibatches
    """
    dataset = data.TensorDataset(*data_arrays) # * allows variable number of arguments passed from calling environment (acts like tuple).
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

""" Create the data """
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = generate_data(true_w, true_b, 1000)

""" Make iterator """
batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

""" Create model and initialize params """
network = nn.Sequential(nn.Linear(2, 1)) # sequential network with a linear (fully connected) layer with 2 input features and 1 output
network[0].weight.data.normal_(0, 0.01) # set weights of first layer (linear) to follow normal distribution with mean of 0, std deviation of 0.01
network[0].bias.data.fill_(0) # set bias of first layer (linear) to 0

loss = nn.MSELoss() # mean squared error loss function (average loss over examples)

trainer = torch.optim.SGD(network.parameters(), lr=0.03) # use stochastic gradient descent to optimize parameters (lr needed for minibatch)

""" Training """
epochs = 3
for epoch in range(epochs):
    for X, y in data_iter:
        l = loss(network(X), y) # finds the loss of the network's prediction and the actual value, y
        trainer.zero_grad() # clears the old gradients
        l.backward() # finds the gradient of the loss function with respect to w and b
        trainer.step() # updates the parameters (w and b) with the current gradients
    l = loss(network(features), labels) # finds the average loss on the training set after the epoch
    print(f"epoch {epoch}, loss {l:f}")


w = network[0].weight.data
print('error in estimating w:', true_w - w.reshape(true_w.shape))
b = network[0].bias.data
print('error in estimating b:', true_b - b)