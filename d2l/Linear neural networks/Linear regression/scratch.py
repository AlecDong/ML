import torch
import random
import matplotlib.pyplot as plt

def generate_data(w, b, num_examples):
    """
    (tensor, num, num) -> (tensor, tensor)
    Generates data with noise
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = generate_data(true_w, true_b, 1000)

fig = plt.figure(figsize=(5,4))
plt.scatter(features[:, (0)].detach().numpy(), labels.detach().numpy(), 1)
plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.show()

def batch_generator(batch_size, features, labels):
    """
    (num, tensor, tensor) -> (tensor, tensor)
    Generator that separates the input into minibatches
    """
    num_examples = len(features)
    indices = list(range(num_examples))

    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# batch_size = 10
# for X, y in batch_generator(batch_size, features, labels):
#     print(X)
#     print(y)

""" Initialize parameters """
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) # Column vector
b = torch.zeros(1, requires_grad=True) # Broadcasting will fix the size

""" Defining linear model """
def linear_regression(X, w, b):
    """
    (tensor, tensor, tensor) -> tensor
    Makes the model y = Xw + b
    """
    return torch.matmul(X, w) + b

""" Defining loss """
def squared_loss(y_hat, y):
    """
    (tensor, tensor) -> tensor
    Returns the squared loss
    """
    return 0.5 * (y_hat-y.reshape(y_hat.shape)) ** 2

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
lr = 0.03
num_epochs = 3
batch_size = 10
net = linear_regression
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in batch_generator(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward() # Gradient of loss with respect to w and b
        sgd([w, b], lr, batch_size)
    
    # Test the model on training data
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        
print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')