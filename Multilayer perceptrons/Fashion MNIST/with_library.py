import torch
from dataset import load_data_fashion_mnist
from scratch import Animator, Accumulator
import matplotlib.pyplot as plt

def evaluate_accuracy(net, data_iter):  
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def accuracy(y_hat, y):  
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    """ Initialize model and parameters """
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)) # First flatten the image (28 by 28 into 1 by 784) and then add a linear layer with 784 inputs and 10 outputs
    net.apply(lambda m : torch.nn.init.normal_(m.weight, std=0.01) if type(m) == torch.nn.Linear else None)
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 10
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Set the model to training mode
        if isinstance(net, torch.nn.Module):
            net.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        for X, y in train_iter:
            # Compute gradients and update parameters
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        train_metrics = (metric[0] / metric[2], metric[1] / metric[2])
        test_accuracy = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_accuracy,))
    plt.show()

