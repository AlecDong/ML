import torch
import dataset
from scratch import Accumulator, Animator, evaluate_accuracy, accuracy
import matplotlib.pyplot as plt

net = torch.nn.Sequential(torch.nn.Flatten(), # Flatten image
                          torch.nn.Linear(784, 256), # Hidden layer
                          torch.nn.ReLU(), # Activation function
                          torch.nn.Linear(256, 10)) # Output layer

# Initialize weights to random normal distribution with std of 0.01 for all linear layers
net.apply(lambda m : torch.nn.init.normal_(m.weight, std=0.01) if type(m) == torch.nn.Linear else None)
batch_size = 256
lr = 0.1
num_epochs = 10
loss = torch.nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# optimizer with decay for weights only (default both weights and biases have decay)
# trainer = torch.optim.SGD(
#     [
#         {"params": [net[1].weight, net[3].weight], "weight_decay": 3},
#         {"params": [net[1].bias, net[3].bias]}
#     ], lr=lr
# )

train_iter, test_iter = dataset.load_data_fashion_mnist(batch_size)

if __name__ == '__main__':
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        train_metrics = (metric[0] / metric[2], metric[1] / metric[2])
        test_accuracy = evaluate_accuracy(net, test_iter) # Will set net.eval()
        animator.add(epoch + 1, train_metrics + (test_accuracy,))
    plt.show()
