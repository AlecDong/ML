import torch
import dataset
from scratch import Accumulator, Animator, evaluate_accuracy, accuracy
import matplotlib.pyplot as plt

# def dropout_layer(X, dropout):
#     if dropout == 1:
#         return torch.zeros_like(X)
#     elif dropout == 0:
#         return X
#     else:
#         return (torch.rand(X.shape) > dropout).float() * X / (1.0 - dropout)

# # 2 hidden layers 
# class Net(torch.nn.Module):
#     def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training = True):
#         super().__init__()
#         self.num_inputs = num_inputs
#         self.training = is_training
#         self.lin1 = torch.nn.Linear(num_inputs, num_hiddens1)
#         self.lin2 = torch.nn.Linear(num_hiddens1, num_hiddens2)
#         self.lin3 = torch.nn.Linear(num_hiddens2, num_outputs)
#         self.relu = torch.nn.ReLU()
    
#     def forward(self, X):
#         H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
#         if self.training: # Apply dropout if training but not if testing
#             H1 = dropout_layer(H1, dropout1)
#         H2 = self.relu(self.lin2(H1))
#         if self.training: # Apply dropout if training but not if testing
#             H2 = dropout_layer(H2, dropout2)
#         output = self.lin3(H2)
#         return output



# # 2 hidden layers with 256 units each
# num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
# dropout1, dropout2 = 0.2, 0.5

# net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
# num_epochs, lr, batch_size = 10, 0.5, 256
# loss = torch.nn.CrossEntropyLoss(reduction='none') # For classification
# train_iter, test_iter = dataset.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)

# if __name__ == '__main__':
#     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
#                             legend=['train loss', 'train acc', 'test acc'])
#     for epoch in range(num_epochs):
#         net.train()
#         metric = Accumulator(3)
#         for X, y in train_iter:
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             trainer.zero_grad()
#             l.mean().backward()
#             trainer.step()
#             metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#         train_metrics = (metric[0] / metric[2], metric[1] / metric[2])
#         test_accuracy = evaluate_accuracy(net, test_iter) # Will set net.eval()
#         animator.add(epoch + 1, train_metrics + (test_accuracy,))
#     plt.show()

''' Concise '''
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5
num_epochs, lr, batch_size = 10, 0.5, 256
train_iter, test_iter = dataset.load_data_fashion_mnist(batch_size)

net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(dropout1),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(dropout2),
    torch.nn.Linear(256,10)
)

# Initialize parameters
net.apply(lambda x : torch.nn.init.normal_(x.weight, std=0.01) if type(x) == torch.nn.Linear else None)

loss = torch.nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

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