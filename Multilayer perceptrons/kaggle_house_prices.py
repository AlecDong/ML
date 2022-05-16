import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
def download(name, cache_dir=os.path.join('C:/Users/alecd/Desktop/PythonScripts/d2l/Multilayer perceptrons/', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)

import numpy as np
import pandas as pd
import torch
from torch import nn, relu
from torch.utils import data

def load_array(data_arrays, batch_size, is_train = True):
    """
    (tuple of tensors, num, boolean) -> Pytorch data iterator
    Loads the data as minibatches
    """
    dataset = data.TensorDataset(*data_arrays) # * allows variable number of arguments passed from calling environment (acts like tuple).
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# First 4 features, last 2 features, label from first 4 examples
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# Remove the id and sale price
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features.iloc[[0,1,2,3,-3,-2,-1], [0, 1, 2, 3, -3, -2, -1]])

# If test data were inaccessible, mean and standard deviation could be
# calculated from training data
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# Use one-hot encoding for categorical data
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape

# Convert to tensor
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1), dtype=torch.float32)

''' Training '''
loss = nn.MSELoss()
in_features = train_features.shape[1]
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256,1)
    )
    return net

def log_rmse(net, features, labels):
    # Set values < 1 equal to 1
    constrained_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(constrained_preds), torch.log(labels)))
    return rmse

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# def get_k_fold_data(k, i, X, y):
#     assert k > 1
#     fold_size = X.shape[0] // k
#     X_train, y_train = None, None
#     for j in range(k):
#         idx = slice(j * fold_size, (j + 1) * fold_size)
#         X_part, y_part = X[idx, :], y[idx]
#         if j == i:
#             X_valid, y_valid = X_part, y_part
#         elif X_train is None:
#             X_train, y_train = X_part, y_part
#         else:
#             X_train = torch.cat([X_train, X_part], 0)
#             y_train = torch.cat([y_train, y_part], 0)
#     return X_train, y_train, X_valid, y_valid

# def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
#            batch_size):
#     train_l_sum, valid_l_sum = 0, 0
#     for i in range(k):
#         data = get_k_fold_data(k, i, X_train, y_train)
#         net = get_net()
#         train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
#                                    weight_decay, batch_size)
#         train_l_sum += train_ls[-1]
#         valid_l_sum += valid_ls[-1]
#         print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
#               f'valid log rmse {float(valid_ls[-1]):f}')
#     return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 8, 200, 0.01, 0.5, 128
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
#                           weight_decay, batch_size)
# print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
#       f'avg valid log rmse: {float(valid_l):f}')

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = net(test_features).detach().numpy()
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('C:/Users/alecd/Desktop/PythonScripts/d2l/Multilayer perceptrons/submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)