import sys

from needle.data import MNISTDataset, DataLoader

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    m = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob),
                      nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(m)
    return nn.Sequential(res, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    module = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
        module.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    module.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*module)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    loss_sum, errors_sum = 0., 0.
    steps = 0
    sample_num = len(dataloader.dataset)
    if opt is not None:
        model.train()
        for X, y in dataloader:
            steps += 1
            logits = model(X)
            loss = loss_func(logits, y)

            errors_sum += (logits.numpy().argmax(axis=1) != y.numpy()).sum()
            loss_sum += loss.numpy()

            loss.backward()
            opt.step()
        return errors_sum / sample_num, loss_sum / steps
    else:
        model.eval()
        for X, y in dataloader:
            steps += 1
            logits = model(X)
            loss = loss_func(logits, y)

            errors_sum += (logits.numpy().argmax(axis=1) != y.numpy()).sum()
            loss_sum += loss.numpy()

        return errors_sum / sample_num, loss_sum / steps
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_data = MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_data = MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = ndl.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_data, batch_size)
    module = MLPResNet(28 * 28, hidden_dim, num_classes=10)
    opt = optimizer(module.parameters(), lr=lr, weight_decay=weight_decay)
    train_err, train_loss = 0., 0.
    for i in range(epochs):
        train_err, train_loss = epoch(train_loader, module, opt)
    test_err, test_loss = epoch(test_loader, module)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
