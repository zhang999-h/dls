import torch
import torch.nn as nn
import numpy as np

model = nn.LSTMCell(20, 100)
print(model.weight_hh.shape)
print(model.weight_ih.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstm_cell(x, h, c, W_hh, W_ih, b):
    i, f, g, o = np.split(W_ih @ x + W_hh @ h + b, 4)
    i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f * c + i * g
    h_out = o * np.tanh(c_out)
    return h_out, c_out


x = np.random.randn(1, 20).astype(np.float32)
h0 = np.random.randn(1, 100).astype(np.float32)
c0 = np.random.randn(1, 100).astype(np.float32)
print(x.shape)
h_, c_ = model(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))

h, c = lstm_cell(x[0], h0[0], c0[0],
                 model.weight_hh.detach().numpy(),
                 model.weight_ih.detach().numpy(),
                 (model.bias_hh + model.bias_ih).detach().numpy())

print(np.linalg.norm(h_.detach().numpy() - h),
      np.linalg.norm(c_.detach().numpy() - c))

model = nn.LSTM(20, 100, num_layers=1)

X = np.random.randn(50, 20).astype(np.float32)
h0 = np.random.randn(1, 100).astype(np.float32)
c0 = np.random.randn(1, 100).astype(np.float32)


def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], h.shape[0]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t, :] = h
    return H, c


H, cn = lstm(X, h0[0], c0[0],
             model.weight_hh_l0.detach().numpy(),
             model.weight_ih_l0.detach().numpy(),
             (model.bias_hh_l0 + model.bias_ih_l0).detach().numpy())
# LSTM 的结果
H_, (hn_, cn_) = model(torch.tensor(X)[:, None, :],
                       (torch.tensor(h0)[:, None, :],
                        torch.tensor(c0)[:, None, :]))

print(np.linalg.norm(H - H_[:, 0, :].detach().numpy()),
      np.linalg.norm(cn - cn_[0, 0, :].detach().numpy()))

print((H_[0, 0, :]).shape)

#############################
#####   batch
#############################
print("\n\nLSTM batch:")


def lstm_cell(x, h, c, W_hh, W_ih, b):
    # h:batch_size*hidden_size  x: batch_size*input_size
    i, f, g, o = np.split(x @ W_ih + h @ W_hh + b[None, :], 4, axis=1)
    i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f * c + i * g
    h_out = o * np.tanh(c_out)
    return h_out, c_out


def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t, :, :] = h
    return H, c


X = np.random.randn(50, 80, 20).astype(np.float32)
h0 = np.random.randn(80, 100).astype(np.float32)
c0 = np.random.randn(80, 100).astype(np.float32)
H_, (hn_, cn_) = model(torch.tensor(X),
                       (torch.tensor(h0)[None, :, :],
                        torch.tensor(c0)[None, :, :]))
H, cn = lstm(X, h0, c0,
             model.weight_hh_l0.detach().numpy().T,  # 转置用来矩阵乘法
             model.weight_ih_l0.detach().numpy().T,
             (model.bias_hh_l0 + model.bias_ih_l0).detach().numpy())
print(np.linalg.norm(H - H_.detach().numpy()),
      np.linalg.norm(cn - cn_[0].detach().numpy()))

#
def train_lstm(X, h0, c0, Y, W_hh, W_ih, b, opt):
    H, cn = lstm(X, h0, c0, W_hh, W_ih, b)
    l = loss(H, Y)
    l.backward()
    opt.step()

def train_deep_lstm(X, h0, c0, Y, W_hh, W_ih, b, opt):
    H = X
    depth = len(W_hh)   # W_hh此时是一个参数列表, h0, c0, Y, W_ih, b同样
    for d in range(depth):
        H, cn = lstm(H, h0[d], c0[d], W_hh[d], W_ih[d], b[d])
    l = loss(H, Y)
    l.backward()
    opt.step()
