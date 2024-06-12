import numpy as np
import torch
import torch.nn as nn


def softmax(Z):
    Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return Z / Z.sum(axis=-1, keepdims=True)


# 和LSTM类似，不设置三个矩阵（这样会做三次运算）
# 而是只设置一个大矩阵做一次运算，然后将结果split开
def self_attention(X, mask, W_QKV, W_out):
    K, Q, V = np.split(X @ W_QKV, 3, axis=-1)
    # 原讲义中最开始实现为K @ Q.T，和定义更接近，但是后面为了mini-batch场景改成了swapaxes
    # attn = softmax(K @ Q.T / np.sqrt(X.shape[-1]) + mask)

    attn = softmax(K @ Q.swapaxes(-1, -2) / np.sqrt(X.shape[-1]) + mask)
    return attn @ V @ W_out, attn


T, d = 100, 64
attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
M = torch.triu(-float("inf") * torch.ones(T, T), 1)
X = torch.randn(1, T, d)
Y_, A_ = attn(X, X, X, attn_mask=M)

Y, A = self_attention(X[0].numpy(), M.numpy(),
                      attn.in_proj_weight.detach().numpy().T,
                      attn.out_proj.weight.detach().numpy().T)
print(np.linalg.norm(A - A_[0].detach().numpy()))
print(np.linalg.norm(Y - Y_[0].detach().numpy()))

# minibatch
C = np.random.randn(5, 4, 10, 3)
D = np.random.randn(3, 6)
print((C @ D).shape)  # 5, 4, 10, 6
# 实际上通过下面的变换得到同样的结果。reshape以后退化成普通的二维矩阵乘法
(C.reshape(-1, 3) @ D).reshape(5, 4, 10, 6)

C = np.random.randn(5, 10, 3)
D = np.random.randn(5, 3, 6)
(C @ D).shape  # 5, 10, 6。具体实现不是通过大矩阵计算，而是通过循环计算
# numpy实际上支持更多维度的计算。例如可以计算(5, 4, 10, 3) x (5, 4, 3, 6)
# 最后也能得到一个形状为(5, 4, 10, 6)的张量
# 但是在torch.bmm函数里，严格限制两个输入张量的维度都必须是3维
# 两者第一个维度（可以理解为batch大小）必须相等


N = 10
M = torch.triu(-float("inf") * torch.ones(T, T), 1)
X = torch.randn(N, T, d)
Y_, A_ = attn(X, X, X, attn_mask=M)
Y, A = self_attention(X.numpy(), M.numpy(),
                      attn.in_proj_weight.detach().numpy().T,
                      attn.out_proj.weight.detach().numpy().T)
print(np.linalg.norm(A - A_.detach().numpy()))
print(np.linalg.norm(Y - Y_.detach().numpy()))


###################################################
def multihead_attention(X, mask, heads, W_KQV, W_out):
    N, T, d = X.shape
    K, Q, V = np.split(X @ W_KQV, 3, axis=-1)
    K, Q, V = [a.reshape(N, T, heads, d // heads).swapaxes(1, 2) for a in (K, Q, V)]

    attn = softmax(K @ Q.swapaxes(-1, -2) / np.sqrt(d // heads) + mask)
    return (attn @ V).swapaxes(1, 2).reshape(N, T, d) @ W_out, attn


heads = 4
attn = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)
Y_, A_ = attn(X, X, X, attn_mask=M)
Y, A = multihead_attention(X.numpy(), M.numpy(), 4,
                           attn.in_proj_weight.detach().numpy().T,
                           attn.out_proj.weight.detach().numpy().T)
print(A_.shape)
print(A.shape)


print(np.linalg.norm(Y - Y_.detach().numpy()))
print(np.linalg.norm(A.mean(1) - A_.detach().numpy()))


##############################################
def layer_norm(Z, eps):
    return (Z - Z.mean(axis=-1, keepdims=True)) / np.sqrt(Z.var(axis=-1, keepdims=True) + eps)


def relu(Z):
    return np.maximum(Z, 0)


def transformer(X, mask, heads, W_KQV, W_out, W_ff1, W_ff2, eps):
    Z = layer_norm(multihead_attention(X, mask, heads, W_KQV, W_out)[0] + X, eps)
    return layer_norm(Z + relu(Z @ W_ff1) @ W_ff2, eps)
trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=0.0, batch_first=True)
trans.linear1.bias.data.zero_()
trans.linear2.bias.data.zero_();
Y_ = trans(X, M)
Y = transformer(X.numpy(), M.numpy(), heads,
                trans.self_attn.in_proj_weight.detach().numpy().T,
                trans.self_attn.out_proj.weight.detach().numpy().T,
                trans.linear1.weight.detach().numpy().T,
                trans.linear2.weight.detach().numpy().T,
                trans.norm1.eps)
print(np.linalg.norm(Y - Y_.detach().numpy()))
