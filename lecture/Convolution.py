import time

import torch
import torch.nn as nn
import numpy as np


def conv_reference(Z, weight):
    # NHWC -> NCHW
    Z_torch = torch.tensor(Z).permute(0, 3, 1, 2)

    # KKIO -> OIKK
    W_torch = torch.tensor(weight).permute(3, 2, 0, 1)

    # run convolution
    out = nn.functional.conv2d(Z_torch, W_torch)

    # NCHW -> NHWC
    return out.permute(0, 2, 3, 1).contiguous().numpy()


Z = np.random.randn(10, 32, 32, 8)
W = np.random.randn(3, 3, 8, 16)
start_time = time.time()
out = conv_reference(Z, W)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码执行时间: {elapsed_time:.2f} 秒")

print(out.shape)


def conv_naive(Z, weight):
    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape

    out = np.zeros((N, H - K + 1, W - K + 1, C_out));
    for n in range(N):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for y in range(H - K + 1):
                    for x in range(W - K + 1):
                        for i in range(K):
                            for j in range(K):
                                out[n, y, x, c_out] += Z[n, y + i, x + j, c_in] * weight[i, j, c_in, c_out]
    return out


start_time = time.time()
out2 = conv_naive(Z, W)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码执行时间: {elapsed_time:.2f} 秒")

print(np.linalg.norm(out - out2))


def conv_matrix_mult(Z, weight):
    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape
    out = np.zeros((N, H - K + 1, W - K + 1, C_out))

    for i in range(K):
        for j in range(K):
            out += Z[:, i:i + H - K + 1, j:j + W - K + 1, :] @ weight[i, j]
    return out


Z = np.random.randn(100, 32, 32, 8)
W = np.random.randn(3, 3, 8, 16)

start_time = time.time()
out = conv_reference(Z, W)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码执行时间: {elapsed_time:.2f} 秒")

start_time = time.time()
out2 = conv_matrix_mult(Z, W)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码执行时间: {elapsed_time:.2f} 秒")

print(np.linalg.norm(out - out2))


###################################################################
# 为了简单起见，这里观察使用2维卷积核在2维图片上做卷积的操作

def conv_im2col(Z, weight):
    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape
    Ns, Hs, Ws, Cs = Z.strides

    inner_dim = K * K * C_in
    A = np.lib.stride_tricks.as_strided(Z, shape=(N, H - K + 1, W - K + 1, K, K, C_in),
                                        strides=(Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1, inner_dim)
    out = A @ weight.reshape(-1, C_out)
    return out.reshape(N, H - K + 1, W - K + 1, C_out)


Z = np.random.randn(100, 32, 32, 8)
W = np.random.randn(3, 3, 8, 16)
start_time = time.time()
out = conv_reference(Z, W)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码执行时间: {elapsed_time:.2f} 秒")

start_time = time.time()
out2 = conv_im2col(Z, W)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码执行时间: {elapsed_time:.2f} 秒")
print(np.linalg.norm(out - out2))
