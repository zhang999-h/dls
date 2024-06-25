# 1.52560
import torch
import numpy as np

# 创建一个数组
array = np.random.randn(1, 2,2)
print(array)
sp = np.split(array, 1, axis=0)
print(sp[0].shape)
st = np.stack(sp, axis=0)
print(st)
# x = np.full((1, 1, 1), 1.7886285)
# h0 = np.full((1, 1, 1), 0.43650985)
#
# model_ = torch.nn.RNN(1, 1, num_layers=1, bias=True, nonlinearity='tanh')
# output_, h_ = model_(torch.tensor(x), torch.tensor(h0))
