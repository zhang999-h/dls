import numpy as np

import needle as ndl
from needle import backend_ndarray as nd

_A = np.random.random((1, 1, 1)).astype(np.float32)
a = ndl.Tensor(nd.array(_A), device=ndl.cpu())
x = nd.array(_A)
b = x.max()
print(_A)
print(b.is_compact())

Z = np.zeros((3, 2))
slices = []
slices.append(slice(0, 3))
slices.append(slice(0, 1))
print(slices)
arr1 = ndl.Tensor(nd.array([1, 2, 3]), device=ndl.cpu())
arr2 = ndl.Tensor(nd.array([4, 5, 6]), device=ndl.cpu())
Z = ndl.stack([arr1,arr2],axis=1)
print(Z)
