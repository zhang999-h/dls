import needle as ndl
import needle.init as init
import numpy as np

np.random.seed(1337)

print(np.max([1, 2, 3], axis=(0)))
arr = np.arange(27).reshape((3, 3, 3))
l = ndl.Tensor(np.arange(27).reshape((3, 3, 3)))
sum_n = np.max(arr, (2))
print(sum_n)

print(l.shape)
ans = ndl.ops.logsumexp(l, 0)
print(ans)
a = ndl.Tensor([1])
b = ndl.Tensor([2])
c = a + b
f2 = init.kaiming_uniform(4, 1, device=None, dtype="float32").reshape((1, 4))
p = ndl.nn.Parameter
f = ndl.nn.Linear(7, 4)
print(type(p))
print(f2)
print(f.bias)

n = np.array([[1], [2], [3]])
br = np.broadcast_to(n, (3, 3, 3))
br2 = np.broadcast_to(n.reshape((1, 3, 1)), (3, 3, 3))
br3 = np.broadcast_to(n.reshape((1, 1, 3)), (3, 3, 3))

# print(br)
# print(br2)
# print(br3)
#
# print(np.array([1, 2, 3]).shape)

xx = np.arange(4)
e=np.eye(10,4)
print(e[[3,2]])
t=np.arange(16).reshape((4,4))
print(t[[3,2]])
