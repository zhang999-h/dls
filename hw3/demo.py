import needle as ndl
from needle import backend_ndarray as nd

x = nd.NDArray([1, 2, 3, 4, 5, 6], )
x.compact()
print(1)
print(x[1:4:2])
