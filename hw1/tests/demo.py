import sys

sys.path.append("../python")

import needle as ndl
import numpy as np
a = ndl.Tensor([1], dtype="float32")
b = ndl.Tensor([2], dtype="float32")
c = a + b
print(c.detach().inputs)  # detach

def f(x):
    return x
x = [3,4]
y=f(x)
y[0]=9
print(x)

