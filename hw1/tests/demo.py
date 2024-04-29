import sys

sys.path.append("../python")

import needle as ndl

a = ndl.Tensor([1], dtype="float32")
b = ndl.Tensor([2], dtype="float32")
c = a + b
print(c.detach().inputs)  # detach

