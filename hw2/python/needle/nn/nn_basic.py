"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
            self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(
            init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_mul_W = X.matmul(self.weight)
        b = self.bias.broadcast_to(X_mul_W.shape)
        return ops.add(X_mul_W, b)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        new_len = 1
        for s in shape[1:]:
            new_len = s * new_len
        return X.reshape((shape[0], new_len))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot_y = init.one_hot(logits.shape[1], y)
        z_y = ops.summation(logits * one_hot_y, axes=1)
        loss = ops.logsumexp(logits, axes=1) - z_y
        return ops.summation(loss) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, dtype=dtype))
        self.running_mean = Parameter(init.zeros(dim, dtype=dtype))
        self.running_var = Parameter(init.ones(dim, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        layer_size = x.shape[1]

        if self.training:
            mean = (x.sum(0) / batch_size)
            var = ops.summation(ops.power_scalar(x - mean.reshape((1, layer_size)).broadcast_to(x.shape),
                                                 2), axes=0) / batch_size

            # update running_mean&running_var for testing
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            mean = mean.reshape((1, layer_size)).broadcast_to(x.shape)
            var = var.reshape((1, layer_size)).broadcast_to(x.shape)

            norm = ((x - mean) /
                    ops.power_scalar(var + self.eps, 0.5))
            ans = self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
            return ans
        else:
            mean = self.running_mean.reshape((1, layer_size)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, layer_size)).broadcast_to(x.shape)
            norm = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
            ans = self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
            return ans
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = (x.sum(1) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = ops.summation(ops.power_scalar(x - mean, 2), axes=1) / x.shape[1]
        var = var.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        norm = ((x - mean) /
                ops.power_scalar(var + self.eps, 0.5))
        ans = self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        return ans
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training is True:
            a = init.randb(*x.shape, p=1 - self.p)
            return x * a / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
