from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        sum_Z = array_api.sum(array_api.exp(Z - max_Z), axis=self.axes)
        log_Z = array_api.log(sum_Z)
        ans = log_Z + max_Z.reshape(log_Z.shape)
        return ans
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inputs = node.inputs[0]
        max_z = inputs.cached_data.max(self.axes, keepdims=True)
        exp_z = exp(inputs - max_z)
        sum_exp_z = summation(exp_z, self.axes)
        grad = exp_z/(sum_exp_z.reshape(max_z.shape).broadcast_to(inputs.shape))
        new_out_grad=out_grad.reshape(max_z.shape).broadcast_to(inputs.shape)
        ans= new_out_grad*grad
        return ans
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

