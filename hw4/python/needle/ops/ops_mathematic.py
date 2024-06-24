"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -(out_grad * (lhs / (rhs * rhs)))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)
        axes = list(range(a.ndim))
        if self.axes is None:
            self.axes = axes[-2:]  # last two axes
        axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]
        # 参数2是这个数组的所有维度序列（1,2,3,...）
        return a.permute(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return reshape(out_grad, node.inputs[0].shape)

        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_shape = out_grad.shape
        intput_shape = node.inputs[0].shape
        intput_shape_len = len(intput_shape) - 1
        axes_to_reduce = []
        for i in range(len(out_shape) - 1, -1, -1):
            if intput_shape_len < 0:
                axes_to_reduce.append(i)
                continue
            if intput_shape[intput_shape_len] != out_shape[i]:
                axes_to_reduce.append(i)
            intput_shape_len -= 1
        return reshape(summation(out_grad, tuple(axes_to_reduce)), intput_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(self.axes, int):
            self.axes = (self.axes,)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (List, Tuple)):
            for ax in self.axes:
                a = a.sum(axis=ax)
            return a
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape  # 输入的shape
        reduce_shape = list(input_shape)
        if self.axes is not None:
            for axis in self.axes:
                reduce_shape[axis] = 1
            grad = reshape(out_grad, reduce_shape)
        else:
            grad = out_grad
        return broadcast_to(grad, input_shape)


        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_tmp_grad = matmul(out_grad, transpose(rhs))
        rhs_tmp_grad = matmul(transpose(lhs), out_grad)

        # Adjust the shape of the gradients if necessary.
        # For example, if lhs.shape is (5, 4), and rhs.shape is (6, 6, 4, 3)
        # then out_grad.shape is (6, 6, 5, 3),
        # and now grad_A.shape is (6, 6, 5, 4) and grad_B.shape is (6, 6, 4, 3).
        # So we need to sum over the first two axes of grad_A.
        # grad_A(6, 6, 5, 4) --->>> grad_A(5, 4)
        if lhs_tmp_grad.shape != lhs.shape:
            lhs_tmp_grad = summation(lhs_tmp_grad, tuple(range(len(lhs_tmp_grad.shape) - 2)))
        if rhs_tmp_grad.shape != rhs.shape:
            rhs_tmp_grad = summation(rhs_tmp_grad, tuple(range(len(rhs_tmp_grad.shape) - 2)))

        return lhs_tmp_grad, rhs_tmp_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 注意不要修改原来a的值
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad_data = out_grad.realize_cached_data()
        input_data = node.inputs[0].realize_cached_data()
        out_grad_data = out_grad_data * (input_data > 0)
        return Tensor(out_grad_data, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return numpy.tanh(a)
        #return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (1 + -tanh(node.inputs[0])**2) * out_grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 计算输出张量的形状
        shape = args[0].shape
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        # 创建一个新的空张量以保存结果
        out = array_api.empty(new_shape, device=args[0].device)
        slices = [slice(0, s) for s in new_shape]
        for i, arr in enumerate(args):
            # 切片
            slices[self.axis] = slice(i, i + 1)
            # 把arr赋值给切片
            out[tuple(slices)] = arr
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0, s) for s in A.shape]
        splits = []
        for i in range(n):
            slices[self.axis] = slice(i, i + 1)
            splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(tuple(out_grad), self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for i in self.axes:
            new_shape[i] = new_shape[i] * (self.dilation + 1)
        slices = [slice(0, n) for n in a.shape]
        for ax in self.axes:
            slices[ax] = slice(0, new_shape[ax], self.dilation + 1)
        out = array_api.full(tuple(new_shape), 0, device=a.device)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for i in self.axes:
            new_shape[i] = int(new_shape[i] / (self.dilation + 1))
        slices = [slice(0, n) for n in a.shape]
        for ax in self.axes:
            slices[ax] = slice(0, a.shape[ax], self.dilation + 1)
        out = array_api.full(tuple(new_shape), 0, device=a.device)
        out = a[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, weight):
        ### BEGIN YOUR SOLUTION
        if self.padding != 0:
            A = A.pad(((0, 0), (self.padding, self.padding),
                       (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = weight.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K * K * C_in
        # for strides
        out_H = int((H - K + 1) / self.stride)
        out_W = int((W - K + 1) / self.stride)
        A = (A.as_strided(shape=(N, out_H, out_W, K, K, C_in),
                          strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs))
             .compact()
             .reshape((N * out_H * out_W, inner_dim)))
        out = A @ weight.compact().reshape((inner_dim, C_out))

        return out.reshape(N, out_H, out_W, C_out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        X, W = node.inputs
        K, _, _, _ = W.shape
        W_permute = transpose(flip(W, (0, 1)), (2, 3))  # 转置
        X_grad = conv(out_grad, W_permute, padding=K - 1 - self.padding)

        # W的梯度必须在批次上累积,考虑通过转置将批处理转换为通道
        out_grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))
        # 为了能对X作卷积，要把X的N也转为通道
        X_permute = transpose(X, (0, 3))

        W_grad = conv(X_permute, out_grad_permute, padding=self.padding)

        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
