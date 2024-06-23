"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = (kernel_size - 1) // 2

        shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(init.kaiming_uniform(shape=shape, device=device, dtype=dtype))
        if bias:
            scope = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(init.rand(out_channels, low=-scope, high=scope, device=device, dtype=dtype))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_permute = x.transpose((1, 2)).transpose((2, 3))
        x_conv = ops.conv(x_permute, self.weight, padding=self.padding, stride=self.stride)
        out_permute = x_conv + ops.broadcast_to(self.bias, x_conv.shape)
        out = out_permute.transpose((2, 3)).transpose((1, 2))
        return out
        ### END YOUR SOLUTION