# Homework 2

```python
class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
```
make_from_op

realize_cached_data中调用compute

def gradient(self, out_grad, node)参数是什么？？

Public repository and stub/testing code for Homework 2 of 10-714.
## Q2
### Linear
```python
 self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, 
                                            dtype=dtype).reshape((1,out_features)))
```
因为`bias`和`weight`是参数，防止更新时有计算图所以让他为参数类型`self.weight = Parameter(...)`
?为什么要先生成随机数再reshape，而不是直接用合适shape生成随机数
A:为了测试

### relu

### Sequential
Sequential 的主要用法是将多个层按顺序排列，形成一个网络模型。
example:
```python
# 创建一个包含两个线性层的简单神经网络
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)
...
# 前向传播
outputs = model(inputs)
```
### LogSumExp
#### forward
它用于稳定计算对数的和，从而避免浮点数下溢或上溢。
```python
ndl.ops.logsumexp(x, axes=axes)
```
（二维数组）第0个维度是有几行，沿着第0个轴加是把列加起来
（三维数组）第0个维度是有几个二维矩阵，沿着第0个轴加是把每个二维矩阵相同位置加起来
**总之**沿着哪个轴相加就是数组里其他维度索引不变[1..x][y][z]

若要找某个轴的max也是一样

首先要注意找最大时维度要保持，否则后面减会报错
```python
max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
sum_Z = array_api.sum(array_api.exp(Z - max_Z), axis=self.axes)
```
必须reshape,否则会触发自动广播：max_Z(1, 2, 1, 4) 的数组与形状为 log_Z(2, 4) 的数组相加后，结果数组的形状为 (1, 2, 2, 4)
```python
ans = log_Z + max_Z.reshape(log_Z.shape)
ans2 = log_Z + max_Z
```
#### gradient

求导时对于变量的形状，要先`reshape`再`broadcast_ta`防止默认广播情况出现混乱
```python
new_out_grad=out_grad.reshape(max_z.shape).broadcast_to(inputs.shape)
```
`np.array([1, 2, 3]).shape`是 `(3,)`

### SoftmaxLoss
最后记得除以样本数量，得到平均值
```python
one_hot_y = init.one_hot(logits.shape[1], y)
```
可以直接用init.one_hot()函数,其用
```python
def one_hot(self, n, i, dtype="float32"):
    return numpy.eye(n, dtype=dtype)[i]
```
`numpy.eye(n)`是创建了一个n*n的对角线数组，`[i]`是根据数组i对对角线数组进行索引排序，以生成one_hot数组
数组i的第x个数是c，则新数组的第x行就是对角线数组的第c行




