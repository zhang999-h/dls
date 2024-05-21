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

### 为什么要用Normalization
来自gpt
1. 稳定训练过程：
归一化可以使训练过程更加稳定，防止梯度爆炸或梯度消失。这是因为归一化后数据的范围被控制在一定的区间内，使得训练更加平滑。
2. 加快收敛速度：
归一化能够使损失函数在更平滑的表面上优化，从而加快模型的收敛速度。这意味着模型能够在较少的迭代次数内达到较好的性能。
3. 防止过拟合：
通过归一化，模型对输入数据的微小变化不敏感，能够增强模型的泛化能力，减少过拟合的风险。
4. 使数据分布一致：
不同特征可能具有不同的尺度，归一化可以使各个特征的数值范围相似，从而避免某些特征对模型训练的影响过大。

常见的归一化方法包括：
1. 批量归一化（Batch Normalization）：
在每个小批量数据上对每一层进行归一化，通常应用在激活函数之前或之后。它通过学习缩放和平移参数来保持模型的表达能力。
2. 层归一化（Layer Normalization）：
对每一个样本在某一层的所有特征进行归一化，主要用于RNN等递归神经网络。
3. 实例归一化（Instance Normalization）：
对每一个样本独立进行归一化，常用于生成对抗网络（GAN）等图像生成任务。
4. 群归一化（Group Normalization）：
将特征分成小组，在每一小组内进行归一化，适用于小批量数据训练。

### LayerNorm1d
`needle.nn.LayerNorm1d(dim, eps=1e-5, device=None, dtype="float32")`
BatchNorm是对一个batch-size样本内的每个特征做归一化，LayerNorm是对每个样本的所有特征做归一化

weight,bias的是一维的，大小是dim(这个也参与backward吗??)

### BatchNorm1d



### Flatten
只是将输入的多维数据拉成一维的，直观上可理解为将数据“压平”
Takes in a tensor of shape **(B,X_0,X_1,...)** , 
and flattens all non-batch dimensions so that the output is of shape **(B, X_0 * X_1 * ...)**


### 为什么需要正则化
1. 防止过拟合：
在训练过程中，模型可能会过度拟合训练数据，学习到数据中的噪声和细节，导致在测试数据或新数据上的表现不佳。正则化通过在损失函数中添加惩罚项，约束模型的复杂度，从而减少过拟合现象。
2. 提高泛化能力：
通过正则化，模型在面对新数据时能够表现得更好，因为正则化技术帮助模型学会更加简洁和通用的特征，而不是仅仅记住训练数据的细节。
3. 限制模型复杂度：
正则化通过对模型参数施加限制，避免模型变得过于复杂。复杂的模型虽然在训练数据上表现很好，但在测试数据上往往表现较差。正则化有助于保持模型的简洁性和可解释性。

常见的正则化方法包括：
1. L1正则化（Lasso正则化）：
在损失函数中添加权重参数的绝对值和（即L1 范数），鼓励产生稀疏模型，使部分权重参数趋向于零，从而选择出重要的特征。
2. L2正则化（Ridge正则化）：
在损失函数中添加权重参数的平方和（即 L2 范数），防止权重参数变得过大，使得模型更加平滑和稳定。
3. Dropout：
在训练过程中，随机丢弃一部分神经元及其连接，防止神经元之间的共适应关系，从而提高模型的鲁棒性和泛化能力。
4. 数据增强：
通过对训练数据进行各种随机变换（如旋转、缩放、翻转等），生成更多的训练样本，增加模型的训练数据量，从而提高模型的泛化性能。
5. 早停（Early Stopping）：
在验证集上的性能不再提升时提前停止训练，防止模型在训练集上过度拟合。


### Dropout
生成值为0~1的随机数组,值小于p置1，反之0
```python
def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """Generate binary random Tensor"""
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) <= p
```
星号 (*) 运算符来解包元组。解包元组是指将元组中的元素**分别传递给一个函数或语句**。
```python
init.randb(*x.shape, p=1 - self.p)
```
只有在训练时才丢弃
`if self.training is True:`

### Residual
???
1. 退化问题
当神经网络变得非常深时，训练变得困难，表现为训练误差和测试误差都不再降低，甚至可能增加。这种现象被称为退化问题。原因之一是梯度消失和梯度爆炸问题，使得信息在反向传播过程中无法有效传递。

2. 残差连接的基本思想
残差网络通过引入“残差连接”来缓解这些问题。其基本思想是学习一个“残差函数”而不是直接学习输入到输出的映射。 
残差连接可以表示为：y=F(x)+x
其中， x 是输入，F(x) 是需要学习的残差函数，y 是输出。通过这种方式，即使在极端情况下F(x) 接近于零，网络也可以轻松地将输入x 传递到输出，这样就避免了梯度消失的问题。



