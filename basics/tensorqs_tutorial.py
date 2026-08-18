"""
**学习基础** ||
`快速入门 <quickstart_tutorial.html>`_ ||
**张量** ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`数据变换 <transforms_tutorial.html>`_ ||
`构建模型 <buildmodel_tutorial.html>`_ ||
`自动求导 <autogradqs_tutorial.html>`_ ||
`优化 <optimization_tutorial.html>`_ ||
`保存与加载模型 <saveloadrun_tutorial.html>`_

张量
==========================

张量是一种专门的数据结构，与数组和矩阵非常相似。
在 PyTorch 中，我们用张量来编码模型的输入和输出，以及模型的参数。

张量与 `NumPy 的 <https://numpy.org/>`_ ndarray 类似，但张量可以在 GPU 或其他硬件加速器上运行。事实上，张量和
NumPy 数组常常可以共享底层内存，从而无需复制数据（参见 :ref:`bridge-to-np-label`）。张量
也为自动微分做了优化（我们稍后会在 `自动求导 <autogradqs_tutorial.html>`__ 一节中了解更多）。如果你熟悉 ndarray，你会对
Tensor API 感到得心应手；如果不熟悉，就跟着学吧！
"""

import torch
import numpy as np


######################################################################
# 初始化张量
# ~~~~~~~~~~~~~~~~~~~~~
#
# 张量可以通过多种方式初始化。请看下面的示例：
#
# **直接从数据创建**
#
# 张量可以直接从数据创建。数据类型会自动推断。

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

######################################################################
# **从 NumPy 数组创建**
#
# 张量可以从 NumPy 数组创建（反之亦然——参见 :ref:`bridge-to-np-label`）。
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 张量转 NumPy 数组
# arr = x_np.numpy()
# 张量转 CPU 上的 NumPy 数组
# arr = x.cpu().numpy()
# 张量转 CPU 上的 NumPy 数组，并释放梯度
# arr = x.detach().numpy()
# 张量转 CPU 上的 NumPy 数组，并释放梯度，并使用 dtype 指定数据类型
# arr = x.detach().cpu().numpy().astype(np.float32)
# 张量转 CPU 上的 NumPy 数组，并释放梯度，并使用 dtype 指定数据类型，并指定形状
# arr = x.detach().cpu().numpy().astype(np.float32).reshape(2,2)
# 张量转 CPU 上的 NumPy 数组，并释放梯度，并使用 dtype 指定数据类型，并指定形状，并指定步长
# arr = x.detach().cpu().numpy().astype(np.float32).reshape(2,2).strides(2,1)
# 张量转 CPU 上的 NumPy 数组，并释放梯度，并使用 dtype 指定数据类型，并指定形状，并指定步长，并指定偏移量
# arr = x.detach().cpu().numpy().astype(np.float32).reshape(2,2).strides(2,1).offset(0)
# 张量转 CPU 上的 NumPy 数组，并释放梯度，并使用 dtype 指定数据类型，并指定形状，并指定步长，并指定偏移量，并指定步长
# arr = x.detach().cpu().numpy().astype(np.float32).reshape(2,2).strides(2,1).offset(0).step(1)


###############################################################
# **从另一个张量创建：**
#
# 新张量会保留参数张量的属性（形状、数据类型），除非显式覆盖。

x_ones = torch.ones_like(x_data) # 保留 x_data 的属性
print(f"Ones Tensor: \n {x_ones} \n")

# 创建随机向量，范围在 [0, 100) 之间
x_rand = torch.rand(x_data.shape) * 100
print(f"Random Tensor: \n {x_rand} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # 覆盖 x_data 的数据类型
print(f"Random Tensor: \n {x_rand} \n")


######################################################################
# **使用随机值或常量：**
#
# ``shape`` 是张量各维度的元组。在下面的函数中，它决定了输出张量的维度。

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



######################################################################
# --------------
#

######################################################################
# 张量的属性
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 张量属性描述了张量的形状、数据类型以及它们存储所在的设备。

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


######################################################################
# --------------
#

######################################################################
# 张量上的操作
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# 超过 1200 种张量运算，包括算术、线性代数、矩阵操作（转置、索引、切片）、采样等等，
# 在 `这里 <https://pytorch.org/docs/stable/torch.html>`__ 有全面介绍。
#
# 这些操作中的每一个都可以在 CPU 和 `加速器 <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# （如 CUDA、MPS、MTIA 或 XPU）上运行。如果你使用的是 Colab，请通过 运行时 > 更改运行时类型 > GPU 分配一个加速器。
#
# 默认情况下，张量在 CPU 上创建。我们需要使用 ``.to`` 方法
# （在检查加速器是否可用之后）显式地将张量移动到加速器上。请记住，在设备之间复制大型张量
# 在时间和内存上的开销都很大！

# 如果可用，我们将张量移动到当前加速器上
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())


######################################################################
# 试试上面列表中的一些操作吧。
# 如果你熟悉 NumPy API，你会发现 Tensor API 用起来得心应手。
#

###############################################################
# **类似 NumPy 的标准索引与切片：**

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

######################################################################
# **拼接张量** 你可以使用 ``torch.cat`` 沿给定维度拼接一组张量。
# 另请参见 `torch.stack <https://pytorch.org/docs/stable/generated/torch.stack.html>`__，
# 这是另一个拼接张量的算子，与 ``torch.cat`` 略有不同。
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
print(t1.shape) # (3, 4)

t2 = torch.stack([tensor, tensor], dim=0)
print(t2)
print(t2.shape) # (2, 3, 4)


######################################################################
# **算术运算**

# 这行代码计算两个张量之间的矩阵乘法。y1、y2、y3 的值相同
# ``tensor.T`` 返回张量的转置
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# 这行代码计算逐元素乘积。z1、z2、z3 的值相同
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


######################################################################
# **单元素张量** 如果你有一个单元素张量，例如通过将张量的所有值聚合
# 为一个值得到，你可以使用 ``item()`` 将其转换为 Python
# 数值：

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


######################################################################
# **原地操作**
# 将结果存储到操作数中的操作称为原地操作。它们以 ``_`` 后缀表示。
# 例如：``x.copy_(y)``、``x.t_()`` 会改变 ``x``。

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

######################################################################
# .. note::
#      原地操作可以节省一些内存，但在计算导数时可能会带来问题，因为会立即丢失
#      历史记录。因此，不建议使用它们。



######################################################################
# --------------
#


######################################################################
# .. _bridge-to-np-label:
#
# 与 NumPy 的桥接
# ~~~~~~~~~~~~~~~~~
# CPU 上的张量与 NumPy 数组可以共享其底层内存
# 位置，改变其中一个也会改变另一个。


######################################################################
# 张量转 NumPy 数组
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

######################################################################
# 张量的变化会反映在 NumPy 数组中。

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


######################################################################
# NumPy 数组转张量
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n = np.ones(5)
t = torch.from_numpy(n)

######################################################################
# NumPy 数组的变化会反映在张量中。
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
