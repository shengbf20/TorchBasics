"""
`学习基础 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`数据变换 <transforms_tutorial.html>`_ ||
`构建模型 <buildmodel_tutorial.html>`_ ||
**自动求导** ||
`优化 <optimization_tutorial.html>`_ ||
`保存与加载模型 <saveloadrun_tutorial.html>`_

使用 ``torch.autograd`` 进行自动微分
=================================================

在训练神经网络时，最常用的算法是**反向传播**。在该算法中，参数（模型权重）根据损失函数相对于给定参数的**梯度**进行调整。

为了计算这些梯度，PyTorch 内置了一个称为 ``torch.autograd`` 的微分引擎。它支持对任意计算图自动计算梯度。

考虑最简单的单层神经网络，其输入为 ``x``，参数为 ``w`` 和 ``b``，以及某个损失函数。它可以用 PyTorch 按以下方式定义：
"""

import torch

x = torch.ones(5)  # 输入张量
y = torch.zeros(3)  # 期望输出
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


######################################################################
# 张量、函数与计算图
# ------------------------------------------
#
# 这段代码定义了如下**计算图**：
#
# .. figure:: /_static/img/basics/comp-graph.png
#    :alt:
#
# 在这个网络中，``w`` 和 ``b`` 是**参数**，我们需要对它们进行优化。
# 因此，我们需要能够计算损失函数相对于这些变量的梯度。为此，我们设置了
# 这些张量的 ``requires_grad`` 属性。

#######################################################################
# .. note:: 你可以在创建张量时设置 ``requires_grad`` 的值，
#           也可以稍后通过 ``x.requires_grad_(True)`` 方法进行设置。

#######################################################################
# 我们应用于张量以构建计算图的函数实际上是 ``Function`` 类的对象。该对象知道如何
# 在*前向*方向上计算函数，也知道如何在*反向传播*步骤中计算其导数。
# 反向传播函数的引用存储在张量的 ``grad_fn`` 属性中。
# 你可以在`文档 <https://pytorch.org/docs/stable/autograd.html#function>`__ 中
# 找到有关 ``Function`` 的更多信息。
#

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

######################################################################
# 计算梯度
# -------------------
#
# 为了优化神经网络中参数的权重，我们需要计算损失函数相对于参数的导数，
# 即在 ``x`` 和 ``y`` 取某些固定值时，我们需要 :math:`\frac{\partial loss}{\partial w}` 和
# :math:`\frac{\partial loss}{\partial b}`。为了计算这些导数，我们调用
# ``loss.backward()``，然后从 ``w.grad`` 和 ``b.grad`` 中获取值：
#

loss.backward()
print(w.grad)
print(b.grad)


######################################################################
# .. note::
#   - 我们只能获取计算图叶节点的 ``grad`` 属性，这些叶节点的 ``requires_grad`` 属性
#     被设置为 ``True``。对于图中的所有其他节点，梯度将不可用。
#   - 出于性能原因，对于给定的图，我们只能使用 ``backward`` 进行一次梯度计算。
#     如果需要对同一图进行多次 ``backward`` 调用，我们需要向 ``backward`` 调用传递
#     ``retain_graph=True``。
#


######################################################################
# 禁用梯度跟踪
# ---------------------------
#
# 默认情况下，所有 ``requires_grad=True`` 的张量都在跟踪它们的
# 计算历史并支持梯度计算。然而，在某些情况下我们不需要这样做，例如，当我们已经
# 训练好模型，只想将其应用于某些输入数据时，即我们只想通过网络进行*前向*计算。
# 我们可以通过将计算代码包裹在 ``torch.no_grad()`` 块中来停止跟踪计算：
#

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)


######################################################################
# 另一种达到同样效果的方法是使用张量的 ``detach()`` 方法：
#

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

######################################################################
# 你可能想要禁用梯度跟踪的原因有：
#   - 将神经网络中的某些参数标记为**冻结参数**。
#   - 当只进行前向传播时**加快计算速度**，因为对不跟踪梯度的张量进行计算会更高效。


######################################################################

######################################################################
# 关于计算图的更多内容
# ----------------------------
# 从概念上讲，autograd 在一个由
# `Function <https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function>`__
# 对象组成的有向无环图（DAG）中记录数据（张量）和所有已执行的操作（以及由此产生的新张量）。
# 在这个 DAG 中，叶节点是输入张量，根节点是输出张量。
# 通过从根到叶追踪该图，你可以利用链式法则自动计算梯度。
#
# 在前向传播中，autograd 同时做两件事：
#
# - 运行请求的操作以计算结果张量
# - 在 DAG 中维护该操作的*梯度函数*。
#
# 当在 DAG 的根节点上调用 ``.backward()`` 时，反向传播开始。随后 ``autograd``：
#
# - 从每个 ``.grad_fn`` 计算梯度，
# - 将它们累加到相应张量的 ``.grad`` 属性中
# - 利用链式法则，一直传播到叶张量。
#
# .. note::
#   **PyTorch 中的 DAG 是动态的**
#   一个重要的注意事项是，图是每次从头重新创建的；在每次
#   ``.backward()`` 调用之后，autograd 都会开始填充一个新图。这正是
#   允许你在模型中使用控制流语句的原因；
#   如果需要，你可以在每次迭代中改变形状、大小和操作。

######################################################################
# 可选阅读：张量梯度与雅可比积
# --------------------------------------------------------
#
# 在许多情况下，我们有一个标量损失函数，需要计算相对于某些参数的梯度。
# 然而，在某些情况下，输出函数是一个任意张量。在这种情况下，PyTorch
# 允许你计算所谓的**雅可比积**，而不是实际的梯度。
#
# 对于向量函数 :math:`\vec{y}=f(\vec{x})`，其中
# :math:`\vec{x}=\langle x_1,\dots,x_n\rangle`，
# :math:`\vec{y}=\langle y_1,\dots,y_m\rangle`，:math:`\vec{y}` 相对于
# :math:`\vec{x}` 的梯度由**雅可比矩阵**给出：
#
# .. math::
#
#
#    J=\left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)
#
# PyTorch 允许你不直接计算雅可比矩阵本身，而是针对给定的输入向量
# :math:`v=(v_1 \dots v_m)` 计算**雅可比积** :math:`v^T\cdot J`。
# 这可以通过将 :math:`v` 作为参数调用 ``backward`` 来实现。:math:`v` 的大小应与
# 我们想要计算乘积的原始张量的大小相同：
#

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")


######################################################################
# 请注意，当我们用相同的参数第二次调用 ``backward`` 时，梯度的值是不同的。
# 这是因为在进行 ``backward`` 传播时，PyTorch **会累加梯度**，
# 即计算出的梯度值会被加到计算图所有叶节点的 ``grad`` 属性中。
# 如果你想计算正确的梯度，需要事先将 ``grad`` 属性清零。
# 在真实的训练中，*优化器*会帮助我们完成这一步。

######################################################################
# .. note:: 之前我们调用 ``backward()`` 函数时没有传递参数。
#           这实际上等同于调用 ``backward(torch.tensor(1.0))``，
#           这是在标量值函数（例如神经网络训练期间的损失）情况下计算梯度的有用方法。
#

######################################################################
# --------------
#

#################################################################
# 延伸阅读
# ~~~~~~~~~~~~~~~~~
# - `自动求导机制 <https://pytorch.org/docs/stable/notes/autograd.html>`_
