"""
`学习基础 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`数据变换 <transforms_tutorial.html>`_ ||
**构建模型** ||
`自动求导 <autogradqs_tutorial.html>`_ ||
`优化 <optimization_tutorial.html>`_ ||
`保存与加载模型 <saveloadrun_tutorial.html>`_

构建神经网络
========================

神经网络由对数据执行操作的层/模块组成。`torch.nn <https://pytorch.org/docs/stable/nn.html>`_ 命名空间提供了构建你自己的神经网络所需的全部构建块。PyTorch 中的每个模块都继承自 `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_。神经网络本身也是一个模块，由其他模块（层）组成。这种嵌套结构使得构建和管理复杂架构变得很容易。

在接下来的章节中，我们将构建一个神经网络来对 FashionMNIST 数据集中的图像进行分类。

"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#############################################
# 获取用于训练的设备
# -----------------------
# 我们希望能在 `加速器 <https://pytorch.org/docs/stable/torch.html#accelerators>`__（如 CUDA、MPS、MTIA 或 XPU）上训练模型。
# 如果当前加速器可用，我们就使用它；否则，我们使用 CPU。

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

##############################################
# 定义类
# -------------------------
# 我们通过继承 ``nn.Module`` 来定义神经网络，并在 ``__init__`` 中初始化神经网络的层。
# 每个 ``nn.Module`` 子类都在 ``forward`` 方法中实现针对输入数据的操作。

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

##############################################
# 我们创建一个 ``NeuralNetwork`` 实例，将它移动到 ``device`` 上，并打印
# 它的结构。

model = NeuralNetwork().to(device)
print(model)


##############################################
# 要使用模型，我们将输入数据传给它。这会执行模型的 ``forward`` 方法，
# 以及一些 `后台操作 <https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866>`_。
# 不要直接调用 ``model.forward()``！
#
# 对输入调用模型会返回一个二维张量，其中 dim=0 对应每个类别的 10 个原始预测值的输出，dim=1 对应每个输出的各个值。
# 我们通过将输出传入 ``nn.Softmax`` 模块的实例来获得预测概率。

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


######################################################################
# --------------
#


##############################################
# 模型层
# -------------------------
#
# 让我们来拆解 FashionMNIST 模型中的各层。为了说明这一点，我们
# 将取一个包含 3 张 28x28 图像的示例小批量，观察它在
# 通过网络的过程中会发生什么。

input_image = torch.rand(3,28,28)
print(input_image.size())

##################################################
# nn.Flatten
# ^^^^^^^^^^^^^^^^^^^^^^
# 我们初始化 `nn.Flatten  <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_
# 层，将每张 2D 28x28 图像转换为一个包含 784 个像素值的连续数组（
# 小批量维度（dim=0）保持不变）。

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

##############################################
# nn.Linear
# ^^^^^^^^^^^^^^^^^^^^^^
# `线性层 <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_
# 是一个使用其存储的权重和偏置对输入应用线性变换的模块。
#
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


#################################################
# nn.ReLU
# ^^^^^^^^^^^^^^^^^^^^^^
# 非线性激活函数负责创建模型输入与输出之间的复杂映射。
# 它们在线性变换之后应用，以引入*非线性*，帮助神经网络
# 学习各种各样的现象。
#
# 在这个模型中，我们在线性层之间使用 `nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_，
# 但也有其他激活函数可以为你的模型引入非线性。

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")



#################################################
# nn.Sequential
# ^^^^^^^^^^^^^^^^^^^^^^
# `nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ 是一个有序的
# 模块容器。数据会按照定义时的相同顺序依次通过所有模块。你可以使用
# 顺序容器来快速组装一个类似 ``seq_modules`` 的网络。

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

################################################################
# nn.Softmax
# ^^^^^^^^^^^^^^^^^^^^^^
# 神经网络的最后一个线性层返回 `logits` —— [-\infty, \infty] 范围内的原始值——它们被传递给
# `nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ 模块。logits 会被缩放到
# [0, 1] 范围内的值，表示模型对每个类别的预测概率。``dim`` 参数指明了
# 各值必须求和为 1 的维度。

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


#################################################
# 模型参数
# -------------------------
# 神经网络中的许多层都是*参数化*的，即它们带有在训练期间优化的关联权重
# 和偏置。继承 ``nn.Module`` 会自动
# 跟踪模型对象内定义的所有字段，并使所有参数
# 都可以通过模型的 ``parameters()`` 或 ``named_parameters()`` 方法访问。
#
# 在本示例中，我们遍历每个参数，并打印它的大小和值的预览。
#


print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

######################################################################
# --------------
#

#################################################################
# 进一步阅读
# -----------------
# - `torch.nn API <https://pytorch.org/docs/stable/nn.html>`_
