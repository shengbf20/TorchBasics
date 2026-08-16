"""
`学习基础 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
**数据变换** ||
`构建模型 <buildmodel_tutorial.html>`_ ||
`自动求导 <autogradqs_tutorial.html>`_ ||
`优化 <optimization_tutorial.html>`_ ||
`保存与加载模型 <saveloadrun_tutorial.html>`_

数据变换
===================

数据并不总是以训练机器学习算法所需的最终处理形式出现。我们使用**数据变换**对数据进行一些处理，使其适合训练。

所有 TorchVision 数据集都有两个参数：``transform`` 用于修改特征，``target_transform`` 用于修改标签，它们接受包含变换逻辑的可调用对象。`torchvision.transforms <https://pytorch.org/vision/stable/transforms.html>`_ 模块开箱即用地提供了几种常用的变换。

FashionMNIST 的特征是 PIL Image 格式，标签是整数。为了训练，我们需要将特征转换为归一化的张量，将标签转换为独热编码（one-hot）张量。为了实现这些变换，我们使用 ``torchvision.transforms.v2`` API 以及 ``torch.nn.functional.one_hot``。
"""

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)

#################################################
# ToImage() 与 ToDtype()
# -------------------------------
#
# ``torchvision.transforms.v2`` API 用两步流水线取代了旧的 ``ToTensor`` 变换。
# `v2.ToImage <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToImage.html>`_
# 将 PIL 图像或 NumPy ``ndarray`` 转换为 ``torchvision.tv_tensors.Image`` 张量，
# `v2.ToDtype <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html>`_
# 在 ``scale=True`` 时将其转换为 ``float32``，并将像素强度值缩放到 [0., 1.] 范围内。
#

##############################################
# Lambda 变换
# -------------------------------
#
# Lambda 变换可以应用任何用户定义的 lambda 函数。这里，我们使用
# `torch.nn.functional.one_hot <https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html>`_
# 将整数标签转换为大小为 10 的独热编码张量（即数据集中标签的数量），
# 然后将其转换为 ``float`` 以匹配预期的数据类型。

target_transform = v2.Lambda(
    lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)

######################################################################
# --------------
#

#################################################################
# 延伸阅读
# ~~~~~~~~~~~~~~~~~
# - `transforms v2 入门 <https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html>`_
# - `torchvision.transforms.v2 API <https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended>`_
