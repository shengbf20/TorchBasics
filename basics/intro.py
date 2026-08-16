"""
**学习基础** ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`数据变换 <transforms_tutorial.html>`_ ||
`构建模型 <buildmodel_tutorial.html>`_ ||
`自动求导 <autogradqs_tutorial.html>`_ ||
`优化 <optimization_tutorial.html>`_ ||
`保存与加载模型 <saveloadrun_tutorial.html>`_

学习基础
===================

作者：
`Suraj Subramanian <https://github.com/subramen>`_、
`Seth Juarez <https://github.com/sethjuarez/>`_、
`Cassie Breviu <https://github.com/cassiebreviu/>`_、
`Dmitry Soshnikov <https://soshnikov.com/>`_、
`Ari Bornstein <https://github.com/aribornstein/>`_

大多数机器学习工作流都涉及处理数据、创建模型、优化模型参数，以及保存训练好的模型。本教程将带你了解一个用 PyTorch 实现的完整机器学习工作流，并提供了深入了解其中每个概念的链接。

我们将使用 FashionMNIST 数据集来训练一个神经网络，用于预测输入图像属于以下哪个类别：T恤/上衣、裤子、套头衫、连衣裙、大衣、凉鞋、衬衫、运动鞋、包或短靴。

`本教程假定你已具备 Python 和深度学习的基础知识。`


运行教程代码
-------------------------
你可以通过以下几种方式运行本教程：

- **在云端运行**：这是最省事的入门方式！每个小节顶部都有一个"在 Google Colab 中运行"的链接，点击后会在 Google Colab 中打开一个集成了代码的 notebook，运行在完全托管的云端环境中。
- **在本地运行**：这种方式需要你先在本地机器上安装好 PyTorch 和 TorchVision（`安装说明 <https://pytorch.org/get-started/locally/>`_）。下载 notebook 或将代码复制到你喜欢的 IDE 中。


如何使用本指南
---------------------
如果你熟悉其他深度学习框架，可以先看 `0. 快速入门 <quickstart_tutorial.html>`_，
以便快速熟悉 PyTorch 的 API。

如果你是深度学习框架的新手，可以直接进入这份分步指南的第一节：`1. 张量 <tensorqs_tutorial.html>`_。


.. include:: /beginner_source/basics/qs_toc.txt

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart_tutorial
   tensorqs_tutorial
   data_tutorial
   transforms_tutorial
   buildmodel_tutorial
   autogradqs_tutorial
   optimization_tutorial
   saveloadrun_tutorial

"""
