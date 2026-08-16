"""
`学习基础 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`数据变换 <transforms_tutorial.html>`_ ||
`构建模型 <buildmodel_tutorial.html>`_ ||
`自动求导 <autogradqs_tutorial.html>`_ ||
`优化 <optimization_tutorial.html>`_ ||
**保存与加载模型**

保存与加载模型
============================

在本节中，我们将了解如何通过保存、加载以及运行模型预测来持久化模型状态。
"""

import torch
import torchvision.models as models


#######################################################################
# 保存与加载模型权重
# --------------------------------
# PyTorch 模型将学习到的参数存储在一个内部的
# 状态字典中，称为 ``state_dict``。这些参数可以通过 ``torch.save``
# 方法持久化保存：

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

##########################
# 要加载模型权重，你需要先创建同一个模型的实例，然后使用 ``load_state_dict()`` 方法加载参数。
#
# 在下面的代码中，我们设置 ``weights_only=True`` 来限制
# 反序列化（unpickling）期间执行的函数，仅保留加载权重所需的函数。
# 使用 ``weights_only=True`` 被认为是加载权重时的最佳实践。

model = models.vgg16() # 我们不指定 ``weights``，即创建未经训练的模型
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

###########################
# .. note:: 在推理之前，请务必调用 ``model.eval()`` 方法，将 dropout 和批归一化层设置为评估模式。否则会产生不一致的推理结果。

#######################################################################
# 保存和加载带结构的模型
# -------------------------------------
# 加载模型权重时，我们需要先实例化模型类，因为该类
# 定义了网络的结构。我们可能希望将类的结构连同
# 模型一起保存，此时可以将 ``model``（而不是 ``model.state_dict()``）传给保存函数：

torch.save(model, 'model.pth')

########################
# 然后，我们可以像下面演示的那样加载模型。
#
# 正如 `保存和加载 torch.nn.Modules <https://pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules>`_ 中所述，
# 保存 ``state_dict`` 被认为是最佳实践。然而，
# 下面我们使用 ``weights_only=False``，因为这涉及加载
# 模型本身，这是 ``torch.save`` 的一个遗留用例。

model = torch.load('model.pth', weights_only=False)

########################
# .. note:: 这种方法在序列化模型时使用 Python 的 `pickle <https://docs.python.org/3/library/pickle.html>`_ 模块，因此在加载模型时需要实际的类定义可用。

#######################
# 相关教程
# -----------------
# - `在 PyTorch 中保存和加载通用检查点 <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`_
# - `从检查点加载 nn.Module 的技巧 <https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html?highlight=loading%20nn%20module%20from%20checkpoint>`_
