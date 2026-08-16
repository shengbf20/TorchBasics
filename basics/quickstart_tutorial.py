"""
`学习基础 <intro.html>`_ ||
**快速入门** ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`数据变换 <transforms_tutorial.html>`_ ||
`构建模型 <buildmodel_tutorial.html>`_ ||
`自动求导 <autogradqs_tutorial.html>`_ ||
`优化 <optimization_tutorial.html>`_ ||
`保存与加载模型 <saveloadrun_tutorial.html>`_

快速入门
===================
本节介绍机器学习中常见任务的 API。请参考各节中的链接以深入了解。

处理数据
-----------------
PyTorch 提供两个 `处理数据的原语 <https://pytorch.org/docs/stable/data.html>`_：
``torch.utils.data.DataLoader`` 和 ``torch.utils.data.Dataset``。
``Dataset`` 存储样本及其对应的标签，``DataLoader`` 在 ``Dataset`` 之上封装了一个可迭代对象。

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

######################################################################
# PyTorch 提供了特定领域的库，如 `TorchText <https://pytorch.org/text/stable/index.html>`_、
# `TorchVision <https://pytorch.org/vision/stable/index.html>`_ 和 `TorchAudio <https://pytorch.org/audio/stable/index.html>`_，
# 这些库都包含数据集。本教程将使用 TorchVision 的数据集。
#
# ``torchvision.datasets`` 模块包含许多真实世界视觉数据（如 CIFAR、COCO，`完整列表见此处 <https://pytorch.org/vision/stable/datasets.html>`_）
# 对应的 ``Dataset`` 对象。在本教程中，我们使用 FashionMNIST 数据集。每个 TorchVision ``Dataset`` 都包含两个参数：``transform`` 和
# ``target_transform``，分别用于修改样本和标签。

# 从开放数据集下载训练数据。
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# 从开放数据集下载测试数据。
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

######################################################################
# 我们将 ``Dataset`` 作为参数传递给 ``DataLoader``。它在我们数据集之上封装了一个可迭代对象，并支持
# 自动批处理、采样、打乱和多进程数据加载。这里我们定义批量大小为 64，即 dataloader 可迭代对象中的每个元素
# 将返回一批包含 64 个特征和标签的数据。

batch_size = 64

# 创建数据加载器。
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

######################################################################
# 了解更多关于 `在 PyTorch 中加载数据 <data_tutorial.html>`_ 的信息。
#

######################################################################
# --------------
#

################################
# 创建模型
# ------------------
# 要在 PyTorch 中定义神经网络，我们创建一个继承自
# `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ 的类。我们在 ``__init__`` 函数中定义网络的层，
# 并在 ``forward`` 函数中指定数据如何通过网络。为了加速神经网络中的运算，我们将其移动到
# `加速器 <https://pytorch.org/docs/stable/torch.html#accelerators>`__（如 CUDA、MPS、MTIA 或 XPU）上。如果当前加速器可用，我们将使用它；否则，我们使用 CPU。

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

######################################################################
# 了解更多关于 `在 PyTorch 中构建神经网络 <buildmodel_tutorial.html>`_ 的信息。
#


######################################################################
# --------------
#


#####################################################################
# 优化模型参数
# ----------------------------------------
# 要训练模型，我们需要一个 `损失函数 <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# 和一个 `优化器 <https://pytorch.org/docs/stable/optim.html>`_。

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


#######################################################################
# 在单个训练循环中，模型对训练数据集（按批次输入）进行预测，并
# 反向传播预测误差以调整模型的参数。

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

##############################################################################
# 我们还会在测试数据集上检查模型的性能，以确保它在学习。

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

##############################################################################
# 训练过程会进行多次迭代（*epochs*，轮次）。在每一轮中，模型学习
# 参数以做出更好的预测。我们在每一轮打印模型的准确率和损失；我们希望看到
# 准确率随每一轮增加，损失随每一轮降低。

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

######################################################################
# 了解更多关于 `训练你的模型 <optimization_tutorial.html>`_ 的信息。
#

######################################################################
# --------------
#

######################################################################
# 保存模型
# -------------
# 保存模型的一种常用方法是序列化内部状态字典（其中包含模型参数）。

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



######################################################################
# 加载模型
# ----------------------------
#
# 加载模型的过程包括重新创建模型结构，并将
# 状态字典加载到其中。

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

#############################################################
# 现在可以使用该模型进行预测。

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


######################################################################
# 了解更多关于 `保存与加载模型 <saveloadrun_tutorial.html>`_ 的信息。
#
