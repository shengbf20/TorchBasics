"""
`学习基础 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`数据变换 <transforms_tutorial.html>`_ ||
`构建模型 <buildmodel_tutorial.html>`_ ||
`自动求导 <autogradqs_tutorial.html>`_ ||
**优化** ||
`保存与加载模型 <saveloadrun_tutorial.html>`_

优化模型参数
===========================

现在我们已经有了模型和数据，是时候通过在我们的数据上优化模型参数来训练、验证和测试模型了。训练模型是一个迭代的过程；在每次迭代中，模型会对输出做出一个猜测，计算
其猜测的误差（*损失*），收集误差相对于其参数的导数（正如我们在
`上一节 <autogradqs_tutorial.html>`_ 中看到的那样），并使用梯度下降来**优化**这些参数。想更详细地了解这一过程，
请观看 3Blue1Brown 关于`反向传播 <https://www.youtube.com/watch?v=tIeHLnjs5U8>`__ 的视频。

前置代码
-----------------
我们加载之前章节中关于`数据集与数据加载器 <data_tutorial.html>`_
和`构建模型 <buildmodel_tutorial.html>`_ 的代码。
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

model = NeuralNetwork()


##############################################
# 超参数
# -----------------
#
# 超参数是可调整的参数，让你能够控制模型优化的过程。
# 不同的超参数取值会影响模型训练和收敛速度
# （关于超参数调优，请`阅读更多 <https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>`__）
#
# 我们为训练定义以下超参数：
#
# - **轮数** - 在数据集上迭代的次数
# - **批大小** - 在参数更新之前通过网络传播的数据样本数量
# - **学习率** - 每个批次/轮次中更新模型参数的程度。较小的值会导致学习速度缓慢，而较大的值可能导致训练过程中出现不可预测的行为。
#

learning_rate = 1e-3
batch_size = 64
epochs = 5



#####################################
# 优化循环
# -----------------
#
# 设置好超参数后，我们就可以通过优化循环来训练和优化模型。优化循环的
# 每一次迭代称为一个**轮次（epoch）**。
#
# 每个轮次由两个主要部分组成：
#
# - **训练循环** - 遍历训练数据集，尝试收敛到最优参数。
# - **验证/测试循环** - 遍历测试数据集，检查模型性能是否在提升。
#
# 让我们先简单了解一下训练循环中使用的一些概念。可以直接跳到后面
# 查看优化循环的 :ref:`full-impl-label`。
#
# 损失函数
# ~~~~~~~~~~~~~~~~~
#
# 当输入某些训练数据时，我们未经训练的网络很可能无法给出正确的
# 答案。**损失函数**衡量得到的结果与目标值之间的差异程度，
# 而它正是我们在训练中想要最小化的对象。为了计算损失，我们使用
# 给定数据样本的输入做出预测，并将其与真实的数据标签值进行比较。
#
# 常见的损失函数包括用于回归任务的 `nn.MSELoss <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`_（均方误差）和用于分类的
# `nn.NLLLoss <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss>`_（负对数似然）。
# `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss>`_ 结合了 ``nn.LogSoftmax`` 和 ``nn.NLLLoss``。
#
# 我们将模型输出的 logits 传给 ``nn.CrossEntropyLoss``，它会归一化 logits 并计算预测误差。

# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()



#####################################
# 优化器
# ~~~~~~~~~~~~~~~~~
#
# 优化是在每个训练步骤中调整模型参数以降低模型误差的过程。**优化算法**定义了这一过程如何执行（在本例中我们使用随机梯度下降）。
# 所有优化逻辑都封装在 ``optimizer`` 对象中。这里我们使用 SGD 优化器；此外，PyTorch 中还提供了许多`不同的优化器 <https://pytorch.org/docs/stable/optim.html>`_，
# 例如 ADAM 和 RMSProp，它们对不同类型的模型和数据效果更好。
#
# 我们通过注册需要训练的模型参数并传入学习率超参数来初始化优化器。

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#####################################
# 在训练循环内部，优化分三个步骤进行：
#
# - 调用 ``optimizer.zero_grad()`` 重置模型参数的梯度。梯度默认会累加；为避免重复计数，我们在每次迭代中显式地将它们清零。
# - 通过调用 ``loss.backward()`` 反向传播预测损失。PyTorch 会计算出损失关于每个参数的梯度。
# - 得到梯度后，我们调用 ``optimizer.step()``，根据反向传播过程中收集到的梯度来调整参数。


########################################
# .. _full-impl-label:
#
# 完整实现
# -----------------------
# 我们定义了 ``train_loop`` 来循环执行我们的优化代码，以及 ``test_loop`` 来
# 评估模型在测试数据上的性能。

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 将模型设置为训练模式 - 对批归一化和 dropout 层很重要
    # 本例中并非必需，但为了遵循最佳实践而加上
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测和损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # 将模型设置为评估模式 - 对批归一化和 dropout 层很重要
    # 本例中并非必需，但为了遵循最佳实践而加上
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 使用 torch.no_grad() 评估模型可确保在测试模式下不计算梯度，
    # 同时也有助于减少不必要的梯度计算以及 requires_grad=True 张量的内存占用
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


########################################
# 我们初始化损失函数和优化器，并将它们传给 ``train_loop`` 和 ``test_loop``。
# 你可以随意增加轮数，以跟踪模型性能的提升。

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")



#################################################################
# 延伸阅读
# -----------------------
# - `损失函数 <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# - `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
# - `模型的热启动训练 <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_
#
