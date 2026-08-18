"""
`学习基础 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
**数据集与数据加载器** ||
`数据变换 <transforms_tutorial.html>`_ ||
`构建模型 <buildmodel_tutorial.html>`_ ||
`自动求导 <autogradqs_tutorial.html>`_ ||
`优化 <optimization_tutorial.html>`_ ||
`保存与加载模型 <saveloadrun_tutorial.html>`_

数据集与数据加载器
======================

"""

#################################################################
# 处理数据样本的代码可能会变得杂乱且难以维护；理想情况下，我们希望数据集代码与模型训练代码解耦，以提高可读性和模块化程度。
# PyTorch 提供了两种数据原语：``torch.utils.data.DataLoader`` 和 ``torch.utils.data.Dataset``，
# 它们允许你使用预加载的数据集以及你自己的数据。
# ``Dataset`` 存储样本及其对应的标签，``DataLoader`` 在 ``Dataset`` 之上封装了一个可迭代对象，以便轻松访问样本。
#
# PyTorch 领域库提供了许多预加载的数据集（如 FashionMNIST），
# 它们继承自 ``torch.utils.data.Dataset``，并实现了针对特定数据的功能。
# 它们可用于构建模型原型并进行基准测试。你可以在这里找到它们：
# `图像数据集 <https://pytorch.org/vision/stable/datasets.html>`_、
# `文本数据集 <https://pytorch.org/text/stable/datasets.html>`_ 和
# `音频数据集 <https://pytorch.org/audio/stable/datasets.html>`_
#

############################################################
# 加载数据集
# -------------------
#
# 下面是一个如何从 TorchVision 加载 `Fashion-MNIST <https://research.zalando.com/project/fashion_mnist/fashion_mnist/>`_ 数据集的示例。
# Fashion-MNIST 是 Zalando 商品图像的集合，包含 60,000 个训练样本和 10,000 个测试样本。
# 每个样本包含一张 28×28 的灰度图像，以及来自 10 个类别之一的对应标签。
#
# 我们使用以下参数加载 `FashionMNIST 数据集 <https://pytorch.org/vision/stable/datasets.html#fashion-mnist>`_：
#  - ``root`` 是训练/测试数据的存储路径，
#  - ``train`` 指定训练数据集或测试数据集，
#  - ``download=True`` 表示如果 ``root`` 下没有数据，则从互联网下载。
#  - ``transform`` 和 ``target_transform`` 指定特征和标签的变换


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt


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


#################################################################
# 遍历并可视化数据集
# -------------------------------------
#
# 我们可以像索引列表一样手动索引 ``Datasets``：``training_data[index]``。
# 我们使用 ``matplotlib`` 来可视化训练数据中的一些样本。

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#################################################################
# ..
#  .. figure:: /_static/img/basics/fashion_mnist.png
#    :alt: fashion_mnist


######################################################################
# --------------
#

#################################################################
# 为你的文件创建自定义数据集
# ---------------------------------------------------
#
# 自定义 Dataset 类必须实现三个函数：`__init__`、`__len__` 和 `__getitem__`。
# 请看下面的实现；FashionMNIST 图像存储在目录 ``img_dir`` 中，其标签单独存储在 CSV 文件 ``annotations_file`` 中。
#
# 在接下来的章节中，我们将逐一分析这些函数的作用。


import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # 读取标签文件
        self.img_dir = img_dir # 图像目录
        self.transform = transform # 对图像的变换函数
        self.target_transform = target_transform # 对标签的变换函数

    def __len__(self):
        return len(self.img_labels) # 返回数据集的长度

    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # 获取图像路径
        image = decode_image(img_path) # 来自 torchvision.io：根据文件路径读盘上的图片（jpg/png 等），解码成 torch.Tensor
        label = self.img_labels.iloc[idx, 1] # 获取标签
        if self.transform:
            image = self.transform(image) # 应用图像变换，输出仍是图像相关张量，但已按训练需要改过
        if self.target_transform:
            label = self.target_transform(label) # 应用标签变换，输出仍是标签相关张量，但已按训练需要改过
        return image, label # 返回图像和标签


#################################################################
# ``__init__``
# ^^^^^^^^^^^^^^^^^^^^
#
# 实例化 Dataset 对象时，__init__ 函数只运行一次。我们在此初始化包含图像的目录、标注文件以及两种变换（下一节会详细介绍）。
#
# labels.csv 文件的内容如下： ::
#
#     tshirt1.jpg, 0
#     tshirt2.jpg, 0
#     ......
#     ankleboot999.jpg, 9


def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform


#################################################################
# ``__len__``
# ^^^^^^^^^^^^^^^^^^^^
#
# __len__ 函数返回数据集中样本的数量。
#
# 示例：


def __len__(self):
    return len(self.img_labels)


#################################################################
# ``__getitem__``
# ^^^^^^^^^^^^^^^^^^^^
#
# __getitem__ 函数加载并返回数据集中给定索引 ``idx`` 处的样本。
# 它根据索引确定图像在磁盘上的位置，使用 ``decode_image`` 将其转换为张量，
# 从 ``self.img_labels`` 的 csv 数据中获取对应的标签，对它们调用变换函数（如果适用），
# 并以元组形式返回张量图像和对应标签。

def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = decode_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label


######################################################################
# --------------
#


#################################################################
# 使用 DataLoader 为训练准备数据
# -------------------------------------------------
# ``Dataset`` 每次检索数据集中的一个样本的特征和标签。在训练模型时，
# 我们通常希望以小批量（"minibatches"）的方式传递样本，在每个 epoch 重新打乱数据以减少模型过拟合，
# 并使用 Python 的 ``multiprocessing`` 来加速数据检索。

# ``DataLoader`` 是一个可迭代对象，通过简单的 API 为我们抽象了这种复杂性。

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

###########################
# 遍历 DataLoader
# -------------------------------
#
# 我们已经将该数据集加载到 ``DataLoader`` 中，可以按需遍历数据集。
# 下面的每次迭代都会返回一批 ``train_features`` 和 ``train_labels``（分别包含 ``batch_size=64`` 个特征和标签）。
# 由于我们指定了 ``shuffle=True``，在遍历完所有批次后数据会被重新打乱（如需更细粒度地控制
# 数据加载顺序，请参阅 `采样器 <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_）。

# 显示图像和标签：
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

######################################################################
# --------------
#

#################################################################
# 进一步阅读
# ----------------
# - `torch.utils.data API <https://pytorch.org/docs/stable/data.html>`_
