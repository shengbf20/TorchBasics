import torch
import numpy as np

from torchvision import datasets
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



# 加载 data

print("Downloading training data")

training_data = datasets.FashionMNIST(
    root="data", # 训练/测试数据的存储路径
    train=True, # 指定训练数据集(True)或测试数据集(False)
    download=True, # 如果root目录下没有数据，则从互联网下载
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) # 预处理：将数据转换为张量
)

# transform 的作用：输入是刚解码的图像张量，输出是 float32 图像张量，像素值被压缩到 [0, 1] 之间，适合神经网络处理
# v2.Compose() 按顺序将多个变换串起来
# v2.ToImage() 将数据转换为图像
# v2.ToDtype(torch.float32, scale=True) 将数据转换为浮点数
# scale=True 将数据缩放到 [0, 1] 之间

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

# print(training_data)
# print(test_data)

print("--------------------------------")
img, label = training_data[0]   # 取一条
print(img.shape, label)         # 例如 torch.Size([1, 28, 28]) 和类别数字
print(len(training_data))              # 60000
# print(training_data[0])
print("--------------------------------")

# 绘制 3x3 的随机样本拼图
figure = plt.figure(figsize=(8, 8)) # 创建一个 8x8 的画布
cols, rows = 3, 3 # 子图网格：3行3列
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # 在0-59999之间随机取一个整数下标
    img, label = training_data[sample_idx] # 取对应下标的图像和标签
    figure.add_subplot(rows, cols, i) # 添加子图
    plt.title(label)
    plt.axis("off") # 关闭坐标轴，只看图
    plt.imshow(img.squeeze(), cmap="gray") # 显示图像，cmap="gray" 表示灰度图

plt.savefig("fashion_mnist.png", bbox_inches="tight", dpi=300) # 保存图像
plt.close(figure)

print("--------------------------------")

# 使用 DataLoader 为训练准备数据

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True) # shuffle=True 表示每个 epoch 重新打乱数据
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 显示图像和标签：
training_features, training_labels = next(iter(train_dataloader)) # 取一个批次的数据
print(f"Feature batch shape: {training_features.size()}")
print(f"Labels batch shape: {training_labels.size()}")
img = training_features[0].squeeze() # 展示其中第一张图像
label = training_labels[0].item() # 展示其中第一张图像的标签
plt.imshow(img.squeeze(), cmap="gray")
plt.title(label) # 标题为标签
plt.axis("off") # 关闭坐标轴，只看图
plt.savefig("example.png", bbox_inches="tight", dpi=300) # 保存图像
plt.close(img)


'''
# 遍历 DataLoader：
for batch in train_dataloader:
    print(batch) # 返回一个元组，第一个元素是图像张量，第二个元素是标签张量
    break
'''

print("--------------------------------")