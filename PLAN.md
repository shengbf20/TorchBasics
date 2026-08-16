# 🚀 PyTorch 通关计划（3–4 天）

> **把学 PyTorch 当成打游戏**：每个版块是一关，每关都有「学什么 / 看什么 / 做什么」，
> 做完小任务、留下运行证据才算通关。
>
> - 淡化时间：下面只给「预计用时」做参考，按自己的节奏走，快慢都行；
> - 主线剧情：全程以 **MNIST 手写数字**为"主角"，一个数据集贯穿所有关卡，
> 学完你就拥有一整套能跑通的深度学习代码库；
> - 标准版总用时 **≈ 4 天**；赶时间可走文末「极速版 3 天」路线。

---

## 🗺️ 通关地图（总览）


| 关卡  | 版块                   | 预计用时  | 交付物（通关凭证）                |
| --- | -------------------- | ----- | ------------------------ |
| 第0关 | 环境搭建                 | 0.5 天 | `env_check.py` 运行截图      |
| 第1关 | Tensor 张量            | 0.5 天 | 张量游乐场 notebook（含趣味实验）    |
| 第2关 | autograd 自动求导        | 0.5 天 | 手写线性回归对比 notebook        |
| 第3关 | torch.nn 模型搭建        | 1 天   | MLP + LeNet 两段可运行代码      |
| 第4关 | torch.optim 优化器      | 0.5 天 | 优化器对比 loss 曲线图           |
| 第5关 | Dataset / DataLoader | 0.5 天 | 自定义 Dataset + 增强九宫格图     |
| 第6关 | 训练流程与工程化             | 0.5 天 | 可复用 `train.py` 模板        |
| 第7关 | 🐉 BOSS 战：综合实战       | 1 天   | 端到端项目 + README + 推理 demo |
| 第8关 | 进阶彩蛋（可选）             | 弹性    | 任选一项进阶实验                 |




## 📁 建议目录结构

```
PyTorch/
├── PLAN.md          # 本计划
├── README.md        # 「成果墙」：贴每关证据截图（也是最终项目入口）
├── data/            # 数据集缓存
├── 01_tensor/       # 第1关
├── 02_autograd/     # 第2关
├── 03_nn/           # 第3关
├── 04_optim/        # 第4关
├── 05_data/         # 第5关
├── 06_training/     # 第6关
└── 07_project/      # 第7关 BOSS 战
```



## 🎮 玩法说明

1. **按顺序闯关**。每关先看「学什么」定目标，再看「看什么」——视频和文档各挑 **1 个** 就够，资料是"自助餐"不是"必吃清单"；
2. **独立完成「做什么」**。交付任务不许抄参考代码，写不出来 = 没过关，回头补；
3. **通关标准**：自测题能答上来 ✅ + 交付物能跑出结果 ✅；
4. **留证据**：每关通关后在成果墙 `README.md` 贴一张运行截图（loss 曲线 / 模型结构 / 预测图都行），集齐 8 枚"徽章" 🏅 即通关毕业。

---



## 关卡详情



### 第0关 环境搭建：开机！⚙️

**学什么**：Python + PyTorch 安装（CPU 即可，有 N 卡可装 CUDA 版）、Jupyter Notebook / VSCode、GPU 检测与设备搬运（`.to(device)`）。
**看什么**：

- 官方安装向导：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- 官方 Quickstart（10 分钟感受全流程）：[https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- 土堆《PyTorch 快速入门》前几集（环境 + 张量入门）：[https://www.bilibili.com/video/BV1hE411t7NQ](https://www.bilibili.com/video/BV1hE411t7NQ)
**做什么**：

1. 安装 PyTorch，跑通 `import torch`；
2. 写 `env_check.py`：打印 `torch.__version__`、`torch.cuda.is_available()`、设备信息；
3. 建好上面的目录结构。

**自测**：CPU/GPU 张量怎么互搬？`.to(device)` 做了什么？没有 GPU 会影响学习吗？

### 第1关 张量 Tensor：数字积木 🧱

**学什么**：创建（`tensor`/`zeros`/`randn`/`arange`）、数据类型与转换、索引切片、形状操作（`view`/`reshape`/`permute`/`squeeze`/`unsqueeze`）、广播、与 NumPy 互转、随机种子、设备。
**看什么**：

- 官方 60 分钟入门（Tensor 部分）：[https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- Learning PyTorch with Examples（保姆级带代码）：[https://pytorch.org/tutorials/beginner/pytorch_with_examples.html](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
- Datawhale《深入浅出 PyTorch》第2章：[https://github.com/datawhalechina/thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch)
- Tensor API 速查：[https://pytorch.org/docs/stable/tensors.html](https://pytorch.org/docs/stable/tensors.html)
**做什么**（`01_tensor/tensor_playground.ipynb`）：

1. 从零手写一个 `softmax`（用张量运算实现，不调 `torch.nn.functional`）；
2. 用张量做一次 **Monte Carlo 算 π** 趣味实验（随机撒点 + 统计比例）；
3. 用 MNIST 数据集（`torchvision.datasets.MNIST`）加载一张图，完成翻转/裁剪/像素统计等小操作。

**自测**：tensor 和 numpy 数组的区别与互转？广播是什么，举个例？`view` 和 `permute` 的区别？怎么固定随机种子？

### 第2关 autograd 自动求导：计算图记账本 📒

**学什么**：计算图、`requires_grad`、`backward()`、`.grad`、`.detach()`、`torch.no_grad()`、`zero_grad()` 的意义。
**看什么**：

- 官方《A Gentle Introduction to torch.autograd》：[https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- 官方文档：[https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html)
- 李沐《动手学深度学习》(PyTorch 版) 线性回归章节，配合视频食用：[https://zh.d2l.ai/](https://zh.d2l.ai/) ｜ 李沐 B站：[https://space.bilibili.com/1567748478](https://space.bilibili.com/1567748478)
**做什么**（`02_autograd/linear_regression_autograd.ipynb`）：

1. 用 autograd 训练一个最简单的线性回归（`y = wx + b`）拟合人造数据；
2. 把学到的 `w, b` 和解析解（最小二乘闭式解）对比，误差应接近 0；
3. 故意**忘了** `zero_grad()` 观察梯度累积现象，再解释原因。

**自测**：`backward()` 后梯度存在哪里？为什么每步要 `zero_grad()`？`torch.no_grad()` 什么时候用？

### 第3关 torch.nn 模型搭建：搭积木套件 🧩

**学什么**：`nn.Module`（`__init__` + `forward`）、常用层（`Linear`/`Conv2d`/`ReLU`/`Dropout`/`BatchNorm`）、`nn.Sequential`、`nn.functional`、损失函数（`CrossEntropyLoss`/`MSELoss`）。
**看什么**：

- 官方《What is torch.nn really?》（边写边学，强烈推荐）：[https://pytorch.org/tutorials/beginner/nn_tutorial.html](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
- 官方 nn 文档（查层用）：[https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)
- 土堆视频的「nn.Module 与网络搭建」部分：[https://www.bilibili.com/video/BV1hE411t7NQ](https://www.bilibili.com/video/BV1hE411t7NQ)
- 模型结构可视化神器 torchinfo：[https://github.com/TylerYep/torchinfo](https://github.com/TylerYep/torchinfo)
**做什么**（`03_nn/`）：

1. 用 `nn.Module` 搭一个 **MLP**（2 个隐层）分类 MNIST；
2. 复刻经典 **LeNet-5**（卷积网络）分类 MNIST；
3. 用 `torchinfo.summary()` 打印两个模型的结构与参数量，截图留档。

**自测**：`__init__` 和 `forward` 各干什么？`model(x)` 与 `model.forward(x)` 有区别吗？训练/评估模式（`.train()`/`.eval()`）差异在哪？

### 第4关 torch.optim 优化器：方向盘与油门 🎛️

**学什么**：`SGD`（含 momentum）、`Adam`、`AdamW`、`RMSprop`、`weight_decay`、学习率调度 `lr_scheduler`（`StepLR`/`CosineAnnealingLR`）。
**看什么**：

- 官方 optim 文档：[https://pytorch.org/docs/stable/optim.html](https://pytorch.org/docs/stable/optim.html)
- 优化器原理与 Scheduler 实战指南（中文博客）：[https://blog.csdn.net/m0_56086190/article/details/156224313](https://blog.csdn.net/m0_56086190/article/details/156224313)
- 优化器深潜（SGD→Adam→AdamW，英文）：[https://engineersofai.com/docs/math-for-ai/calculus-and-optimization/optimization-algorithms-deep-dive](https://engineersofai.com/docs/math-for-ai/calculus-and-optimization/optimization-algorithms-deep-dive)
**做什么**（`04_optim/optimizer_compare.ipynb`）：

1. 用**同一个 MLP + 同一份 MNIST 数据**，分别用 SGD、SGD+Momentum、Adam 训练；
2. 画三条 loss 下降曲线对比，写 2 句话点评差异；
3. 给 Adam 加一个 `StepLR` 学习率衰减，观察曲线变化。

**自测**：SGD 和 Adam 的核心区别？`weight_decay` 的作用？`lr_scheduler` 解决什么问题？

### 第5关 Dataset / DataLoader：食堂打饭流水线 🍚

**学什么**：`torchvision.datasets`、`transforms`（归一化/随机翻转/裁剪等数据增强）、自定义 `Dataset`（`__len__` + `__getitem__`）、`DataLoader` 参数（`batch_size`/`shuffle`/`num_workers`/`drop_last`）。
**看什么**：

- 官方《Writing Custom Datasets, DataLoaders and Transforms》：[https://pytorch.org/tutorials/beginner/data_loading_tutorial.html](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- Roboflow《Build a PyTorch Custom Dataset in Minutes》：[https://blog.roboflow.com/pytorch-custom-dataset/](https://blog.roboflow.com/pytorch-custom-dataset/)
- torchvision 文档（查 transforms 用）：[https://pytorch.org/vision/stable/](https://pytorch.org/vision/stable/)
**做什么**（`05_data/`）：

1. 写一个自定义 `Dataset`：从文件夹/CSV 读自己的数据（没有现成数据就用 MNIST 图片模拟）；
2. 用 `transforms` 做 4 种数据增强，拼一张 **9 宫格对比图**（原图 vs 增强后）；
3. 实验 `batch_size` 和 `shuffle` 对训练数据流的影响（打印几个 batch 的 shape 与标签分布）。

**自测**：`Dataset` 和 `DataLoader` 各负责什么？`__len__`/`__getitem__` 返回什么？`num_workers` 干嘛用的？

### 第6关 训练流程与工程化：从"能跑"到"好用" 🔧

**学什么**：标准 train/val/test 循环、tqdm 进度条、评估指标（accuracy）、`state_dict` 保存/加载、`torch.save/load`、TensorBoard 可视化、随机种子固定（可复现实验）。
**看什么**：

- 官方 PyTorch Recipes（保存加载 / TensorBoard 等实用配方）：[https://pytorch.org/tutorials/recipes/recipes/](https://pytorch.org/tutorials/recipes/recipes/)
- Sebastian Raschka《PyTorch in One Hour》（1 小时从张量到多卡训练，英文）：[https://sebastianraschka.com/teaching/pytorch-1h/](https://sebastianraschka.com/teaching/pytorch-1h/)
- 官方基础教程之 Train a Model（Basics 系列）：[https://pytorch.org/tutorials/beginner/basics/trainmodel_tutorial.html](https://pytorch.org/tutorials/beginner/basics/trainmodel_tutorial.html)
**做什么**（`06_training/train.py`）：

1. 写一个**可复用**的训练脚本：命令行参数（`--epochs`/`--lr`/`--batch_size`/`--device`）、训练循环、验证循环、每个 epoch 打印 loss/acc；
2. 加入模型保存（`state_dict`）与加载验证；
3. 用 TensorBoard 记录 loss 曲线并截图。

**自测**：`train()` / `eval()` / `no_grad()` 三者的关系？保存整个模型和只存 `state_dict` 的区别？固定随机种子"三件套"是什么？

### 第7关 🐉 BOSS 战：综合实战（毕业设计）

**学什么**：把 1–6 关全部串起来，完成一个端到端项目。
**看什么**（三选一，按兴趣）：

- 方案 A（CV 入门）：官方《Training a Classifier》用 CIFAR-10 完整走一遍：[https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- 方案 B（迁移学习）：用预训练 ResNet 做猫狗/花朵分类（数据集用 torchvision 或 Kaggle）：[https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- 方案 C（NLP 入门）：用 `nn.Embedding` + LSTM/GRU 做一个中文或英文**情感分析**（正/负评论分类），数据可用 IMDB 或自己标注几十条。
**做什么**（`07_project/`）：

1. 数据 → 模型 → 训练 → 评估 → 保存，全流程独立完成；
2. 写一个 `inference.py` 推理 demo：输入一张图/一句话，输出预测；
3. 写 `README.md`：项目简介、运行方法、结果截图——**这就是你简历上的作品**。



### 第8关 进阶彩蛋（可选，弹性时间）🥚

按兴趣任选 1–2 项：

- **torch.compile** 一行提速：[https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- **混合精度训练**（AMP）：[https://pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
- **ONNX 导出部署**：[https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- **从零手写神经网络**（英文神课，Karpathy《Zero to Hero》，看完对内部原理门儿清）：[https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

---



## 📚 学习资源总表（按需取用）


| 类型  | 资源                                               | 链接                                                                                                                                                   |
| --- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 官方  | PyTorch 教程首页（全英文，配中文镜像 pytorch.ac.cn）            | [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)                                                                                     |
| 官方  | 60 分钟入门（最经典的起点）                                  | [https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)       |
| 官方  | Learning PyTorch with Examples（逐步进阶）             | [https://pytorch.org/tutorials/beginner/pytorch_with_examples.html](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)               |
| 官方  | PyTorch Recipes（工程配方合集）                          | [https://pytorch.org/tutorials/recipes/recipes/](https://pytorch.org/tutorials/recipes/recipes/)                                                     |
| 中文  | 李沐《动手学深度学习》(PyTorch 版) 开源书                       | [https://zh.d2l.ai/](https://zh.d2l.ai/)                                                                                                             |
| 中文  | 李沐 B站 视频课程                                       | [https://space.bilibili.com/1567748478](https://space.bilibili.com/1567748478)                                                                       |
| 中文  | 土堆《PyTorch 快速入门教程》（B站，通俗）                        | [https://www.bilibili.com/video/BV1hE411t7NQ](https://www.bilibili.com/video/BV1hE411t7NQ)                                                           |
| 中文  | Datawhale《深入浅出 PyTorch》                          | [https://github.com/datawhalechina/thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch)                                             |
| 中文  | PyTorch 官方中文教程翻译仓库                               | [https://github.com/fendouai/PyTorchDocs](https://github.com/fendouai/PyTorchDocs)                                                                   |
| 英文  | Sebastian Raschka《PyTorch in One Hour》           | [https://sebastianraschka.com/teaching/pytorch-1h/](https://sebastianraschka.com/teaching/pytorch-1h/)                                               |
| 英文  | fast.ai《Practical Deep Learning for Coders》（实战派） | [https://course.fast.ai/](https://course.fast.ai/)                                                                                                   |
| 英文  | Karpathy《Zero to Hero》（进阶神课）                     | [https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) |




## ⚡ 极速版 3 天路线（时间不够时）

- **第1天**：第0关 + 第1关 + 第2关（环境、张量、autograd 一气呵成）
- **第2天**：第3关 + 第5关（搭模型时顺带用上 DataLoader）
- **第3天**：第4关、第6关合并成"一个能跑的项目"直接做第7关 BOSS 战（把对比实验、训练模板、保存加载全融进项目里）
- 第8关留给日后有空再玩。



## 🎁 趣味彩蛋

- **成就系统**：每关交付一个 🏅，集齐 8 枚截图贴进成果墙 `README.md`，就是你的"毕业证书"；
- **一句话总结**：每关学完，用一句话（不许超过 20 字）向朋友解释这关讲的是什么——能讲明白 = 真学会了；
- **反直觉实验**：第2关的"忘掉 zero_grad"、第4关的"不同优化器"都是很好的聊天谈资，建议录下现象；
- 学累了就去看李沐/土堆的视频当"动画片"，不算摸鱼 😄。

---

*祝通关顺利！遇到报错先读 3 遍错误信息，再查官方文档，实在不行再问——这是成为独立玩家的第一步。*