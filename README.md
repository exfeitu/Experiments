# Experiments

这个仓库包含了两个重要的深度学习项目，专注于脑电图(EEG)信号处理和隐私保护机器学习。

## 项目概览

### 1. EEGnet_Pytorch - 基于PyTorch的EEG信号分类
基于EEGNet架构的脑电图信号分类项目，使用PyTorch实现。

**主要特性：**
- 实现EEGNet卷积神经网络架构
- 支持多类别EEG信号分类
- 包含数据增强功能
- 支持自定义配置参数

**技术栈：**
- PyTorch
- NumPy
- Pandas
- timm (PyTorch Image Models)

### 2. DP-MLD - 差分隐私多模态拉普拉斯丢弃
实现论文"Differentially Private Multimodal Laplacian Dropout (DP-MLD) for EEG Representative Learning"的官方代码。

**主要特性：**
- 差分隐私保护的多模态学习
- 支持EEG、皮肤电导(SC)和加速度(ACC)数据
- 针对帕金森病冻结步态(FoG)检测
- 隐私保护的代表性学习

**技术栈：**
- PyTorch
- 差分隐私算法
- 多模态数据融合

## 项目结构

```
Experiments/
├── EEGnet_Pytorch/           # EEGNet项目
│   └── EEGnet/
│       ├── Code/             # 核心代码
│       │   ├── model.py      # EEGNet模型定义
│       │   ├── dataset.py    # 数据加载器
│       │   ├── train.py      # 训练脚本
│       │   ├── run_train.py  # 训练启动脚本
│       │   └── options.py    # 配置参数
│       ├── Data/             # 数据目录
│       │   ├── train/        # 训练数据
│       │   └── test/         # 测试数据
│       └── main.ipynb        # Jupyter演示
├── DP-MLD/                   # DP-MLD项目
│   ├── src/                  # 源代码
│   ├── data/                 # 数据目录
│   ├── fig/                  # 图片资源
│   └── requirements.txt      # 依赖包
└── README.md                 # 本文件
```

## 快速开始

### EEGnet_Pytorch

1. **环境准备**
```bash
cd EEGnet_Pytorch/EEGnet/Code
pip install torch torchvision
pip install pandas numpy timm
```

2. **运行训练**
```bash
python run_train.py
```

3. **配置参数**
编辑 `options.py` 文件来调整模型参数：
- `input_size`: 输入时间序列长度
- `output_size`: 输出类别数
- `F1, F2`: 卷积核数量
- `T1, T2`: 时间卷积核大小
- `electrodes`: 电极数量

### DP-MLD

1. **环境准备**
```bash
cd DP-MLD
pip install -r requirements.txt
```

2. **数据预处理**
```bash
python src/data/process.py
python src/data/get_embedding.py
```

3. **运行演示**
```bash
python src/demo.py
```

4. **运行实验**
```bash
python src/experiments/compare_modal.py
```

## 数据集

### EEGnet_Pytorch
- 支持CSV格式的EEG数据
- 数据应放置在 `Data/train/` 和 `Data/test/` 目录下
- 支持自定义标签类别

### DP-MLD
- 使用多模态帕金森病数据集
- 包含EEG(30通道)、SC和ACC数据
- 数据采样频率：500Hz
- 总计3003个样本，3小时42分钟的有效数据

## 模型架构

### EEGNet
- **Layer 1**: 时间卷积 + 批归一化
- **Layer 2**: 空间卷积 + 激活函数 + 丢弃 + 平均池化
- **Layer 3**: 时间卷积 + 1x1卷积 + 批归一化 + 激活 + 池化 + 丢弃
- **FC**: 全连接层 + 输出层

### DP-MLD
- 差分隐私保护机制
- 多模态数据融合
- 拉普拉斯丢弃正则化
- 隐私预算管理

## 实验配置

### EEGnet_Pytorch默认配置
- 批次大小: 16
- 学习率: 0.001
- 最大轮数: 100
- 丢弃率: 0.25
- 数据增强: 启用

### DP-MLD实验设置
- 隐私参数: ε = 1.0
- 学习率: 0.001
- 批次大小: 32
- 训练轮数: 50

## 结果

两个项目都提供了完整的训练和测试流程，支持：
- 模型训练和验证
- 性能指标计算
- 结果可视化
- 模型保存和加载

## 引用

如果你使用了这些代码，请引用相应的论文：

**EEGNet:**
```bibtex
@article{lawhern2018eegnet,
  title={EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author={Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal={Journal of neural engineering},
  volume={15},
  number={5},
  pages={056013},
  year={2018},
  publisher={IOP Publishing}
}
```

**DP-MLD:**
```bibtex
@article{your_paper_title,
  title={Differentially Private Multimodal Laplacian Dropout (DP-MLD) for EEG Representative Learning},
  author={Your Authors},
  journal={Your Journal},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证。详见各项目目录下的LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 发送邮件至：[your-email@example.com]

## 更新日志

- **v1.0.0** (2024): 初始版本发布
  - 实现EEGNet PyTorch版本
  - 实现DP-MLD算法
  - 添加完整的训练和测试流程

 