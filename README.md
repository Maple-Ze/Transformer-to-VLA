# Transformer-to-VLA

*从 Transformer 基础到视觉–语言–动作模型*

[![MIT License](https://img.shields.io/github/license/Maple-Ze/Transformer-to-VLA?label=license&color=blue)](LICENSE)

## 项目简介

通过 PyTorch 从零实现 Transformer 核心模块，结合数学推导、张量分析与最小实验，逐步扩展到 ViT、多模态模型和 VLA。

## 项目特点

- **从零实现**：使用 PyTorch 基础组件实现核心计算。
- **张量分析**：追踪 `[B, T, C] → [B, H, T, D]` 等维度变化。
- **数学解释**：对应数学公式、设计动机与实现细节。
- **实验验证**：验证注意力、掩码、序列建模与生成机制。

## 学习路线

| 阶段 | 主题 | 主要内容 | 状态 |
| --- | --- | --- | --- |
| 01 | Attention Fundamentals | Q/K/V、缩放点积注意力、多头注意力、因果掩码、嵌入、Transformer 模块、下一词元预测、自回归生成 | ✅ 已完成 |
| 02 | Encoder-Decoder Transformer | 交叉注意力、后置层归一化、正弦位置编码、教师强制、序列到序列反转 | 🚧 学习中 |
| 03 | Vision Transformer | 图像分块嵌入、位置嵌入、ViT 编码器、图像分类 | 📋 计划中 |
| 04 | Diffusion Transformer | 注意力在扩散与生成模型中的应用 | 📋 计划中 |
| 05 | Multimodal Transformer | 视觉与语言词元、多模态融合 | 📋 计划中 |
| 06 | Vision-Language-Action | 从多模态表征到机器人动作生成 | 📋 计划中 |

## 学习资料

实验笔记本提供可运行代码，配套笔记提供数学解释、张量分析与设计讨论。

| 阶段 | 实验代码 | 笔记 | 在线运行 |
| --- | --- | --- | --- |
| 01 · Attention Fundamentals | [打开笔记本](01_Self_Attention_and_MiniGPT/01_Self_Attention_and_MiniGPT.ipynb) | [阅读笔记](01_Self_Attention_and_MiniGPT/01_Self_Attention_and_MiniGPT.md) | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/01_Self_Attention_and_MiniGPT/01_Self_Attention_and_MiniGPT.ipynb) |
| 02 · Encoder-Decoder Transformer | [打开笔记本](02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.ipynb) | — | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.ipynb) |

按阶段运行代码，结合笔记分析结果；调整头数、序列长度、掩码或任务规则进行对比实验。

## 快速开始

### 使用 Google Colab

1. 点击上方资料导航中的 Colab 按钮在线运行；也可下载 `.ipynb`，在 [Google Colab](https://colab.research.google.com/) 中选择 **文件 → 上传笔记本**。
2. 从第一个代码单元开始按顺序执行；重新开始实验时，重启运行时并从头运行。
3. 代码自动选择可用的 CUDA GPU，否则使用 CPU。可通过 **运行时 → 更改运行时类型** 选择 GPU。

实验数据由代码生成，无需外部文件。缺少依赖时，在导入单元前执行：

```python
%pip install torch numpy matplotlib
```

### 本地运行

在独立 Python 虚拟环境中安装[依赖清单](requirements.txt)中的软件包：

```bash
python -m pip install -r requirements.txt
```

使用 Jupyter 或 VS Code 打开实验笔记本，选择安装了上述依赖的内核。

## 项目结构

```text
Transformer-to-VLA/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── 01_Self_Attention_and_MiniGPT/
│   ├── 01_Self_Attention_and_MiniGPT.ipynb
│   ├── 01_Self_Attention_and_MiniGPT.md
│   └── assets/
│       ├── minigpt_flow.png
│       └── minigpt_flow.py
└── 02_Encoder_Decoder_Transformer/
    └── 02_Encoder_Decoder_Transformer.ipynb
```

## 参考资料

- Vaswani 等，[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017。
- Radford 等，[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)，2019。
- Dosovitskiy 等，[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)，2020。

## 参与贡献

欢迎提交文档、公式和代码运行问题，或提出更清晰的教学实验。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
