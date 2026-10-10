# Transformer-to-VLA

*From Transformer Fundamentals to Vision-Language-Action Models*

[![MIT License](https://img.shields.io/github/license/Maple-Ze/Transformer-to-VLA?label=license&color=blue)](LICENSE)

## Overview

Modern Transformer architectures power large language models,
vision models, and robot foundation models.

This repository aims to build a bottom-up understanding of these models
through minimal PyTorch implementations and tensor-level analysis.

## Features

- **Implementation First**: Understand model mechanisms by building, running, and modifying code.
- **From-Scratch Modules**: Implement core Transformer computations using basic PyTorch components.
- **Tensor-Level Analysis**: Trace shape changes such as `[B, T, C] → [B, H, T, D]`.
- **Math-Driven Explanations**: Connect equations, design choices, and implementation details.
- **Minimal Experiments**: Explore attention behavior, masking, sequence modeling, and generation.

## Learning Roadmap

| Stage | Topic | Focus | Status |
| --- | --- | --- | --- |
| 01 | Attention Fundamentals | Q/K/V, scaled dot-product attention, multi-head attention, causal masking, embeddings, Transformer blocks, next-token prediction, autoregressive generation | ✅ Completed |
| 02 | Encoder-Decoder Transformer | Cross-attention, post-layer normalization, sinusoidal positional encoding, teacher forcing, sequence-to-sequence reversal | 🚧 In Progress |
| 03 | Vision Transformer | Patch embeddings, positional embeddings, ViT encoder, image classification | 📋 Planned |
| 04 | Diffusion Transformer | Attention in diffusion and generative models | 📋 Planned |
| 05 | Multimodal Transformer | Vision and language tokens, multimodal fusion | 📋 Planned |
| 06 | Vision-Language-Action | From multimodal representations to robot action generation | 📋 Planned |

## Implementations & Notes

Notebooks provide runnable implementations; accompanying notes cover mathematical explanations, tensor analysis, and design discussions.

| Stage | Notebook | Notes | Colab |
| --- | --- | --- | --- |
| 01 · Attention Fundamentals | [Notebook](01_Attention_Fundamentals/01_Attention_Fundamentals.ipynb) | [Notes](01_Attention_Fundamentals/01_Attention_Fundamentals.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/01_Attention_Fundamentals/01_Attention_Fundamentals.ipynb) |
| 02 · Encoder-Decoder Transformer | [Notebook](02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.ipynb) | — | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.ipynb) |

Work through the stages in order, using the notes to interpret results. Compare behavior by changing head counts, sequence lengths, masks, or task rules.

## Getting Started

### Google Colab

1. Click a Colab badge above, or upload a downloaded `.ipynb` through **File → Upload notebook** in [Google Colab](https://colab.research.google.com/).
2. Run cells in order. Restart the runtime and rerun all cells for a fresh experiment.
3. Optionally select a GPU under **Runtime → Change runtime type**. The notebooks use CUDA when available and CPU otherwise.

### Local Setup

In a Python virtual environment, install the [dependencies](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

Open a notebook in Jupyter or VS Code and select the environment's Python kernel.

## Project Structure

```text
Transformer-to-VLA/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── 01_Attention_Fundamentals/
│   ├── 01_Attention_Fundamentals.ipynb
│   ├── 01_Attention_Fundamentals.md
│   └── assets/
│       ├── minigpt_flow.png
│       └── minigpt_flow.py
└── 02_Encoder_Decoder_Transformer/
    └── 02_Encoder_Decoder_Transformer.ipynb
```

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019.
- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), 2020.

## License

This project is licensed under the [MIT License](LICENSE).
