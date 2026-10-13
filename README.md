# Transformer-to-VLA

*From Transformer Fundamentals to Vision-Language-Action Models*

[![MIT License](https://img.shields.io/github/license/Maple-Ze/Transformer-to-VLA?label=license&color=blue)](LICENSE)
[![Notebooks](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMaple-Ze%2FTransformer-to-VLA%2Fmain%2F.github%2Fbadges%2Fnotebooks.json&logo=jupyter&logoColor=white)](#implementations--notes)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/01_Attention_Fundamentals/01_Attention_Fundamentals.ipynb)

## Overview

Modern Transformer architectures power large language models,
vision models, and robot foundation models.

This repository aims to build a bottom-up understanding of these models
through minimal PyTorch implementations and tensor-level analysis.

The learning path follows the evolution from attention mechanisms to
encoder-decoder architectures, vision transformers, multimodal models,
and vision-language-action systems.

Diffusion and flow matching experiments introduce generative modeling and conditional action sequences through small MLPs.

## Features

- **From-Scratch Modules**: Implement core Transformer computations using basic PyTorch components.
- **Implementation First**: Understand model mechanisms by building, running, and modifying code.
- **Tensor-Level Analysis**: Trace shape changes such as `[B, T, C] → [B, H, T, D]`.
- **Math-Driven Explanations**: Connect equations, design choices, and implementation details.
- **Minimal Experiments**: Explore attention, masking, sequence modeling, image classification, denoising, and flow-based generation.

## Learning Roadmap

| Stage | Topic | Focus | Status |
| --- | --- | --- | --- |
| 01 | Attention Fundamentals | Q/K/V, scaled dot-product attention, multi-head attention, causal masking, embeddings, Transformer blocks, next-token prediction, autoregressive generation | ✅ Completed |
| 02 | Encoder-Decoder Transformer | Cross-attention, post-layer normalization, sinusoidal positional encoding, teacher forcing, sequence-to-sequence reversal | ✅ Completed |
| 03 | [Vision Transformers](03_Vision_Transformers/) | Patch/CLS tokens, positional embeddings, image classification, visual projection, joint attention, padding masks | ✅ Completed |
| 04 | Multimodal Transformer | Vision and language tokens, multimodal fusion | 📋 Planned |
| 05 | [Diffusion and Flow Models](05_Diffusion_and_Flow_Models/) | DDPM fundamentals, flow matching, conditional trajectory generation | 🚧 In Progress |
| 06 | Vision-Language-Action | From multimodal representations to robot action generation | 📋 Planned |
| 07 | World-Action Models (WAM) | World dynamics prediction and action generation | 📋 Planned |

Stages 03–04 connect visual tokens with multimodal fusion; Stage 05 develops generative action models before their integration in Stage 06. The current Stage 05 experiments can also be studied independently of Stage 04.

## Implementations & Notes

Notebooks provide runnable implementations; accompanying notes cover mathematical explanations, tensor analysis, and design discussions.

Explore the [Knowledge Base](KNOWLEDGE_BASE.md) for core concepts, cross-chapter connections, and open questions.

| Stage | Notebook | Notes | Colab |
| --- | --- | --- | --- |
| 01 · Attention Fundamentals | [Notebook](01_Attention_Fundamentals/01_Attention_Fundamentals.ipynb) | [Notes](01_Attention_Fundamentals/01_Attention_Fundamentals.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/01_Attention_Fundamentals/01_Attention_Fundamentals.ipynb) |
| 02 · Encoder-Decoder Transformer | [Notebook](02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.ipynb) | [Notes](02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.ipynb) |
| 03.1 · ViT from Scratch | [Notebook](03_Vision_Transformers/01_ViT_From_Scratch.ipynb) | [Notes](03_Vision_Transformers/01_ViT_From_Scratch.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/03_Vision_Transformers/01_ViT_From_Scratch.ipynb) |
| 03.2 · Visual Token Projection | [Notebook](03_Vision_Transformers/02_Visual_Token_Projection.ipynb) | [Notes](03_Vision_Transformers/02_Visual_Token_Projection.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/03_Vision_Transformers/02_Visual_Token_Projection.ipynb) |
| 05.1.1 · DDPM: 2D Ring | [Notebook](05_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/01_2D_Ring.ipynb) | [Notes](05_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/01_2D_Ring.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/05_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/01_2D_Ring.ipynb) |
| 05.1.2 · DDPM: 2D Gaussian Mixture | [Notebook](05_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/02_2D_Gaussian_Mixture.ipynb) | [Notes](05_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/02_2D_Gaussian_Mixture.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/05_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/02_2D_Gaussian_Mixture.ipynb) |
| 05.2.1 · Flow Matching: 1D Gaussian | [Notebook](05_Diffusion_and_Flow_Models/02_Flow_Matching/01_1D_Gaussian.ipynb) | [Notes](05_Diffusion_and_Flow_Models/02_Flow_Matching/01_1D_Gaussian.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/05_Diffusion_and_Flow_Models/02_Flow_Matching/01_1D_Gaussian.ipynb) |
| 05.3.1 · DDPM Policy | [Notebook](05_Diffusion_and_Flow_Models/03_Generative_Policies/01_DDPM_Policy.ipynb) | [Notes](05_Diffusion_and_Flow_Models/03_Generative_Policies/01_DDPM_Policy.md) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/05_Diffusion_and_Flow_Models/03_Generative_Policies/01_DDPM_Policy.ipynb) |
| 05.3.2 · Flow Matching Policy | [Notebook](05_Diffusion_and_Flow_Models/03_Generative_Policies/02_Flow_Matching_Policy.ipynb) | — | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maple-Ze/Transformer-to-VLA/blob/main/05_Diffusion_and_Flow_Models/03_Generative_Policies/02_Flow_Matching_Policy.ipynb) |

**Stage 03** covers a ViT classifier trained on synthetic stripes and a visual projection experiment with joint image/text attention, padding masks, per-head visualization, and gradient checks. Both notebooks have illustrated notes and use PyTorch attention/encoder components. The projection experiment demonstrates an untrained interface; it does not train visual-language semantic alignment.

**Stage 05** progresses from DDPM noise prediction on a ring and an eight-mode mixture, to velocity-field learning with Euler integration, then start/goal-conditioned 2D waypoint generation. The two policy examples use the same synthetic data distribution and produce `[B, H, 2]` trajectories, with `H=16` by default. They demonstrate generative policy mechanisms using MLPs; DiT, visual conditioning, and robot execution remain future extensions.

Follow the numbered notebooks within each available stage. Use the notes to interpret results, then vary attention heads, masks, noise schedules, or ODE steps to compare behavior.

## Getting Started

### Google Colab

1. Click a Colab badge in the table above, or upload a downloaded `.ipynb` through **File → Upload notebook** in [Google Colab](https://colab.research.google.com/).
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
├── KNOWLEDGE_BASE.md
├── AGENTS.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── .github/
│   ├── badges/notebooks.json
│   └── workflows/notebook-count.yml
├── 01_Attention_Fundamentals/
│   └── 01_Attention_Fundamentals.ipynb
├── 02_Encoder_Decoder_Transformer/
│   └── 02_Encoder_Decoder_Transformer.ipynb
├── 03_Vision_Transformers/
│   ├── 01_ViT_From_Scratch.ipynb
│   └── 02_Visual_Token_Projection.ipynb
└── 05_Diffusion_and_Flow_Models/
    ├── 01_DDPM_Fundamentals/
    │   ├── 01_2D_Ring.ipynb
    │   └── 02_2D_Gaussian_Mixture.ipynb
    ├── 02_Flow_Matching/
    │   └── 01_1D_Gaussian.ipynb
    └── 03_Generative_Policies/
        ├── 01_DDPM_Policy.ipynb
        └── 02_Flow_Matching_Policy.ipynb
```

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019.
- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), 2020.
- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020.
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), 2023.
- Chi et al., [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), 2023.

## License

This project is licensed under the [MIT License](LICENSE).
