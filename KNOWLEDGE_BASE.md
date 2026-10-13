# Transformer-to-VLA · 技术知识库

本知识库围绕技术问题组织已实现的机制、适用条件与跨章节联系。实验步骤、完整推导和局部问答保留在各章 Notebook / Notes；项目使用方法与路线见 [README](README.md)。维护规则见 [AGENTS.md](AGENTS.md)。

## 阅读与证据约定

| 标记 | 含义 |
| --- | --- |
| 数学事实 / 数学推论 | 在写明的定义与假设下成立；不等于某个有限模型已经学到它 |
| 实现事实 | 可在链接的代码中直接核对的结构、数据构造或算法 |
| 实验观察 | 某次实际运行的输出；必须同时考虑任务、参数与评价范围 |
| 教学简化 | 为隔离机制而采用的设计，不能直接推广到真实任务 |
| Future Topics / 研究问题 | 尚未实现或验证的内容，不作为已有能力 |

除另有注明外，下文引用的实验数值来自 [已保存输出的版本 `38dd0d4`](https://github.com/Maple-Ze/Transformer-to-VLA/commit/38dd0d4)：默认参数、随机种子 42、CUDA 执行。ViT 的逐 Head 热力图与对应输出另见 [`faff6d6`](https://github.com/Maple-Ze/Transformer-to-VLA/commit/faff6d6)，训练配置未变。它们是可追溯的单次观察，不是跨种子基准；修改模型或数据后应重新判断结论。

为避免不同章节同名变量混淆，本文用 $B$ 表示批量， $L_q,L_k$ 表示 Query / Key 数， $C$ 表示隐藏维度， $n_h$ 表示 Attention 头数， $d_h=C/n_h$ 。动作长度用 $H$ ，DDPM 总步数用 $K$ （代码变量 `T`），Flow 时间用 $\tau\in[0,1]$ 。原代码中的 `H` 在 Attention 章节表示头数，在 Policy 章节表示动作长度。

- [Part I — Transformer Fundamentals](#part-i)
- [Part II — Vision Representation](#part-ii)
- [Part III — Generative Modeling](#part-iii)
- [Part IV — From Transformer to VLA/WAM](#part-iv)
- [Part V — Open Questions](#part-v)

<a id="part-i"></a>
## Part I — Transformer Fundamentals

### I.1 Attention 究竟在选择什么，为什么需要 Q/K/V？

**数学事实。** 对一组 Query 和一组可读取的特征，Attention 先计算匹配分数，再沿 Key 维分配权重，最后聚合 Value：

$$
Q=X_qW_Q,\qquad K=X_kW_K,\qquad V=X_kW_V,
$$

$$
S=\frac{QK^\top}{\sqrt{d_h}}+M,\qquad
A_{ij}=\frac{\exp(S_{ij})}{\sum_{j'}\exp(S_{ij'})},\qquad
y_i=\sum_j A_{ij}v_j.
$$

这里 $M$ 是可见性掩码；每一行必须至少有一个合法 Key。Q/K 决定“根据什么匹配”，V 决定“传递什么内容”。输出 Token 数由 Query 数 $L_q$ 决定，不由 Value 数决定。

直接用 $XX^\top$ 也能构造 Attention，但独立投影让匹配和内容具有不同的可学习参数。特别是 $W_Q\ne W_K$ 时，两个位置的匹配分数不必对称。上式使用行向量右乘的数学记法；PyTorch `nn.Linear` 存储的权重方向相反，计算的是 `x @ weight.T + bias`。

除以 $\sqrt{d_h}$ 是尺度控制：在 Q/K 各分量近似独立、零均值、单位方差的启发式假设下，点积方差随 $d_h$ 增长，缩放可以减轻 Softmax 过度饱和；它不是对任意训练后分布的精确方差保证。

**解释边界。** Dropout 前，每行权重和为 1；训练时 Attention Dropout 后不一定如此。权重大只表示当前层的加权关系，不能单独证明语义理解或因果重要性。Key Bias 的冗余条件、Value 聚合与 Dropout 的细节见 [Attention Notes][note-attention]。

### I.2 Self-Attention 与 Cross-Attention 的差别在哪里？

**实现事实。** 差别在于输入来源，而不是两套不同的 Softmax 算法。

| 机制 | Q 的来源 | K/V 的来源 | 多头权重形状 | 当前例子 |
| --- | --- | --- | --- | --- |
| Self-Attention | 当前序列 | 同一序列 | `[B, n_h, L, L]` | Mini-GPT、ViT、拼接后的视觉/文本序列 |
| Cross-Attention | 待读取信息的序列 | 另一组 memory | `[B, n_h, L_q, L_k]` | Decoder 读取 Encoder 输出 |

第二章 Decoder 的 Query 长度可以与 Encoder memory 长度相等或不同；每个 Query 读取全部合法源位置后，得到自身的输出表示。代码中的 `assert T == S` 只限制当前因果掩码分支，不能用来判定是否为 Cross-Attention。第三章把视觉和文本拼成同一序列再做 Self-Attention，虽然发生了跨模态交互，但并不是代码层面的 Cross-Attention 模块。

这两种连接方式为后续多模态融合提供接口思路；仓库尚未比较它们在真实视觉语言任务上的效果。来源：[Encoder–Decoder 实验][nb-encdec] 的 `MultiHeadAttention` 与 [Notes §4–7][note-encdec]；[Visual Token Projection][nb-projector] 的 `combined` / `attn`。

### I.3 多头是否把信息割裂了？

**数学事实与实现事实。** 当前实现先对完整 $C$ 维输入投影，再重排成多个 Head：

```text
[B, L, C] → Q/K/V → [B, n_h, L, d_h]
                         ↓ 每个 Head 独立匹配全部合法 Token
                     [B, n_h, Lq, Lk]
                         ↓ Value 聚合、拼接、输出投影
                     [B, Lq, C]
```

每个 Head 的投影可以使用全部输入 Channel；独立的是投影后的匹配和聚合。拼接后的输出投影重新混合各头信息。因此多头提供的是多套 Token 权重分布，而不是把原始 Token 分成互不通信的小组。

**权重与特征不能混淆。** 在 TinyViT 中，每个 Head 的权重是 `[B,17,17]`，Value 与加权输出是 `[B,17,16]`；4 个 Head 拼接的是 Value 加权后的输出，得到 `[B,17,64]`，再做输出投影。对权重取 Head Mean 只是可视化汇总，不能替代这个前向计算，也不是模型最终的 CLS Feature。推导与图解见 [ViT Notes §5–6][note-vit]。

在固定 $C$ 下，增加头数会减小 $d_h$ 。参数独立和随机初始化提供了不同分工的可能性，但不保证各头学到不同语义，也不保证更多 Head 更好。完整论证见 [Attention Notes §7][note-attention]；代码见 [Attention 实验][nb-attention] 的 `MultiHeadSelfAttention`。

### I.4 位置与 Mask 分别解决什么问题？

**数学事实。** 不含位置机制、全可见且参数共享的 Self-Attention 对 Token 重排是等变的：同步重排输入，输出也相应重排。它不会凭 Token 内容自动知道空间位置或序列顺序。Causal Mask 本身引入了可见性结构，因此不能把这一无掩码结论不加条件地套到所有 Attention。

**实现事实。** Mini-GPT 和 TinyViT 加入可学习位置表；第二章使用固定 Sin/Cos：

$$
PE(p,2i)=\sin\left(\frac{p}{10000^{2i/C}}\right),\qquad
PE(p,2i+1)=\cos\left(\frac{p}{10000^{2i/C}}\right).
$$

位置表示提供“在哪里”，Mask 规定“允许读取哪里”。两者不能互相替代。

| Mask | 解决的问题 | 当前范围 |
| --- | --- | --- |
| Causal Mask | 下一 Token 训练不能读取未来目标 | 第一章与第二章 Decoder Self-Attention：屏蔽 `j > i` 的分数，再 Softmax |
| Key Padding Mask | 有效 Query 不应从填充 Token 读取内容 | 第三章投影实验：`key_padding_mask=True` 屏蔽相应 Key/Value |

Padding Mask 不会自动将填充位置的 Query 输出清零；逐 Token 损失或池化还需单独排除填充位置。投影实验的布尔 Mask 为 `[B,19]`，沿 Head 与 Query 维广播，屏蔽的是 Key 列；第二个样本的一个 PAD Key 对应 `4×19=76` 个零权重，PAD Query 行仍可读取有效 Keys。图解与形状区别见 [Visual Token Projection Notes §5][note-projector]。第二章使用固定长度数据，没有实现 Padding Mask。Cross-Attention 中已给定完整源序列，当前实验不需要对源位置施加因果下三角限制。

**实验观察。** 第一章修改未来 Token 后，前三个位置 logits 最大差异为 0，验证了该输入上的因果隔离。投影实验断言 Padding Key 对应的注意力权重为 0。前者检查信息泄漏，后者检查填充隔离，都不评价任务语义。来源：[Attention][nb-attention]、[Encoder–Decoder][nb-encdec]、[Visual Token Projection][nb-projector]。

### I.5 Encoder-only、Decoder-only 和 Encoder–Decoder 如何分工？

| 架构 | 信息路径 | 本仓库实例 |
| --- | --- | --- |
| Encoder-only | 在给定输入内做双向交互，再读取任务输出 | TinyViT 用 CLS 表示做分类 |
| Decoder-only | 在已有前缀内做因果交互，预测后续 Token | Mini-GPT |
| Encoder–Decoder | Encoder 编码源输入；Decoder 因果处理目标前缀，并跨注意力读取源 memory | 数字序列翻转 |

归一化位置是另一条设计轴，不能用 Pre-LN / Post-LN 来定义 Encoder 或 Decoder。以一个子层 $F$ 为例：

$$
\text{Pre-LN: }Y=X+F(\mathrm{LN}(X)),\qquad
\text{Post-LN: }Y=\mathrm{LN}(X+F(X)).
$$

Mini-GPT 与 TinyViT 使用 Pre-LN / GELU；第二章使用 Post-LN / ReLU；第三章独立的 `ToyVisionEncoder` 复用 PyTorch Encoder 的 Post-LN / ReLU 配置，并未加载 TinyViT 权重。Attention 负责跨 Token 交互，逐 Token MLP 负责非线性特征变换；LayerNorm 沿特征维归一化，残差连接保留直接的信息与梯度路径。

这些实现展示不同组合，尚无控制变量实验证明某种组合在本项目任务中更优。来源：[Attention Notes §9][note-attention]、[Encoder–Decoder][nb-encdec]、[TinyViT][nb-vit]、[Visual Token Projection][nb-projector]。

### I.6 为什么训练可以并行，而自回归生成仍需循环？

**数学事实。** 自回归分解将联合分布写成条件分布的乘积：

$$
p_\theta(x_{1:L})=\prod_{i=1}^{L}p_\theta(x_i\mid x_{1:i-1}).
$$

其中 $L$ 为序列长度， $x_i$ 为第 $i$ 个 Token， $x_{1:i-1}$ 表示此前的 Token 前缀；当 $i=1$ 时，该前缀为空。

训练时真实序列已经给定，将输入与下一 Token 标签错开一位，配合 Causal Mask 可以一次计算各位置的损失。第二章以 `[BOS, target[:-1]]` 输入 Decoder，监督为含 EOS 的完整目标；右移和因果遮罩缺一不可。

生成时未来前缀尚不存在，必须取最后位置的 logits 选择下一 Token，再追加到前缀。当前两个实验都用 argmax；第二章只计算一次 Encoder memory，但每一步仍重算目标前缀，没有 KV Cache。

**教学简化与实验观察。** Mini-GPT 只学数字加一，该规则只依赖当前 Token，不能以 loss 下降证明长距离建模能力。第二章的 300 步输出中，前两个例子翻转正确，第三个例子提前生成 EOS；返回序列还包含 BOS，比较时应先去掉它。低训练 loss 和一次 Teacher Forcing 预测正确，不等于所有自由生成均正确。

来源：[Attention Notes §10–13][note-attention]、[Encoder–Decoder][nb-encdec] 的 `make_reverse_batch` / `translate_reverse`。依赖历史的消融实验、变长序列、KV Cache 和 RoPE 属于 **Future Topics**。

<a id="part-ii"></a>
## Part II — Vision Representation

### II.1 为什么图像可以成为 Transformer 的序列？

**实现事实。** Transformer 接收的是特征向量序列，Token 不必来自离散词表。对可被 patch 边长 $P$ 整除的图像：

$$
N_v=\frac{H_{\mathrm{img}}W_{\mathrm{img}}}{P^2},\qquad
Z_{\mathrm{patch}}\in\mathbb R^{B\times N_v\times D_v}.
$$

每个 patch 展平为 $P^2C_{\mathrm{in}}$ 维，再通过共享线性投影映射到 $D_v$ 。当前 `Conv2d(kernel_size=P, stride=P)` 实现这一操作；展平的是 patch 内像素，不是把整幅图像变成一个 Token。

TinyViT 的默认路径是 `[B,1,32,32] → [B,16,64]`。空间结构通过 patch 的排列及位置嵌入进入模型。减小 $P$ 会增加 Token 数；在隐藏维度不变时，标准稠密 Attention 的匹配计算量和权重矩阵大小随序列长度平方增长。`P=8 → 4` 时，含 CLS 的长度从 17 变成 65，权重元素数量比为 $65^2/17^2$ 。

这里 `16` 是 Token 数，`64` 是每个 Token 的特征维度；默认配置下单个 Patch 的原始像素数也为 64，只是数值巧合，改变 Patch 大小不必同时改变隐藏维度。来源：[TinyViT][nb-vit] 的 `PatchEmbedding` 与 [ViT Notes §2][note-vit]。这是形状与计算规模推论，不是实际运行时间测量。

### II.2 CLS Token 如何得到整张图像的信息？

输入序列为：

$$
Z_0=[z_{\mathrm{cls}};Z_{\mathrm{patch}}]+E_{\mathrm{pos}}.
$$

CLS 是参与双向 Self-Attention 的可学习向量，可以在多层交互中聚合 patch 信息。当前分类头读取最后的 CLS 表示；它不是将所有 patch 简单求均值。实现使用 2 个 Encoder Block、4 个 Head、64 维隐藏特征，并复用 `nn.MultiheadAttention`，不是重新手写一套 Q/K/V 底层算子。

**共享初值不等于共享最终特征。** `cls_token` 是所有图像共用的可学习参数，初始化为零；加入 CLS 位置向量后，进入首层的表示是 $z_{\mathrm{cls}}+e_{\mathrm{pos},0}$ ，通常已非零。之后它读取各图像不同的 Patch 信息，最终表征可以随图像改变，但不保证任意两张图像的表征严格不同。分类代码读取最后一个 Block 的 CLS，再沿其特征维做 LayerNorm；这与先逐 Token 做同一 LayerNorm、再取 CLS 等价。单层 Attention 的一行权重还不是这个最终特征，见 [ViT Notes §3、§7][note-vit]。

**实验观察。** 800 张程序生成条纹图像分为 640 张训练、160 张测试，8 个 epoch 后保存的测试准确率为 1.000。这只支持该合成分布上的分类表现；单个 patch 已含条纹方向信息，所以不能据此证明全局空间推理能力，也不能用去位置嵌入后的表现断言位置编码无用。

**热力图解释。** 设最后一层第 $h$ 个 Head 的权重为 $A^{(h)}$ ，Token 0 是 CLS， $N_v$ 是 Patch 数；本节未使用 Attention Dropout，因此：

$$
\sum_{j=1}^{N_v} A^{(h)}_{0j}=1-A^{(h)}_{00}.
$$

截取 `attention[0, :, 0, 1:]` 后保留各 Head，默认重排为 `[4,4,4]`（Head、网格行、网格列），而非重新归一化到和为 1。当前各 Head 使用独立 colorbar 和自身最小／最大值映射颜色，以观察内部空间变化；跨 Head 比较绝对值需读各自刻度，不能直接比较颜色。颜色变化鲜明可能只对应很小的权重差异，也不能证明 Head 具有独立语义分工。权重还未包含 Value 内容、输出投影、残差与 MLP 的完整作用，热力图不能替代归因实验。来源：[TinyViT][nb-vit] 的训练与 CLS Attention 单元、[ViT Notes §5–7][note-vit]。

### II.3 Visual Projector 对齐了什么，又没有对齐什么？

**数学与实现事实。** 投影先解决隐藏维度的接口兼容：

$$
Z_v\in\mathbb R^{B\times N_v\times D_v},\qquad
Z'_v=Z_vW_p^\top+b_p\in\mathbb R^{B\times N_v\times D_l},
\quad W_p\in\mathbb R^{D_l\times D_v}.
$$

当前实验从 `D_v=64` 投影到 `D_l=96`，随后沿序列维拼接视觉与文本 Token：`[B,16,96] + [B,3,96] → [B,19,96]`。这里的 `+` 表示拼接两个序列，不是逐元素相加；投影本身没有压缩视觉 Token 数量。

维度相同意味着后续算子可以接收两类表示，不意味着它们已经具有共同语义。当前视觉编码器、文本 Embedding、Projector 和融合 Attention 均随机初始化；没有真实 tokenizer、语义训练、文本位置编码或自回归语言解码器。

**实验观察。** 人工标量探针只作用于一个 Text Token 的输出，仍可沿 Visual Keys / Values 的路径传回共享 Projector；这与 ViT 只监督 CLS、其余 Token 参与计算的机制相通。[保存输出 `9a85235`](https://github.com/Maple-Ze/Transformer-to-VLA/commit/9a85235) 中，权重与梯度形状均为 `[96,64]`，梯度 Frobenius 范数为 9.463445，汇总了 6144 个权重偏导数，不包含 96 个 Bias 梯度。它验证本次输入和探针的非零参数依赖；没有优化器更新，也不是图文对齐指标。若某次梯度为零，也不能仅据此断言图上没有路径，仍需排查局部导数与抵消。来源：[Visual Token Projection][nb-projector] 的梯度检查、[Notes §8][note-projector] 的链式法则与范数解释。

### II.4 Vision Encoder 如何连接后续多模态模型？

当前已演示的连接是“多个视觉特征 → Projector → 与文本 Token 拼接 → 双向 Self-Attention”。视觉侧使用完整 Encoder Layer；融合侧仅有 `nn.MultiheadAttention`，没有附加 MLP、LayerNorm 或残差，不能把两者视作相同结构。多个 Token 保留了空间位置对应的表示槽位，但经过视觉 Attention 后，每个槽位可以包含全局信息，不再只代表孤立 patch。

去掉分类 Head 与导出 Patch Features 也不是同一件事：TinyViT 当前 `forward` 返回分类 logits，可选返回值是 Attention 权重，而非完整 Encoder 特征。用于下游的 Patch Features 应来自编码后的 Token 表征，不能把热力图作为它们的替代品；本项目尚未为该分类器实现这一导出接口。概念区分见 [ViT Notes §7–8][note-vit]，独立的接口演示见 [Visual Token Projection][nb-projector]。

第二章提供另一种接口思路：让某类 Query 通过 Cross-Attention 读取视觉 memory。联合 Self-Attention 的 Text → Visual 子块可以承担跨模态读取，但它不是一个独立的 Cross-Attention 层。两种连接都需要明确位置表示、可见性和任务监督；是否共享隐藏宽度还取决于具体模块的输入投影设计。

**Attention Mass 的计数基线。** 对固定样本的 Text Query $i$ 与 Head $h$ ，视觉 Keys 为索引 $0,\ldots,N_v-1$ ，有效文本 Key 数为 $L_{\mathrm{valid}}$ 。在本实验双向可见、视觉 Keys 全部有效且 Attention Dropout 为零的条件下：

$$
M_{\mathrm{visual}}^{(h)}=\sum_{j=0}^{N_v-1}A_{ij}^{(h)},\qquad
M_{\mathrm{visual}}^{(h)}+M_{\mathrm{text}}^{(h)}=1,\qquad
M_{\mathrm{uniform}}=\frac{N_v}{N_v+L_{\mathrm{valid}}}.
$$

前两式描述实际权重分配，最后一式是假设所有有效 Key 均匀分配时的参照值，并非随机初始化网络的必然输出。16 个视觉 Key、3 个文本 Key 时基线为 `16/19≈0.8421`；屏蔽一个 PAD 后变为 `16/18≈0.8889`。因此较大的视觉权重总和可能仅来自视觉 Token 较多。它还不是信息量或因果贡献占比，解释时需结合 Value、输出投影及干预证据。与 II.2 的 CLS → Patch 图相同，本实验保留每个 Head 的原始权重，独立色标只调整颜色映射。定义、示意图与本次输出的区分见 [Projection Notes §6–7][note-projector]。

**Future Topics。** 当前没有训练这两种多模态方案的对照实验。预训练视觉主干、图文训练目标、因果/前缀 Mask、视觉压缩和真实 tokenizer 留待多模态章节；不能把当前拼接实验称作已训练的 VLM。

<a id="part-iii"></a>
## Part III — Generative Modeling

### III.1 为什么可以一步采样 DDPM 的任意噪声阶段？

**数学事实。** 设干净样本为 $x_0$ ，前向步 $t=1,\ldots,K$ ：

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_tI),\qquad
\alpha_t=1-\beta_t,\quad \bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

各步噪声彼此独立，并独立于干净样本。条件于 $x_0$ ，线性组合的独立高斯噪声可以合并；由于 $\mathrm{Cov}(cX)=c^2\mathrm{Cov}(X)$ ，方差系数在采样式中取平方根。因此边际采样为：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad
\epsilon\sim\mathcal N(0,I).
$$

这允许训练时随机抽取一个时间步，直接构造对应输入，不必执行完整前向链。公式给出的是该步的条件边际分布；分别独立生成多个时间点，并不自动构成同一条马尔可夫链轨迹。

**实现约定。** 代码下标 `t=0` 对应公式中的首次加噪步，不是干净 $x_0$ 。`alpha_bar_prev[0]=1` 对应数学上的 $\bar\alpha_0=1$ 。

**教学简化。** 线性 beta 端点均为 `1e-4` 和 `0.02`：圆环与 DDPM Policy 的 `K=100` 得到 $\bar\alpha_K\approx0.364$ ，八簇的 `K=200` 得到约 0.132。末端仍保留信号，用标准高斯初始化反向采样有先验不匹配。改变步数也改变累计噪声，不能当作只改变计算量的公平对照。

来源：[圆环实验][nb-ring] / [Notes][note-ring]、[八簇实验][nb-gmm] / [Notes][note-gmm]；数学定义参见 [DDPM 原论文](https://arxiv.org/abs/2006.11239)。

### III.2 预测噪声为何能用于生成，是否只是把噪声减掉？

训练通过已知 $x_0,t,\epsilon$ 构造监督：

$$
\mathcal L_\epsilon=\mathbb E\left[\lVert\epsilon_\theta(x_t,t)-\epsilon\rVert^2\right].
$$

时间输入用于区分不同噪声级别的任务。平方回归的理想解是给定 $(x_t,t)$ 的条件平均噪声，不是恢复每条训练样本唯一、可识别的随机噪声。当前无权 MSE 是简化训练目标，不能直接等同于精确最大似然。

**与 ELBO 的关系。** 对数学时间步 $t=2,\ldots,K$ ，设反向协方差 $\sigma_t^2I$ 固定、严格为正且不依赖 $\theta$ 。负 ELBO 的中间项为前向条件后验与反向模型之间的 KL。采用噪声参数化后：

$$
D_{\mathrm{KL}}\left(q(x_{t-1}\mid x_t,x_0)\mathbin{\Vert}p_\theta(x_{t-1}\mid x_t)\right)
=C_t+\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2.
$$

$C_t$ 与 $\theta$ 无关，再对数据和前向噪声取期望即可得到该步训练项。当前代码去掉时间权重，均匀抽样全部时间步；完整目标还包含末端先验项与首步重建项，不能仅由无权 MSE 代替。首步不属于这个非退化高斯 KL 推导。配方与权重推导见 [八簇实验 §3、§6][nb-gmm]，对应 [DDPM 公式 (5)、(8)、(12)、(14)](https://arxiv.org/abs/2006.11239)。

生成时没有真实 $x_0$ 或真实 $\epsilon$ 。模型预测被代入反向均值，再迭代采样：

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right),
\qquad x_{t-1}=\mu_\theta(x_t,t)+\sigma_tz.
$$

其中 $z\sim\mathcal N(0,I)$ ，最后一步不再添加随机项。圆环实验在非末步采用 $\sigma_t^2=\beta_t$ ；八簇与 DDPM Policy 采用：

$$
\sigma_t^2=\tilde\beta_t=\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}.
$$

这是已知干净样本时的前向条件后验方差，不等于已经精确学到了真实反向条件分布，也不保证优于另一种方差选择。数学 $t=1$ 时，已知 $x_0$ 的后验是 Dirac 分布，方差表达式给出零；不能把它代入含方差倒数的精度或 KL 推导。反向均值可由估计的 $\hat x_0$ 代入后验均值得到，详见 [圆环实验 §5、§6][nb-ring]。

当前噪声网络均为 MLP。圆环拼入时间标量，八簇与 Policy 使用正弦时间嵌入。它借用了位置编码的多频率表示思想，但编码的是生成阶段，不是 Token 位置。来源：[圆环 Notes][note-ring]、[八簇 Notes][note-gmm]、[DDPM Policy][nb-ddpm-policy]。

### III.3 Flow Matching 的 Probability Path 与 Velocity Field 是什么关系？

**实现事实。** 当前采用独立端点配对： $x_0\sim p_0$ 为噪声， $x_1\sim p_1$ 为数据；这与 DDPM 中 $x_0$ 表示干净数据的约定不同。令：

$$
x_\tau=(1-\tau)x_0+\tau x_1,\qquad
u=x_1-x_0,\qquad \tau\sim U(0,1).
$$

Probability Path 指每个 $\tau$ 上随机变量 $x_\tau$ 的分布 $p_\tau$ ；给定端点对后的线段是构造它的一种方式。网络只接收 $(x_\tau,\tau)$ ，优化：

$$
\mathcal L_{\mathrm{FM}}=\mathbb E\left[\lVert v_\theta(x_\tau,\tau)-u\rVert^2\right],\qquad
v^*(x,\tau)=\mathbb E[u\mid x_\tau=x,\tau].
$$

**数学推论。** 条件期望来自平方回归的最优解。对某一对端点， $u$ 是常量；但同一位置可能由不同样本对到达，所以边际速度一般随位置和时间改变。训练线段也不意味着学习后 ODE 的每条轨迹都是相同的线段。

一维实验取 $p_0=\mathcal N(0,1)$ 、 $p_1=\mathcal N(4,0.5^2)$ 。由独立高斯变量线性组合可得：

$$
x_\tau\sim\mathcal N\left(4\tau,(1-\tau)^2+0.25\tau^2\right).
$$

分布既移动也改变宽度，因此“所有粒子统一加 4”不能完成目标分布变换。此式是对现有数据构造的解析推论，并非额外训练结果。条件目标自身仍有不确定性，故非零回归 loss 不自动意味着分布生成失败。

来源：[一维 Flow Matching][nb-fm] / [Notes][note-fm]；条件路径到边际速度场的理论依据见 [Flow Matching 原论文 §3](https://arxiv.org/html/2210.02747v2#S3)。当前独立配对不等于已实现端点分布之间的最优传输耦合。

### III.4 为什么训练不求解 ODE，生成时却需要求解？

训练时已知端点，可以直接构造任意 $\tau$ 的状态和速度监督。生成时只有源噪声，需沿预测的速度场前进：

$$
\frac{dx(\tau)}{d\tau}=v_\theta(x(\tau),\tau),\qquad x(0)\sim p_0,
$$

$$
x^{(k+1)}=x^{(k)}+\Delta\tau v_\theta(x^{(k)},k/N),\qquad
\Delta\tau=1/N.
$$

当前用显式 Euler，循环 $k=0,\ldots,N-1$ 。一维实验默认 100 步，Policy 默认 50 步。对于固定模型、条件和初始噪声，算法不再额外抽取每一步的随机扰动；不同初始噪声仍会得到不同生成样本。

增加步数主要控制数值离散误差，不能消除速度场建模误差。端点分布逼近还依赖训练质量及流的适用条件，不能只凭写出 ODE 就保证生成正确。

**实验观察。** 一维实验用 10,000 步训练、100 步 Euler、5,000 个生成样本，保存的均值/标准差为 `4.055 / 0.516`，目标为 `4 / 0.5`。这支持该次分布近似，均值与方差接近本身也不是完整分布相等的证明。来源：[一维实验][nb-fm] 的 `sample_flow`。

### III.5 DDPM 与 Flow Matching 的共同点和区别是什么？

二者都用容易采样的随机变量构造监督，训练条件回归网络，再通过多次网络调用生成样本。生成机制与网络骨干是不同设计维度：当前两者均可由 MLP 实现，使用扩散不等于使用 DiT。

| 维度 | 当前 DDPM | 当前 Flow Matching |
| --- | --- | --- |
| 训练状态 | 按噪声调度衰减数据并叠加噪声 | 独立噪声端点与数据端点线性插值 |
| 监督量 | 注入噪声 $\epsilon$ | 端点差 $x_1-x_0$ |
| 生成方向 | 离散噪声阶段从 $K$ 到 0 | 连续 Flow 时间从 0 到 1 |
| 生成随机性 | 初始噪声及非末步扰动 | 初始噪声；当前 Euler 更新无额外随机项 |
| 主要近似 | 学到的反向模型、固定方差、短链先验不匹配 | 学到的速度场、有限步 Euler 积分 |

这张表只比较本仓库实现，不能推广成“所有 Diffusion 都只能随机采样”或“所有 Flow Matching 都用该直线路径”。其他采样器、路径设计和 Transformer 骨干属于 **Future Topics**。不同回归目标的 loss 数值也不能直接比较优劣。

### III.6 如何从生成点分布，过渡到 Conditional Generation 与 Action Chunk？

**实现事实。** 两个 Policy 将整个片段视为一个联合样本：

$$
A=[a_0,\ldots,a_{H-1}]\in\mathbb R^{H\times2},\qquad
o=[s_x,s_y,g_x,g_y]\in\mathbb R^4,\qquad H=16.
$$

训练数据是起点 $s$ 到目标 $g$ 的线性插值，内部 waypoint 加标准差 0.02 的噪声，首尾噪声为零。两本使用相同的数据分布，但独立抽样。模型将片段展平，与时间嵌入、条件拼接，再输出 `[B,H,2]`。

DDPM 学习 $\epsilon_\theta(A_t,t,o)$ ；FM 学习 $v_\theta(A_\tau,\tau,o)$ 。条件 $o$ 保持干净，不被扩散或流匹配插值；FM 的干净轨迹用于构造中间状态和速度监督，但不是额外直接输入。对整段轨迹采样高斯噪声意味着 $\mathrm{vec}(\epsilon)\sim\mathcal N(0,I_{2H})$ ，而不是非方阵“协方差”。

需要分清三个轴：

| 轴 | 含义 | 不能混淆的对象 |
| --- | --- | --- |
| waypoint 索引 $h=0,\ldots,H-1$ | 输出片段内部的顺序 | 不是第几次去噪 |
| $t$ 或 $\tau$ | 生成算法内部的阶段 | 不是机器人已经执行到的物理时间 |
| 真实执行与重新观测的时刻 | 闭环控制中的时间 | 当前实验没有模拟这一循环 |

两种采样都反复更新整段片段，不是按自回归顺序逐个生成 waypoint。固定长度 MLP 对不同展平槽位有不同参数，因此这里不依赖 Transformer 位置嵌入来区分动作顺序，也不能直接复用同一权重处理任意 $H$ 。

**教学简化。** 输出是二维坐标，尚不是速度、关节或力矩指令。随机采样带来变化，但数据只有直线附近的小扰动，不会凭空学到左右绕障的多峰策略。八簇 DDPM 的“多模态”指概率分布多峰；视觉—语言的“多模态”指输入信息种类，两个词义不可混用。

来源：[DDPM Policy][nb-ddpm-policy] / [Notes][note-ddpm-policy]、[Flow Matching Policy][nb-fm-policy]。

### III.7 给定起终点为什么仍有误差，能否据此比较两种 Policy？

条件输入只提供学习信息，不是输出的硬约束。训练数据端点严格匹配条件，但当前采样器没有端点投影，所以生成结果可以偏离。

**实验观察。** DDPM Policy 用 20,000 步训练，在图示的 4 组条件上得到平均端点误差 0.0203；FM Policy 用 5,000 步训练，在另外 128 组随机条件上得到起点/终点平均误差 0.0666 / 0.0656。两者训练预算、激活函数、时间编码、采样方式及评价样本不同，这些数字不能证明 DDPM 优于 FM。

二阶差分 MSE 衡量离散坐标序列的弯折变化，但没有指定真实时间间隔，也没有动力学或碰撞模型，不能等同于物理加速度约束或安全性。公平比较需要共同测试条件、明确采样预算、多个随机种子，以及任务相关指标；该基准尚未实现。

来源：[DDPM Policy][nb-ddpm-policy] / [Flow Matching Policy][nb-fm-policy] 的质量检查单元；开放问题见 [Part V](#part-v)。

<a id="part-iv"></a>
## Part IV — From Transformer to VLA/WAM

### IV.1 哪些基础已经具备，哪些连接还不存在？

| 已有机制 | 可支持的后续技术问题 | 当前边界 |
| --- | --- | --- |
| Attention、Mask、Encoder–Decoder | 不同信息源如何交互，如何限制信息泄漏 | 数字序列任务，没有真实语言训练 |
| Patch Tokens、Visual Projector | 图像表示如何进入兼容的 Token 接口 | 分类器与投影实验相互独立；后者未做语义训练 |
| DDPM / FM 条件片段生成 | 如何从噪声联合生成连续输出 | 条件只有二维起终点，没有视觉或语言输入 |

这些是已经分别运行的模块。仓库没有把第三章编码器接入第四章 Policy，也没有端到端 VLA 训练。

**Future Topics：待实现的连接方案。** 下列流程描述后续需要明确的接口，不代表已有计算图：

```text
图像 → Vision Encoder → Visual Projector ─┐
语言 → Tokenizer / Text Embedding ────────┼→ 条件表示 / 多模态交互
机器人本体状态 ──────────────────────────┘             ↓
                                     带生成时间的动作模型 + 初始噪声
                                                        ↓
                                                   Action Chunk
                                                        ↓
                                  执行部分动作 → 新观测 → 再生成
```

如何编码状态、视觉/文本与动作是否共享 Transformer、采用何种 Mask、哪些参数冻结，都需要依据后续具体架构和实验确定，不能从“张量能拼接”直接推导。

### IV.2 从当前教学实现到真实 VLA，还缺少哪些证据？

**Future Topics。** 至少需要实际完成图文语义训练或接入经过验证的预训练表示，定义机器人动作空间及归一化，使用匹配的观测—指令—动作数据，再验证条件信息能否改变正确的行为。生成端点接近、Projector 梯度非零，都不足以替代这些证据。

生成式动作模型可以构成 VLA 的一部分，但“使用 Flow Matching”不是 VLA 的充分条件，也不是唯一实现方式。LM Head 与 Action Head 的联系是任务输出接口；当前 LM Head 直接映射到词表 logits，DDPM / FM Policy 则需要条件网络与迭代采样，不能把 Action Expert 一概等同于线性层，见 [Encoder–Decoder Notes §9、§12][note-encdec]。当前 Policy 的 MLP 是研究生成机制的起点；把它换成 Transformer，还要定义动作 Token、时间条件与跨模态信息路径。

用于后续阅读的具体参照：[Diffusion Policy](https://arxiv.org/abs/2303.04137) 将条件动作生成与视觉输入、滚动执行联系起来；[π₀](https://arxiv.org/abs/2410.24164) 是结合预训练视觉语言表示与 Flow Matching 动作生成的实例。这里仅定位后续学习接口，不表示已复现这些论文的结构、训练或机器人结果。

### IV.3 VLA 与 World-Action Models 的联系应如何研究？

**Future Topics / 研究问题。** 本仓库的 WAM 路线关注“动作产生什么世界变化，以及这种预测能否帮助决策”。目前没有世界状态转移数据、世界模型或联合训练实现。

可先区分两类建模目标：策略产生动作，世界模型预测给定动作后的状态或观测变化。用 $s_r$ 表示某种状态表征， $a_r$ 表示动作， $\ell$ 表示指令：

$$
\text{策略目标： }p_\theta(A\mid o,\ell),\qquad
\text{世界预测目标： }p_\phi(s_{r+1}\mid s_r,a_r).
$$

这只是后续问题的形式化；真实环境可能部分可观测，需要历史或隐状态，不能假设一帧图像满足马尔可夫性。WAM 在本项目中是研究方向名称，不预设所有工作共享一种架构。

**待检验的假设：** 学会预测动作后果，可能帮助评估候选 Action Chunk。要支持这一判断，需要定义状态/视频预测目标，检查动作条件是否真正影响预测，并比较有无世界预测的闭环任务表现；单独的预测画面逼真或离线误差低都不充分。

### IV.4 Future Topics 如何转为正式知识？

| 后续主题 | 已有衔接点 | 转入正式内容前所需的证据 |
| --- | --- | --- |
| 长距离依赖、RoPE、KV Cache、变长 Mask | Part I 的因果建模与位置表示 | 对应实现、正确性检查及适用范围 |
| 预训练视觉语言表示与多模态训练 | Part II 的 Token 接口 | 真实监督/预训练来源，任务评估与消融 |
| DiT / Transformer 动作骨干 | Part III 的噪声/速度目标 | Token 与时间条件实现，并保持目标和采样一致 |
| 更可靠的生成比较与约束 | Part III 的 Policy | 共同测试集、预算记录、多种子、端点/多样性/约束评价 |
| VLA 闭环执行 | 已有表示与生成机制 | 动作定义、环境反馈和任务成功评价 |
| WAM 与动作后果预测 | IV.3 的问题定义 | 世界状态或视频建模、动作条件检验与决策评价 |

只迁移已经有证据的子问题，不因新增目录、提到论文或通过形状断言就将整个主题标为完成。

<a id="part-v"></a>
## Part V — Open Questions

`Open` 表示缺少回答该问题所需的证据；`Partially Resolved` 表示已有局部解释，仍缺关键验证；`Resolved` 表示在写明的范围内已回答，不表示所有相关研究都结束。问题编号保持稳定，后续在原条目更新，并保留状态演进记录。

| ID | 技术问题 | 状态 | 已有依据与下一步 |
| --- | --- | --- | --- |
| KB-Q01 | 当前实验能证明 Attention 对历史依赖的贡献吗？ | Open | 数字加一不需要历史；翻转尚无无 Attention 基线。需历史依赖任务、基线和消融，见 I.6 |
| KB-Q02 | 视觉与文本维度对齐后，何时才算学到语义对齐？ | Partially Resolved | 已解释形状、Padding、梯度通路及 Attention Mass 的计数基线；缺真实监督与图文任务评估，见 II.3–4、Projection Notes |
| KB-Q03 | 短 DDPM 链的先验不匹配对结果影响多大？ | Partially Resolved | 累计信号非零已确认；还需控制调度、训练预算和评估分布的比较，见 III.1 |
| KB-Q04 | DDPM Policy 与 FM Policy 在相同预算下如何取舍？ | Open | 当前预算和评价集不同，不能排序；需共同协议，见 III.7 |
| KB-Q05 | 条件输入是否硬性保证生成端点相等？ | Resolved | 当前采样器无端点投影；保存输出有误差。结论仅为“条件输入本身不构成硬约束”，见 III.7 |
| KB-Q06 | Action Chunk 长度与生成步数是否是同一时间轴？ | Resolved | 两个采样器每步更新整段 `[B,H,2]`；物理执行循环尚未实现，见 III.6 |
| KB-Q07 | 条纹分类能否验证对全局位置关系的理解？ | Partially Resolved | 局部 patch 已有分类线索；需依赖布局的数据和位置机制对照，见 II.2 |
| KB-Q08 | 如何把视觉/语言条件接入动作生成并形成闭环？ | Open | 各章模块独立，尚缺共享训练数据和执行环境，见 IV.1–2 |
| KB-Q09 | 世界预测能否帮助动作选择，如何排除只学视觉相关性？ | Open | 尚无 WAM 实现；需动作条件干预及闭环决策对照，见 IV.3 |
| KB-Q10 | 逐 Head 热力图能否说明各 Head 的语义分工或最终分类贡献？ | Partially Resolved | 已厘清权重、Value 聚合、最终 CLS 及色标的区别；缺 Head 消融、输入干预等归因证据，见 I.3、II.2 和 ViT Notes §5–7 |

### 问题演进记录

日期表示记录建立或修订的时间，不表示知识因时间较新而更可靠，也不冒充历史问答发生时间。

| 日期 | 问题 | 状态记录与依据 |
| --- | --- | --- |
| 2026-10-11 | KB-Q01、Q04、Q08、Q09 | 建档为 Open：现有材料明确缺少所需的消融、共同基准或系统实现 |
| 2026-10-11 | KB-Q02、Q03、Q07 | 建档为 Partially Resolved：已有接口断言、调度计算或任务分析，但未覆盖完整问题 |
| 2026-10-11 | KB-Q05、Q06 | 建档为 Resolved（限定当前实现）：依据采样代码、数据构造及已保存输出 |
| 2026-10-13 | KB-Q10 | 建档为 Partially Resolved：ViT Notes 与 `faff6d6` 的逐 Head 输出支持权重读取、拼接机制和色标解释；没有新增语义分工或分类贡献实验 |
| 2026-10-13 | KB-Q02 | 保持 Partially Resolved：Projection Notes 与 `9a85235` 输出补充共享权重的梯度形状、Mask 广播和 Attention Mass 的均匀基线；这些证据仍不能替代语义训练或任务评估 |

后续记录格式：`日期 / 问题 ID / 原状态 → 新状态（或状态不变） / 新证据 / 仍有的边界`。被否定的解释要记录修正原因；不要删掉问题或覆盖旧证据。新问题放入本表，核心答案融入相关 Part，避免末尾不断堆积独立总结。

<!-- Local evidence links: keep these synchronized when notebooks or notes move. -->
[nb-attention]: 01_Attention_Fundamentals/01_Attention_Fundamentals.ipynb
[note-attention]: 01_Attention_Fundamentals/01_Attention_Fundamentals.md
[nb-encdec]: 02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.ipynb
[note-encdec]: 02_Encoder_Decoder_Transformer/02_Encoder_Decoder_Transformer.md
[nb-vit]: 03_Vision_Transformers/01_ViT_From_Scratch.ipynb
[note-vit]: 03_Vision_Transformers/01_ViT_From_Scratch.md
[nb-projector]: 03_Vision_Transformers/02_Visual_Token_Projection.ipynb
[note-projector]: 03_Vision_Transformers/02_Visual_Token_Projection.md
[nb-ring]: 04_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/01_2D_Ring.ipynb
[note-ring]: 04_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/01_2D_Ring.md
[nb-gmm]: 04_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/02_2D_Gaussian_Mixture.ipynb
[note-gmm]: 04_Diffusion_and_Flow_Models/01_DDPM_Fundamentals/02_2D_Gaussian_Mixture.md
[nb-fm]: 04_Diffusion_and_Flow_Models/02_Flow_Matching/01_1D_Gaussian.ipynb
[note-fm]: 04_Diffusion_and_Flow_Models/02_Flow_Matching/01_1D_Gaussian.md
[nb-ddpm-policy]: 04_Diffusion_and_Flow_Models/03_Generative_Policies/01_DDPM_Policy.ipynb
[note-ddpm-policy]: 04_Diffusion_and_Flow_Models/03_Generative_Policies/01_DDPM_Policy.md
[nb-fm-policy]: 04_Diffusion_and_Flow_Models/03_Generative_Policies/02_Flow_Matching_Policy.ipynb
