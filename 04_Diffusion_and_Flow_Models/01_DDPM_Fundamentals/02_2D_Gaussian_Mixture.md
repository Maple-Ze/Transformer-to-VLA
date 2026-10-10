# 02 — DDPM：二维多高斯聚类学习笔记

> 对应 `02_2D_Gaussian_Mixture.ipynb`。根据《重要：：diffusion》中从圆环推进到复杂分布的提问整理；包含后续针对时间嵌入、后验方差、采样参数的细节追问。非逐字转录。

## 一、为什么要从圆环推进到多聚类？

圆环能证明模型可以学习非高斯形状；**多个分离的簇**更容易说明多模态概率分布。当前 Notebook 使用八个二维中心，围绕每个中心添加标准差 `0.12` 的高斯噪声：

$$
p(\mathbf x)=\frac18\sum_{k=1}^{8}\mathcal N(\mathbf x;\boldsymbol\mu_k,0.12^2I).
$$

**问题：模型是不是先决定属于哪一簇再分别生成？** 当前代码没有显式分类器。模型通过一个噪声预测网络联合学习整体分布；采样可能落入不同峰值区域。这可以类比机器人从同一观测条件生成若干不同的可行动作方案，但**学习多峰点分布不等于已经能解决轨迹安全问题**。

## 二、`T=200` 与 `beta=torch.linspace(1e-4,0.02,T)` 如何确定？

**问题：为什么使用 `T=200`？** 这是教学用离散扩散步数，不能只看数字判断模型优劣。步数越多反向采样通常越慢，也需要配合噪声强度安排。

**问题：`1e-4` 和 `0.02` 是怎么定的？** 这是经典线性 beta schedule 的常见示例超参数：每一步的 `beta` 由小到大变化。不是依据这个八簇数据集推导出的唯一正确值。判断调度是否适当，应结合累计信号比例 `alpha_bar[-1]`、末端分布是否足够接近标准高斯和采样质量。

$$
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

改变 `T` 却保留相同的 beta 起止值，会同时改变总的累计噪声；因此比较不同 `T` 时要明确控制变量。

## 三、`alpha_bar_prev` 为什么要在开头拼一个 1？

对应代码：

```python
alpha_bar_prev = torch.cat([
    torch.ones(1, device=device),
    alpha_bar[:-1]
])
```

**问题：`alpha_bar[:-1]` 不就已经有之前的值了吗？** 它只包含 `[bar_alpha_0, ..., bar_alpha_{T-2}]`。为了让数组 `alpha_bar_prev[t]` 对齐“当前 `t` 对应的前一步累计乘积”，需要在前面补入**空乘积为 1**；即 `alpha_bar_prev[0] = 1`。结果长度仍然是 `T`。

**问题：这与之前的圆环 DDPM 代码有何不同？** 圆环案例可直接用 `beta[t]` 作为反向方差的一种固定选择；这里显式计算前向过程的条件后验方差，便于理解另一种常见实现。

## 四、`posterior_variance` 到底是什么？

前向扩散在给定干净样本 `x0` 和当前带噪样本 `xt` 后，有解析后验：

$$
q(\mathbf x_{t-1}\mid\mathbf x_t,\mathbf x_0)
=\mathcal N(\widetilde{\boldsymbol\mu}_t,\widetilde\beta_t I).
$$

其中：

$$
\boxed{\widetilde\beta_t=
\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}.
$$

**问题：为什么 `posterior_variance` 与 `beta` 不一样？** `beta_t` 是前向单步加噪方差；`tilde_beta_t` 是额外知道 `x0` 时前一步的条件后验方差。二者意义不同。实际反向模型的固定方差可以采用不同设计；这份代码选后者。

**注意代码索引**：Python `t=0` 对应数学上首次加噪步，此时 `alpha_bar_prev[0]=1`，后验方差为 0，最后一次更新不用再采样新噪声。

## 五、为什么改用 Sinusoidal Time Embedding？

圆环里直接拼入 `t/T`。多聚类实验把离散时间映射成多个频率的正余弦分量：

$$
\operatorname{TE}(t)=
[\sin(t\omega_1),\ldots,\sin(t\omega_m),
\cos(t\omega_1),\ldots,\cos(t\omega_m)].
$$

**问题：`-math.log(10000)` 是什么意思？** 常见实现写作：

```python
frequencies = torch.exp(
    torch.arange(half, device=t.device)
    * (-math.log(10000) / max(half - 1, 1))
)
```

它通过 `exp(log(...))` 生成跨越多个数量级的**几何级数频率**，让时间编码同时具有快变化和慢变化的分量。`10000` 是频率跨度的设计常数，并非 DDPM 数学推导规定的唯一数字。这里借用 Transformer 正弦位置编码思想，但**输入是扩散时间步而非 token 位置**。

**问题：32 维时间嵌入怎么进入模型？** `SinusoidalTimeEmbedding(dim=32)` 给出 32 个时间特征；经过线性层，与二维点的特征融合，输出最终 2 维预测噪声。时间嵌入维数与输出噪声维数无须相同。

## 六、反向采样为什么仍然逐步进行？

训练时随机 `t`，使用闭式公式取得 `xt`，优化噪声预测 MSE；生成时从 `N(0,I)` 出发，用预测噪声计算每一步反向均值，并根据 `posterior_variance` 加入随机扰动。反向过程不是简单地“把训练噪声减掉”，因为生成时没有真值噪声、也没有真实 `x0`。

## 七、对照代码和实验检查

- `CENTERS` / `sample_data`：八簇中心与每簇标准差分别控制什么？
- `alpha_bar_prev`：打印前 3 项，确认索引对齐。
- `posterior_variance`：检查 `t=0` 为 0、其余项非负。
- `SinusoidalTimeEmbedding`：观察不同 `t` 的编码，比较单标量时间输入。
- `sample_ddpm`：分别观察中间点云，何时开始显现八个模式？
- 试验：2、4、8 簇；簇标准差 `0.12/0.3`；不同 `T`、训练步数。

**连接 Diffusion Policy**：多峰分布意味着一个输入条件下可以有多种合理输出。这是条件轨迹生成的有用直觉，但真实策略还需要条件一致性、动力学和安全性评价。
