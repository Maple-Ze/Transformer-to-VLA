# 01 — DDPM：二维圆环学习笔记

> 对应 Notebook：[01_2D_Ring.ipynb](01_2D_Ring.ipynb)。

## 一、从二维圆环开始，我们到底在学习什么？

**问题：为什么先生成二维点，而不是生成图片？** 生成问题的本质是学习数据分布。二维圆环可以让我们直接看见：训练数据集中在半径约为 1 的环上，随着加噪程度提升分布逐渐变得模糊，而反向采样需要重新恢复环形分布。它没有图像编码器等额外因素，适合专注理解 DDPM。

数据构造：

$$
\theta\sim U(0,2\pi),\quad \mathbf x_0=(\cos\theta,\sin\theta)+\boldsymbol\eta,
\quad\boldsymbol\eta\sim\mathcal N(0,0.05^2I).
$$

代码位置：`sample_data()`，输出 `x0.shape == [B, 2]`。

## 二、`beta`、`alpha`、`alpha_bar` 在做什么？

**问题：`beta = torch.linspace(1e-4, 0.02, T)` 表示什么？** `beta[t]` 对应数学公式中第 `t+1` 个加噪步的噪声方差系数。代码下标从 0 开始，公式中的干净样本记为 $\mathbf x_0$ ，加噪步从 1 开始。

$$
q(\mathbf x_t\mid\mathbf x_{t-1})=
\mathcal N(\sqrt{\alpha_t}\mathbf x_{t-1},\beta_t I),
\quad\alpha_t=1-\beta_t.
$$

`alpha[t]` 控制**每一步**保留信号的程度；`alpha_bar[t] = alpha[0] * ... * alpha[t]` 控制**累计到当前步**保留多少原始信号：

$$
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

**问题：为什么不循环加噪 100 次，而是一步得到任意 `x_t`？** 各步噪声相互独立，且独立于干净样本。给定 `x0` 后，独立高斯噪声的线性组合仍为高斯，方差按系数的平方相加，因此可以把前面所有步骤合并：

$$
\boxed{\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+
\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon},
\qquad\boldsymbol\epsilon\sim\mathcal N(0,I).
$$

这就是 `q_sample(x0, t, noise)`；`t` 可随机取样，训练时不必真的走完前向链。平方根系数来自 $\mathrm{Cov}(cX)=c^2\mathrm{Cov}(X)$ 。推导见 [Notebook §3](01_2D_Ring.ipynb)。分别独立生成多个时刻的边际样本，并不等于获得同一条前向链的轨迹。

## 三、`T=100` 是什么意思？如何选择？

**问题：`T=100` 是否表示加噪 100 步，去噪也执行 100 步？** 在本 Notebook 的传统 DDPM 设置里，可以这样理解：定义长度为 100 的噪声调度；训练对其中的随机时间步构造加噪样本，采样时从最后一步逐步迭代到第一步。**一次训练迭代并不是先跑 100 步加噪。**

**问题：`T` 越大越好吗？** 不一定。它影响离散噪声链和反向采样的计算量；还需要连同 `beta` 调度一起看。`T=100` 是教学实验的选择，不是普适最优值。改变 `T` 要重新构造 `beta/alpha/alpha_bar`，然后重新训练/评估。

**当前配置的近似**：`T=100` 时，`alpha_bar[-1] ≈ 0.364`，信号缩放系数仍约为 `0.603`，末端并非纯标准高斯。因此用标准高斯初始化反向采样存在先验不匹配。相同线性端点下，`T=1000` 得到 `alpha_bar[-1] ≈ 4.04e-5`，更符合这一近似；比较时需重新训练。参见 [DDPM 原论文](https://arxiv.org/abs/2006.11239)。

## 四、为什么学习预测噪声？

训练过程先随机取原始样本 `x0`、时间步 `t` 和噪声 `epsilon`，构造 `xt`；神经网络输入 `(xt,t)`，输出 `epsilon_pred`：

$$
\mathcal L_\epsilon =
\mathbb E\left[\lVert\boldsymbol\epsilon-
\boldsymbol\epsilon_\theta(\mathbf x_t,t)\rVert^2\right].
$$

**问题：为什么使用 MSE？和最大似然有什么关系？** 当反向方差固定且为正时，ELBO 的中间 KL 项可写成带时间权重的噪声 MSE 加上与模型参数无关的项。本例去掉时间权重，使用简化目标；它不是完整 ELBO，也不等同于精确最大似然优化。条件与系数见 [高斯混合实验 §6](02_2D_Gaussian_Mixture.ipynb)。

**问题：预测噪声就能直接恢复原始样本吗？** 由前向公式可以构造估计：

$$
\hat{\mathbf x}_{0}=
\frac{\mathbf x_t-\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon_\theta(\mathbf x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

平方回归的总体最优解是给定 $(\mathbf x_t,t)$ 的条件平均噪声，不是每条训练样本唯一可识别的那份噪声。因此，这个估计不保证恢复原样本，也不是下一步的 $\mathbf x_{t-1}$ 。

**问题：为什么还输入时间步？** 同一个位置 `xt` 在不同噪声级别下意味着不同的去噪任务。当前网络输入 `[x,y,t/T]`，是最简单的时间条件方式，不需要 Transformer。

## 五、反向采样在计算什么？

用预测噪声估计反向均值：

$$
\boldsymbol\mu_\theta(\mathbf x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\left(\mathbf x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\boldsymbol\epsilon_\theta(\mathbf x_t,t)\right).
$$

本 Notebook 从 `torch.randn(n_samples, 2)` 开始，每个反向步先预测噪声，再更新样本；除最终一步外还会加入随机项。生成与训练不同：**生成没有给定 `x0`，只知道当前 `xt`，必须逐步迭代。**

对数学时间步 $t\geq2$ ，本例使用 $\sigma_t^2=\beta_t$ ，下一例使用 $\sigma_t^2=\tilde\beta_t$ 。二者是不同的固定方差选择；后者来自已知干净样本时的前向条件后验 $q(\mathbf x_{t-1}\mid\mathbf x_t,\mathbf x_0)$ ，不能将它当作未知 $\mathbf x_0$ 时的真实反向分布，也不能据此断言生成质量更高。两个实验在数学 $t=1$ 时都只返回均值，不再加噪；代码索引对应 `step=0`。

均值公式来自将 $\hat{\mathbf x}_{0}$ 代入条件后验均值，合并系数后得到噪声参数化，详见 [Notebook §6](01_2D_Ring.ipynb)。

**问题：生成图中的点是在追踪机器人运动吗？** 不是。每个点代表一个随机生成样本，整个点云逐步恢复为圆环概率分布。它与后续轨迹生成中按时间顺序排列的 waypoint 不是一回事。

## 六、对照代码和自测

- `sample_data`：圆环数据为什么用 `cos/sin`？`0.05` 改大时会发生什么？
- `q_sample`：验证 `t` 张量形状如何广播到 `[B,2]`。
- `NoisePredictor`：为什么输出维数仍然是 2？
- `sample_ddpm`：为什么训练可随机时间步，采样却通常要按顺序？
- 试验：比较 `T=50/100/200`、`TRAIN_STEPS=300/1000/3000`、不同环厚度；不要只看 loss，也要看最终二维分布。

**与机器人动作生成的联系**：今天的干净样本是 `x0 ∈ R²`；以后把它换成一整段动作 `A0 ∈ R^{H×d_a}`，前向加噪、时间条件噪声预测和反向生成的总体思路仍然可用。
