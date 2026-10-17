# 01 — DDPM：二维圆环学习笔记

[项目首页](../../README.md) · [实验 Notebook](01_2D_Ring.ipynb) · [知识库](../../KNOWLEDGE_BASE.md#part-iii)

本笔记围绕二维圆环实验，按前向扩散、噪声预测和反向采样解释公式与代码；数学证明集中在附录。

## 0. 总览：我们究竟在训练什么？

二维点云可以直接显示分布随加噪与去噪的变化，便于聚焦 DDPM 本身。训练数据在单位圆上采样，再加入小幅高斯扰动：

$$
\varphi\sim U(0,2\pi),\qquad
\mathbf x_0=(\cos\varphi,\sin\varphi)^\top+\boldsymbol\eta,
\qquad\boldsymbol\eta\sim\mathcal N(\mathbf 0,0.05^2I).
$$

`sample_data(B)` 返回形状为 `[B,2]` 的点集。这里 **一个样本是一个二维点，不是机器人沿着圆环运动的一条轨迹**。目标是训练一个模型，从随机噪声中生成一批新的点，使新点整体服从近似的圆环分布。

DDPM 分为三件事：

1. **前向扩散（已知，无须训练）**：人为把真实数据逐步变成带噪数据。
2. **噪声预测（需要训练）**：神经网络学习在不同噪声程度下预测噪声。
3. **反向采样（生成阶段）**：从随机噪声出发，利用预测网络一步步恢复数据分布。

**索引约定：** 数学上的扩散时间 `t=1,...,T`；Python 数组下标 `step=0,...,T-1`。`step=0` 是数学上第一次加噪，**不是干净的** $x_0$ 。后文所有公式用数学时间步，代码中的 `t` 和 `step` 都使用 Python 下标。本实验中 $I$ 是 $2\times2$ 单位矩阵； $\varphi$ 是采样角度，网络下标 $\theta$ 表示可训练参数。

---

## 1. 噪声调度：`beta`、`alpha`、`alpha_bar`

### 1.1 单步加噪为什么这么定义？

设 $0\lt\beta_t\lt1$ ，各步噪声彼此独立，并独立于干净样本 $x_0$ 。DDPM 的前向更新为：

$$
\boxed{\mathbf x_t=\sqrt{\alpha_t}\mathbf x_{t-1}+\sqrt{\beta_t}\boldsymbol\epsilon_t},
\qquad \boldsymbol\epsilon_t\sim\mathcal N(\mathbf0,I),
\qquad\alpha_t=1-\beta_t.
$$

- $\beta_t$ ：**这一步**加入的噪声方差参数。
- $\alpha_t$ ：**这一步**上一状态信号的平方系数。
- $\sqrt{\beta_t}$ 、 $\sqrt{\alpha_t}$ ：实际乘在向量上的幅值系数。

**Lemma 1（协方差缩放与独立噪声相加）**： $\mathrm{Cov}(cX)=c^2\mathrm{Cov}(X)$ ；独立随机向量相加时，协方差相加。因此要让新噪声的协方差为 $\beta_t I$ ，应乘 $\sqrt{\beta_t}$ ，而不是 $\beta_t$ 。证明见[附录 A.1](#lemma-1)。

$\alpha_t+\beta_t=1$ 使单位协方差在单步更新下保持不变，但不意味着任意数据的协方差都是 $I$ ，也不会自动把圆环数据标准化。

为什么先定义 `beta` 而不是 `alpha`？**只是参数化习惯**：从“每一步加多少噪声”更方便设计 schedule。直接设计 `alpha` 也完全可行，只须满足 $\alpha_t=1-\beta_t$ 。

### 1.2 累计信号保留量

定义：

$$
\boxed{\bar\alpha_t=\prod_{s=1}^{t}\alpha_s},\qquad\bar\alpha_0=1.
$$

对应代码：

```python
T = 100
beta = torch.linspace(1e-4, 0.02, T, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
```

`torch.cumprod` 是 **累积乘积**。例如 `torch.cumprod(torch.tensor([0.9, 0.8, 0.7]), dim=0)` 得到 `[0.9, 0.72, 0.504]`。

| 变量 | 表示什么 | 代码索引的含义 |
|---|---|---|
| `beta[k]` | 第 $k+1$ 步的噪声方差 | $\beta_{k+1}$ |
| `alpha[k]` | 第 $k+1$ 步的单步信号系数的平方 | $\alpha_{k+1}$ |
| `alpha_bar[k]` | 从 $x_0$ 到第 $k+1$ 步的累计系数的平方 | $\bar\alpha_{k+1}$ |

不要说 $\bar\alpha_t$ 本身是“原信号的幅值比例”；**幅值乘子是 $\sqrt{\bar\alpha_t}$**。

---

## 2. 前向闭式采样：为什么可以跳过中间步骤？

如果依次执行加噪，要算 $x_1,x_2,\ldots,x_t$ 。但训练时只需指定某个随机时间步，直接得到 $x_t$ ：

$$
\boxed{\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+
\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon},
\quad\boldsymbol\epsilon\sim\mathcal N(\mathbf0,I).
$$

**Lemma 2（独立高斯线性组合）**：独立高斯向量的线性组合仍为高斯；其均值按线性组合计算、协方差按各系数平方求和。**Lemma 3（前向闭式采样）**：在前向链的独立高斯噪声假设下，给定 $x_0$ ，多次逐步加噪所得 $x_t$ 的**条件分布**等于上式直接采样的条件分布。证明见[附录 A.2](#lemma-2)–[A.3](#lemma-3)。

闭式公式中的 $\epsilon$ 是累计噪声的等效表示，不是单步更新中的最后一份 $\epsilon_t$ 。闭式采样与逐步加噪所得的条件边际同分布，不要求两次独立采样的数值相同。若独立为每个 $t$ 采样一个噪声，也不会自动得到同一条前向链的联合轨迹。

对应代码：

```python
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    scale_x = alpha_bar[t].sqrt().unsqueeze(-1)
    scale_noise = (1.0 - alpha_bar[t]).sqrt().unsqueeze(-1)
    return scale_x * x0 + scale_noise * noise
```

- `torch.randn_like(x0)`：生成与 `x0` 同形状、各分量独立的标准高斯噪声 $\epsilon\sim\mathcal N(0,I)$ 。
- `alpha_bar[t]`：逐样本读取对应时间步的累计系数。
- `unsqueeze(-1)`：将 `[B]` 变成 `[B,1]`。
- `[B,1] * [B,2]`：**广播（broadcasting）**，每个样本的一个系数同时乘在 $x,y$ 两个坐标上。

### 2.1 为什么 `torch.full()`？为什么是 `zip()`？

绘图：

```python
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, t_value in zip(axes, torch.linspace(0, T-1, 5).round().long().tolist()):
    t = torch.full((len(x0),), t_value, device=device, dtype=torch.long)
    xt = q_sample(x0, t)
```

`torch.linspace(0,99,5)` 选出五个分布在全时域的索引；`.round().long().tolist()` 把它们转换成整数列表；`zip` 把五个时间索引与五个子图一一配对；`torch.full((1000,), t_value)` 创建长度为 1000 的同一时间步索引，让这批二维点都在同一个噪声等级下被采样。`q_sample` 也支持训练阶段**每个样本拥有不同的时间步**。

### 2.2 `T=100` 是什么，为什么末端未必是纯高斯？

这是定义 100 个加噪步及对应的反向采样步；**一次训练迭代不会依次执行 100 次前向加噪**。在当前 `beta` 端点固定的线性 schedule 下：

- `T=100`： $\bar\alpha_T\approx0.364$ ，原信号幅值系数仍约为 $0.603$ ；
- `T=1000`： $\bar\alpha_T\approx4.04\times10^{-5}$ ，原信号影响才更接近消失。

因此此 Notebook 从 $\mathcal N(0,I)$ 开始反向生成是有**先验不匹配**的教学近似。 $T$ 不是越大越好：它与 schedule、训练数据和采样成本共同决定效果。调整 $T$ 后应重建 schedule、重新训练和评估；不能只改一个整数便比较模型优劣。

---

## 3. 预测噪声：网络究竟学到了什么？

训练时知道干净数据 $x_0$ ，自己随机抽取 $t$ 和标准高斯噪声 $\epsilon$ ，按前向公式合成 $x_t$ ，让网络预测**注入的噪声**：

$$
\boxed{\mathcal L_{\rm simple}=\mathbb E_{x_0,t,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right]}
$$

代码：

```python
x0 = sample_data(BATCH_SIZE)
t = torch.randint(0, T, (BATCH_SIZE,), device=device)
noise = torch.randn_like(x0)
x_t = q_sample(x0, t, noise)
prediction = model(x_t, t)
loss = torch.nn.functional.mse_loss(prediction, noise)
```

这里 `mse_loss` 默认对元素取平均，相比上面的向量平方范数期望只是差一个固定的坐标维数因子；两者最优解相同。

### 3.1 为什么输入还要包含 `t`？

同一个二维坐标，在数学时间 $t=1$ 和 $t=90$ 下对应不同的噪声水平。网络需要时间信息才能知道自己在做哪个去噪任务。

本实验把代码中的时间下标作为额外特征：

```python
def forward(self, x_t, t):
    t_normalized = t.float().unsqueeze(-1) / T
    return self.net(torch.cat([x_t, t_normalized], dim=-1))
```

维度是 `[B,2]` 和 `[B,1]` **拼接（cat）** 成 `[B,3]`，MLP 输出 `[B,2]` 的噪声。**这不是广播，而是特征拼接**。这里代码下标为 $k=0,\ldots,T-1$ ，实际时间特征是 $k/T$ ，对应数学时间的 $(t-1)/T$ ；当前取值范围是 $0$ 到 $0.99$ 。MLP 结构为 `3 → 128 → 128 → 2`，两个隐藏层均使用 ReLU。

### 3.2 为什么 MSE 预测噪声可以估计 `x0`？

把前向闭式采样公式移项：

$$
\boxed{\hat x_0(x_t,t)=\frac{x_t-
\sqrt{1-\bar\alpha_t} \epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}}
$$

**Lemma 4（MSE 的最优解）**：最小化 $\mathbb E[\lVert Y-f(X)\rVert^2]$ 的总体最优预测是 $f^{\ast}(X)=\mathbb E[Y\mid X]$ 。因此对于预测噪声， $\epsilon_\theta^{\ast}(x_t,t)=\mathbb E[\epsilon\mid x_t,t]$ ；代入上式可得 $\hat x_0=\mathbb E[x_0\mid x_t,t]$ 。证明见[附录 A.4](#lemma-4)。

这解释了两个问题：

- **为什么不能把高噪声时刻的一次 $\hat x_0$ 预测当作完整的分布采样？** 在总体 MSE 最优预测器下，它是条件均值；有限模型只是近似这一目标。该均值可能是多个合理原始结果的平均值，并非某个真实模式。用一个独立的一维例子说明： $x_0\in\lbrace-1,1\rbrace$ 且后验两者等概率，条件均值为 0，但训练数据没有 0。
- **直接预测 $x_0$ 就必然丧失多模态吗？** 不一定。预测 $x_0$ 也可以作为逐步扩散采样的网络参数化；容易均值化的是**将 MSE 条件均值一次性作为最终生成输出**。预测 $\epsilon$ 同样受条件均值性质约束，多模态能力主要来自学习整个反向生成过程，而不是“预测噪声”这三个字。

### 3.3 MSE 与最大似然的关系

原始 DDPM 源于概率模型的变分下界（ELBO）。**Lemma 7（固定高斯方差时 KL 的均值项）** 说明：某些中间 KL 项可化为**带时间权重**的噪声预测平方误差，再加与网络参数无关的常数。本实验只使用**不带时间权重**的 `mse_loss`：它既不是完整 ELBO，也不等价于直接最大化精确对数似然。见[附录 A.7](#lemma-7)。

---

## 4. 反向采样：为什么需要均值，还要随机噪声？

训练结束时网络只能估计给定 $(x_t,t)$ 的噪声。生成时**没有给定真实 $x_0$**，因此从高斯随机样本开始：

$$
x_T\sim\mathcal N(0,I),\qquad
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0.
$$

对数学时间步 $t\ge2$ ，本实验使用高斯模型表示反向转移：

$$
\boxed{p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\sigma_t^2 I)}.
$$

- $\mu_\theta$ ：**模型反向分布**的条件均值，不保证等于真实反向分布的均值，也不是完整的随机采样结果。
- $\sigma_t^2$ ：选定的反向方差；具体如何设置与均值公式是不同问题。
- $z\sim\mathcal N(0,I)$ ：反向采样时的新增随机性。

本实验使用：

$$
\boxed{\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)\right)}
$$

$$
\boxed{x_{t-1}=\mu_\theta(x_t,t)+\sqrt{\beta_t} z\quad(t\ge2)},
\qquad x_0=\mu_\theta(x_1,1)\ \text{（最后一步）}.
$$

### 4.1 反向均值为何是这个式子？

**Lemma 5（贝叶斯分解 + 马尔可夫性质）**：对 $t\ge2$ ，

$$
q(x_{t-1}\mid x_t,x_0)
=\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
\propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).
$$

分母只依赖**已知条件** $(x_t,x_0)$ ，不依赖正在求分布的变量 $x_{t-1}$ ，因此可以并入归一化常数。**不是说分母等于 1，也不是直接删掉后继续写等号。** 证明见[附录 A.5](#lemma-5)。

**Lemma 6（线性高斯后验）**：对于 $t\ge2$ ，上述分布为高斯，且

$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I),
$$

$$
\boxed{\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t},
\qquad
\boxed{\tilde\beta_t=\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}.
$$

证明见[附录 A.6](#lemma-6)：对两个高斯密度的乘积展开平方并配方。

把前向公式解出的 $x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$ 代入 $\tilde\mu_t$ ，利用 $\alpha_t+\beta_t=1$ 和 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$ 化简，即得：

$$
\tilde\mu_t=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right).
$$

生成时不知道真正的 $\epsilon$ ，于是由网络预测 $\epsilon_\theta(x_t,t)$ 代替，得到前述**模型的** $\mu_\theta$ 。注意：已知真实 $x_0$ 的 $q(x_{t-1}\mid x_t,x_0)$ ，并不等于生成所需的 $q(x_{t-1}\mid x_t)$ ；真实边际反向分布还要对可能的 $x_0$ 积分，一般不能认定恰为同一个高斯。

### 4.2 本实验为什么选用 $\beta_t$ 作为反向方差？

代码在 $t\ge2$ 固定 $\sigma_t^2=\beta_t$ ，对应 `mean + sqrt(beta[step])*randn_like(x)`。这是反向模型的设计选择，不能仅由“使用标准高斯随机数”推出；标准高斯乘以 $\sqrt{\beta_t}$ 后，扰动协方差才是 $\beta_t I$ 。

推导中的 $\tilde\beta_t$ 则是额外知道真实 $x_0$ 时， $q(x_{t-1}\mid x_t,x_0)$ 的条件后验方差。在当前 $0\lt\beta_t\lt1$ 的条件下， $t\ge2$ 时有 $0\lt\tilde\beta_t\lt\beta_t$ 。二者对应不同的条件和用途；本节代码不计算或使用 $\tilde\beta_t$ 作为采样方差，推导它是为了说明反向均值的来源。

数学 $t=1$ （代码 `step=0`）时，给定真实 $x_0$ 的条件后验退化为集中在该点的 Dirac 分布。代码则把模型均值作为生成结果，不再添加新噪声；这并不保证输出等于某个真实训练样本。此处不能套用要求方差严格为正的高斯配方或 KL 公式。

### 4.3 对应代码逐行解释

```python
@torch.no_grad()
def sample_ddpm(n_samples=2000):
    model.eval()
    x = torch.randn(n_samples, 2, device=device)
    for step in reversed(range(T)):
        t = torch.full((n_samples,), step, device=device, dtype=torch.long)
        noise_pred = model(x, t)
        mean = (x - beta[step] / torch.sqrt(1 - alpha_bar[step]) * noise_pred) / torch.sqrt(alpha[step])
        if step > 0:
            x = mean + torch.sqrt(beta[step]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

这是采样核心的简化摘录；实际 `sample_ddpm` 还保存中间点云，返回 `(x.cpu().numpy(), snapshots)`。

每次 `step=k` 更新前的状态对应数学 $x_{k+1}$ ，更新后对应 $x_k$ 。因此正向图的 `t=t_value+1` 与反向快照的 `t=step` 都是在标注各自展示状态的数学时间；每幅图显示整批样本的分布，不是机器人运动轨迹。

| 代码 | 意义 |
|---|---|
| `torch.randn(n_samples, 2)` | 从简单高斯先验生成初始样本 |
| `reversed(range(T))` | 时间步从高噪声到低噪声倒序执行 |
| `model(x, t)` | 预测当前噪声 |
| `mean = ...` | 用噪声预测计算反向均值 $\mu_\theta$ |
| `sqrt(beta[step])*randn_like(x)` | 采样方差为 $\beta_t$ 的随机扰动 |
| `step == 0` | 最终输出，按本实现不再加噪声 |
| `@torch.no_grad()` | 采样不构建梯度计算图，与是否随机采样无关 |

**为什么训练可跳步、生成却要迭代？** 训练知道 $x_0$ ，所以使用前向闭式公式直接构造任意噪声等级；生成不知道 $x_0$ ，使用学习到的局部反向转移，所以这里按相邻时间步迭代。这里描述的是当前采样器；高噪声阶段的一次 $\hat x_0$ 预测通常不能替代这条反向生成链。

---

## 5. 用三个层次理解整个 Notebook

**机制主线**

- 加噪：`x0 → xt`，`beta` 控制单步噪声，`alpha_bar` 控制累计信号。
- 训练：网络预测人造噪声，用 MSE 学习去噪信息。
- 生成：从高斯先验出发，网络预测噪声，计算均值，再按选定方差采样下一步。

**代码对应**

- `cumprod` 是累计乘积；`randn_like` 是独立标准高斯噪声。
- `torch.full` 创建一批时间步；`unsqueeze` 为广播准备维度。
- `torch.cat([x_t, t_normalized], dim=-1)` 是特征拼接，不是广播。
- `sample_ddpm` 的反向均值与随机项是两种不同的计算。

**推导索引**

| 关键结论 | 数学依据 |
|---|---|
| 为何有平方根系数 | Lemma 1，协方差缩放 |
| 为何能一步得到 $x_t$ | Lemma 2–3，独立高斯合并 |
| 为何 MSE 输出条件均值 | Lemma 4，平方损失最优性 |
| 为何可用已知前向分布推后验 | Lemma 5，贝叶斯 + 马尔可夫 |
| 为何反向均值出现复杂系数 | Lemma 6，高斯乘积配方 |
| 为何 MSE 与 ELBO 有联系 | Lemma 7，固定协方差高斯 KL |

### 5.1 自测与实验

1. `sample_data` 中圆环为什么用 `cos/sin`？把数据噪声标准差从 `0.05` 改为 `0.2` 会怎样？
2. `T=100` 和 `TRAIN_STEPS=3000` 分别是什么？为何二者不能混为一谈？
3. 为什么 `q_sample` 可以一次采样到某个 $x_t$ ，但单独分别采样多个时刻不代表同一条扩散路径？
4. `alpha_bar[t].sqrt().unsqueeze(-1)` 怎样从 `[B]` 广播到 `[B,2]`？
5. 为什么噪声预测网络接收 `t`？网络输出为什么依然是二维？
6. 已知 $\hat x_0$ ，为什么不在最高噪声时刻直接将它当作最终生成样本？
7. 为什么本节采样使用 $\beta_t$ ，而推导得到的条件后验方差是 $\tilde\beta_t$ ？两者的条件有何区别？
8. 调整 `T=50/100/200`、`TRAIN_STEPS=300/1000/3000` 后，观察生成点云而不是只比较 loss；注意 `T` 改动会连带改变总噪声量。

---

## 附录 A — Lemma 与证明

<a id="lemma-1"></a>
### A.1 Lemma 1：协方差缩放与独立随机向量之和

**命题。** 对具有有限二阶矩的随机向量 $X$ 、标量 $c$ ，

$$
\mathrm{Cov}(cX)=c^2\mathrm{Cov}(X).
$$

若 $X,Y$ 具有有限二阶矩、维度相同且独立，则 $\mathrm{Cov}(X+Y)=\mathrm{Cov}(X)+\mathrm{Cov}(Y)$ 。

**证明。** 按定义，

$$
\mathrm{Cov}(cX)=\mathbb E[(cX-c\mathbb EX)(cX-c\mathbb EX)^\top]
=c^2\mathrm{Cov}(X).
$$

展开 $\mathrm{Cov}(X+Y)$ 有两个自身协方差项和两个交叉项；独立性使两个交叉项都为零，得证。因此 $\mathrm{Cov}(\sqrt{\beta_t}\epsilon)=\beta_t I$ 。 $\square$

<a id="lemma-2"></a>
### A.2 Lemma 2：独立高斯变量的线性组合

**命题。** 若 $X\sim\mathcal N(m_X,\Sigma_X)$ 、 $Y\sim\mathcal N(m_Y,\Sigma_Y)$ 且相互独立，则

$$
aX+bY\sim\mathcal N(am_X+bm_Y,a^2\Sigma_X+b^2\Sigma_Y).
$$

**证明（简述）。** 独立高斯的联合向量仍是高斯，任何线性变换后的向量仍是高斯；均值的线性性与 Lemma 1 决定新均值、新协方差。可以用高斯特征函数或矩母函数直接验证。 $\square$

<a id="lemma-3"></a>
### A.3 Lemma 3：前向闭式采样

**命题。** 对所有 $t\ge1$ ，

$$
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I).
$$

**证明（归纳法）。** $t=1$ 时成立，因为 $\bar\alpha_1=\alpha_1$ ， $1-\bar\alpha_1=\beta_1$ 。假设 $t-1$ 时成立，代入单步加噪：

$$
\begin{aligned}
x_t&=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t\\
&=\sqrt{\alpha_t\bar\alpha_{t-1}}x_0
+\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\epsilon'
+\sqrt{\beta_t}\epsilon_t.
\end{aligned}
$$

其中 $\epsilon^{\prime}$ 来自此前的噪声，与新抽取的 $\epsilon_t$ 独立。由 Lemma 2 合并最后两项；其总协方差是

$$
[\alpha_t(1-\bar\alpha_{t-1})+\beta_t]I
=(1-\bar\alpha_t)I.
$$

所以在给定 $x_0$ 下与直接采样
$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ 同分布。 $\square$

<a id="lemma-4"></a>
### A.4 Lemma 4：MSE 回归的条件均值最优性

**命题。** 对平方可积随机变量 $X,Y$ ，最小化 $\mathbb E\lVert Y-f(X)\rVert^2$ 的最优函数（几乎处处）为 $f^{\ast}(X)=\mathbb E[Y\mid X]$ 。

**证明。** 记 $m(X)=\mathbb E[Y\mid X]$ ，写成

$$
Y-f(X)=[Y-m(X)]+[m(X)-f(X)].
$$

展开平方后，交叉项期望为 0，因为 $\mathbb E[Y-m(X)\mid X]=0$ ，从而

$$
\mathbb E\lVert Y-f(X)\rVert^2
=\mathbb E\lVert Y-m(X)\rVert^2+\mathbb E\lVert m(X)-f(X)\rVert^2.
$$

第二项非负，在 $f=m$ 时为零，得证。将 $Y=\epsilon$ 、 $X=(x_t,t)$ 代入，再对前向闭式公式取条件期望，可得 $\hat x_0=\mathbb E[x_0\mid x_t,t]$ （假设采用总体最优噪声预测器）。 $\square$

<a id="lemma-5"></a>
### A.5 Lemma 5：条件贝叶斯公式与马尔可夫简化

**命题。** 在 $q(b\mid c)\gt0$ 的位置，对条件密度有

$$
q(a\mid b,c)=\frac{q(b\mid a,c)q(a\mid c)}{q(b\mid c)}.
$$

若 $b$ 在已知 $a$ 后与 $c$ 条件独立，则 $q(b\mid a,c)=q(b\mid a)$ 。

**证明。** 由条件密度定义：

$$
q(a\mid b,c)=\frac{q(a,b\mid c)}{q(b\mid c)},\qquad
q(a,b\mid c)=q(b\mid a,c)q(a\mid c).
$$

将第二式代入第一式即得。DDPM 的前向过程满足 $x_t\perp x_0\mid x_{t-1}$ （一阶马尔可夫），故可简化。由于分母对**待求变量** $x_{t-1}$ 是常数，写成 $\propto$ ，事后由密度积分为 1 决定归一化系数。 $\square$

<a id="lemma-6"></a>
### A.6 Lemma 6：前向条件后验为高斯及其参数

**命题。** 对数学时间 $t\ge2$ ，

$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I),
$$

其中

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
\qquad
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
$$

**证明。** 令 $y=x_{t-1}$ 。由 Lemma 5 得：

$$
\begin{aligned}
q(y\mid x_t,x_0)
&\propto q(x_t\mid y)q(y\mid x_0)\\
&\propto\exp\left[-\frac12\left(
\frac{\lVert x_t-\sqrt{\alpha_t}y\rVert^2}{\beta_t}
+\frac{\lVert y-\sqrt{\bar\alpha_{t-1}}x_0\rVert^2}{1-\bar\alpha_{t-1}}
\right)\right].
\end{aligned}
$$

把平方展开，仅保留依赖 $y$ 的部分：

$$
q(y\mid x_t,x_0)\propto\exp[-\tfrac12(Ay^\top y-2b^\top y)],
$$

$$
A=\frac{\alpha_t}{\beta_t}+\frac1{1-\bar\alpha_{t-1}},
\qquad
b=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0.
$$

**高斯配方**：

$$
Ay^\top y-2b^\top y
=A\left\lVert y-\frac bA\right\rVert^2-\frac{b^\top b}{A}.
$$

因此均值为 $b/A$ ，协方差为 $A^{-1}I$ 。由 $\beta_t=1-\alpha_t$ 、 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$ ，

$$
A=\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}
=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}.
$$

于是 $A^{-1}=\tilde\beta_t$ ，且

$$
\begin{aligned}
\frac bA&=\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\right)\\
&=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t+
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0.
\end{aligned}
$$

证毕。若再代入
$x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$ ，则

$$
\begin{aligned}
\tilde\mu_t
&=\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)}(x_t-\sqrt{1-\bar\alpha_t}\epsilon)
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t\\
&=\frac1{\sqrt{\alpha_t}}x_t-
\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\epsilon\\
&=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right).
\end{aligned}
$$

从第一行到第二行合并 $x_t$ 系数时，利用
$\beta_t+\alpha_t(1-\bar\alpha_{t-1})=1-\bar\alpha_t$ 。用 $\epsilon_\theta(x_t,t)$ 代替 $\epsilon$ 即得到模型均值 $\mu_\theta$ 。 $\square$

<a id="lemma-7"></a>
### A.7 Lemma 7：为什么 KL 可化为加权噪声 MSE？（延伸阅读）

**命题。** 对 $t\ge2$ ，若模型反向分布采用固定的、严格正的协方差 $\sigma_t^2 I$ ，则

$$
D_{\rm KL}\left(\mathcal N(\tilde\mu_t,\tilde\beta_tI)
\mathbin{\Vert}\mathcal N(\mu_\theta,\sigma_t^2I)\right)
=C_t+\frac{\lVert\tilde\mu_t-\mu_\theta\rVert^2}{2\sigma_t^2},
$$

其中协方差固定是指不随网络参数 $\theta$ 学习，不要求两边协方差相等。对于维度 $d=2$ ，

$$
C_t=\frac d2\left(
\frac{\tilde\beta_t}{\sigma_t^2}-1+
\log\frac{\sigma_t^2}{\tilde\beta_t}
\right).
$$

$C_t$ 不依赖 $\theta$ 。又因

$$
\tilde\mu_t-\mu_\theta
=\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}
(\epsilon_\theta-\epsilon),
$$

得到

$$
\boxed{D_{\rm KL}=C_t+
\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
\lVert\epsilon-\epsilon_\theta\rVert^2.}
$$

**证明。** 多维高斯的 KL 公式在固定协方差下，所有协方差、行列式和迹项均不依赖 $\theta$ ；唯一与均值预测有关的项是上式中的均值平方差。将 [A.6](#lemma-6) 的两个均值表达式相减即可。 $\square$

**注意边界**：这只解释 ELBO **中间步骤**的部分项；完整负 ELBO 还有末端先验项和首步重建项。`mse_loss` 去掉时间权重是常用简化训练目标，**不是直接等于完整 ELBO**；首步后验零方差也不能直接代入该 KL 推导。

---

## 参考

- Ho, Jain, Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020。前向链与闭式采样见公式 (2)、(4)，条件后验与均值参数化见公式 (6)、(7)、(11)，加权目标与简化训练目标见公式 (12)、(14)。
