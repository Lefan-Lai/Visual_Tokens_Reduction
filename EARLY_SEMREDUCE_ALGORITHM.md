# Early-SemReduce: Training-Free Semantic Response Guided Visual Token Reduction

本文非常详细地说明 **Early-SemReduce** 的完整算法、数学推导、每一步的张量形状、实现细节、复杂度、与 ProtoReduce 的区别，以及如何把它接入 ViT/LLaVA 这类视觉语言模型。

一句话概括：

> Early-SemReduce 不是先完整跑完视觉编码器再压缩 token，而是在视觉编码器的早期或中间层之后插入一次语义响应引导的 token merge，让后续 Transformer 层只处理更少的 semantic prototype tokens。

核心思想可以写成：

```math
I
\rightarrow
F_{1:\ell}
\rightarrow
X^{(\ell)}
\rightarrow
\operatorname{SemReduce}
\rightarrow
\widetilde X^{(\ell)}
\rightarrow
F_{\ell+1:L}
\rightarrow
\text{output}.
```

其中，原始 patch tokens 数量是 $n$，压缩后的 prototype patch tokens 数量是 $m$，并且通常 $m \ll n$。

---

## 1. 为什么需要 Early-SemReduce

视觉 Transformer 和视觉语言模型通常会把一张图像切成很多 patch tokens。例如 ViT-L/14、CLIP-ViT、LLaVA 的 vision tower 或 Qwen-VL 的视觉编码器，都会生成一个视觉 token 序列：

```math
X = [x_{\mathrm{cls}}, x_1, x_2, \dots, x_n],
```

其中：

- $x_{\mathrm{cls}} \in \mathbb{R}^d$ 是 CLS token 或全局 token。
- $x_i \in \mathbb{R}^d$ 是第 $i$ 个 patch token。
- $n$ 是 patch token 数量。
- $d$ 是 token hidden dimension。

在 Transformer 中，self-attention 的主要计算量近似是：

```math
\mathcal{O}(n^2 d).
```

当 patch tokens 很多时，后续每一层都要处理 $n$ 个 token，成本会很高。直观地说，如果我们能把 $n$ 个 patch tokens 合并成 $m$ 个更少但更有代表性的 prototype tokens，后续层的 self-attention 成本就会从：

```math
\mathcal{O}(n^2 d)
```

下降到：

```math
\mathcal{O}(m^2 d).
```

如果 $m \ll n$，这个下降会非常明显。

但是，问题在于不能随便丢 token。视觉 token 里有很多细粒度判别信息，例如：

- 鸟类图像里的眼圈、鸟喙、尾羽、翅膀纹理。
- 车辆图像里的车灯、轮胎、车牌、车标。
- 医学图像里的小病灶、小边缘、小纹理。
- VQA 场景里的目标物体局部证据。

如果只是随机丢 token、平均池化 token，或者只根据视觉特征相似度做普通聚类，可能会把关键细节合并进背景 token，从而损害判断。

Early-SemReduce 的目标是：

> 只用 frozen model 已有的权重，不训练新模块，在中间层根据 patch 对分类/决策的语义响应来合并 token，从而尽量保留对最终判断有贡献的视觉证据。

---

## 2. 与 ProtoReduce 的根本区别

已有的 ProtoReduce 通常做的是：

```math
\{C_j\}_{j=1}^{m}
=
\operatorname{Cluster}
\left(
\{x_i\}_{i=1}^{n}
\right).
```

也就是说，它问的是：

> 哪些 patch tokens 在视觉特征空间里看起来相似？

Early-SemReduce 做的是：

```math
\{C_j\}_{j=1}^{m}
=
\operatorname{Cluster}
\left(
\{
W_{\mathcal S}\operatorname{LN}(x_i^{(\ell)})
\}_{i=1}^{n}
\right).
```

也就是说，它问的是：

> 哪些 patch tokens 对当前图像的候选语义类别或最终判断有相似贡献？

这一区别非常关键：

- ProtoReduce 的聚类依据是 **visual feature similarity**。
- Early-SemReduce 的聚类依据是 **semantic response similarity**。

但是，两者在最终生成 prototype token 时都应该回到原始视觉 token 空间：

```math
r_j^{(\ell)}
=
\sum_{i \in C_j}
\alpha_i^{(j)}
x_i^{(\ell)}.
```

也就是说：

- 聚类用语义响应空间。
- 聚合用原始视觉 token 空间。

这样既能利用语义信息决定哪些 patch 应该合并，又不会把 token 表征替换成分类 logits，而是保留原模型后续 Transformer 层期望接收的 hidden representation。

---

## 3. 问题定义与符号

给定一张图像 $I$，视觉编码器有 $L$ 层 Transformer：

```math
F = F_{1:L}.
```

其中：

- $F_{1:\ell}$ 表示前 $\ell$ 层视觉编码器。
- $F_{\ell+1:L}$ 表示第 $\ell+1$ 层到第 $L$ 层的后续视觉编码器。
- $\ell$ 是插入 token reduction 的层数。

Early-SemReduce 首先只运行前 $\ell$ 层：

```math
X^{(\ell)}
=
F_{1:\ell}(I).
```

得到中间层 token 序列：

```math
X^{(\ell)}
=
[
x_{\mathrm{cls}}^{(\ell)},
x_1^{(\ell)},
x_2^{(\ell)},
\dots,
x_n^{(\ell)}
].
```

这里：

- $x_{\mathrm{cls}}^{(\ell)} \in \mathbb{R}^d$ 是第 $\ell$ 层后的 CLS token。
- $x_i^{(\ell)} \in \mathbb{R}^d$ 是第 $\ell$ 层后的第 $i$ 个 patch token。
- $n$ 是 reduction 前 patch token 数量。
- $d$ 是 hidden dimension。

假设模型已有一个 frozen classifier head：

```math
W_{\mathrm{cls}} \in \mathbb{R}^{K \times d},
```

其中：

- $K$ 是类别数。
- 第 $c$ 行 $W_c$ 表示类别 $c$ 的分类权重。
- $W_{\mathrm{cls}}$ 是 frozen 的，不训练、不更新。

Early-SemReduce 希望把 $n$ 个 patch tokens 压缩成 $m$ 个 semantic prototype tokens：

```math
R^{(\ell)}
=
[
r_1^{(\ell)},
r_2^{(\ell)},
\dots,
r_m^{(\ell)}
],
\qquad
m \ll n.
```

最后把序列替换为：

```math
\widetilde X^{(\ell)}
=
[
x_{\mathrm{cls}}^{(\ell)},
r_1^{(\ell)},
r_2^{(\ell)},
\dots,
r_m^{(\ell)}
],
```

再送入后续视觉层：

```math
H^{(L)}
=
F_{\ell+1:L}
\left(
\widetilde X^{(\ell)}
\right).
```

---

## 4. 为什么它是真正的 early reduction

Early-SemReduce 的 forward 是：

```math
I
\rightarrow
F_{1:\ell}
\rightarrow
\operatorname{SemReduce}
\rightarrow
F_{\ell+1:L}.
```

它不是：

```math
I
\rightarrow
F_{1:L}
\rightarrow
\operatorname{SemReduce}.
```

更不是：

```math
I
\rightarrow
F_{1:L}
\rightarrow
\operatorname{SemReduce}
\rightarrow
F_{1:L}.
```

因此它不会额外完整跑一遍图像。计算节省来自后续层处理的 token 数变少：

原始视觉编码器后半段要处理：

```math
n \text{ patch tokens}.
```

Early-SemReduce 后，后半段只处理：

```math
m \text{ prototype patch tokens}.
```

只要 $\ell$ 不太晚，且 $m \ll n$，后续层 self-attention 的成本就会显著下降。

一般建议：

```math
\ell \in
\left[
\frac{L}{4},
\frac{L}{2}
\right].
```

原因是：

- 如果 $\ell$ 太小，patch token 还比较低级，语义尚不稳定，semantic response 可能不可靠。
- 如果 $\ell$ 太大，虽然语义更稳定，但已经跑完了太多完整 token 层，节省的计算变少。

---

## 5. Step 1: 运行浅层视觉编码器

第一步只运行前 $\ell$ 层：

```math
X^{(\ell)}
=
F_{1:\ell}(I).
```

得到：

```math
X^{(\ell)}
=
[
x_{\mathrm{cls}}^{(\ell)},
x_1^{(\ell)},
\dots,
x_n^{(\ell)}
].
```

这一步的输入是原图 $I$，输出是中间层 token。注意这里不要先跑完整个视觉编码器。SemReduce 必须插入在第 $\ell$ 层后面，这样才能让第 $\ell+1$ 到第 $L$ 层真正少算 token。

张量形状通常是：

```text
X_l:       [B, 1 + n, d]
cls:       [B, d]
patches:   [B, n, d]
```

其中 $B$ 是 batch size。如果只看单张图，可以省略 batch 维：

```text
X_l:       [1 + n, d]
cls:       [d]
patches:   [n, d]
```

---

## 6. Step 2: 选择候选语义类别

如果类别数 $K$ 很大，没有必要让每个 patch 都和所有类别计算 semantic response。可以先用 CLS token 估计当前图像最相关的候选类别。

先对 CLS token 做 frozen layer norm：

```math
\bar x_{\mathrm{cls}}^{(\ell)}
=
\operatorname{LN}
\left(
x_{\mathrm{cls}}^{(\ell)}
\right).
```

然后计算图像级 semantic response：

```math
g^{(\ell)}
=
W_{\mathrm{cls}}
\bar x_{\mathrm{cls}}^{(\ell)}.
```

其中：

```math
g^{(\ell)} \in \mathbb{R}^{K}.
```

$g_c^{(\ell)}$ 可以理解为当前中间层全局 token 对类别 $c$ 的响应。

然后取 top-$k$ 个候选类别：

```math
\mathcal S
=
\operatorname{TopK}
\left(
g^{(\ell)}, k
\right).
```

其中：

```math
|\mathcal S| = k,
\qquad
k \ll K.
```

如果 $K$ 本来就不大，也可以直接令：

```math
\mathcal S = \{1, 2, \dots, K\}.
```

候选类别选择的作用：

1. 降低计算量，从 $K$ 维 response 降到 $k$ 维 response。
2. 聚焦当前图像可能相关的语义类别，减少无关类别噪声。
3. 让 patch response 更像“当前决策空间中的贡献向量”，而不是全局类别全集上的稀疏响应。

张量形状：

```text
W_cls:        [K, d]
cls_norm:     [d]
g:            [K]
S:            [k]
W_S:          [k, d]
```

---

## 7. Step 3: 计算 patch-level semantic response

对于每个 patch token $x_i^{(\ell)}$，先做 frozen layer norm：

```math
\bar x_i^{(\ell)}
=
\operatorname{LN}
\left(
x_i^{(\ell)}
\right).
```

取候选类别对应的 classifier rows：

```math
W_{\mathcal S}
\in
\mathbb{R}^{k \times d}.
```

然后计算 patch-level semantic response：

```math
p_i^{(\ell)}
=
W_{\mathcal S}
\bar x_i^{(\ell)}.
```

其中：

```math
p_i^{(\ell)}
\in
\mathbb{R}^{k}.
```

展开写：

```math
p_i^{(\ell)}
=
\begin{bmatrix}
p_{i,c_1}^{(\ell)}
\\
p_{i,c_2}^{(\ell)}
\\
\vdots
\\
p_{i,c_k}^{(\ell)}
\end{bmatrix},
\qquad
c_1,\dots,c_k \in \mathcal S.
```

这个向量表示第 $i$ 个 patch 对当前候选类别集合的响应模式。

如果两个 patch 的 $p_i$ 很相似，说明它们对候选类别的贡献模式相似。例如：

- 两个 patch 都强烈支持类别 A。
- 两个 patch 都支持 A 但反对 B。
- 两个 patch 都没有明显类别响应，更像背景。
- 两个 patch 都对某个细粒度类别有局部证据。

Early-SemReduce 后续聚类看的不是：

```math
\cos
\left(
x_i^{(\ell)},
x_j^{(\ell)}
\right),
```

而是：

```math
\cos
\left(
p_i^{(\ell)},
p_j^{(\ell)}
\right).
```

张量形状：

```text
patches:       [n, d]
patches_norm:  [n, d]
W_S:           [k, d]
P:             [n, k]
```

其中 $P$ 的第 $i$ 行就是 $p_i^{(\ell)}$。

---

## 8. Step 4: 标准化 semantic response

不同类别的 logit 尺度可能不同。有的类别权重范数大，有的类别响应天然偏高。如果直接用原始 $p_i$ 聚类，某些尺度大的类别会主导距离。

因此需要在 patch 维度上对每个候选类别做标准化。

对每个候选类别 $c \in \mathcal S$，计算所有 patch 上的均值：

```math
\mu_c
=
\frac{1}{n}
\sum_{i=1}^{n}
p_{i,c}^{(\ell)}.
```

计算标准差：

```math
\sigma_c
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left(
p_{i,c}^{(\ell)}
-
\mu_c
\right)^2
}.
```

然后标准化：

```math
\hat p_{i,c}^{(\ell)}
=
\frac{
p_{i,c}^{(\ell)}
-
\mu_c
}{
\sigma_c + \epsilon
}.
```

得到标准化后的 response vector：

```math
\hat p_i^{(\ell)}
=
[
\hat p_{i,c_1}^{(\ell)},
\hat p_{i,c_2}^{(\ell)},
\dots,
\hat p_{i,c_k}^{(\ell)}
].
```

再做 L2 归一化：

```math
q_i^{(\ell)}
=
\frac{
\hat p_i^{(\ell)}
}{
\left\|
\hat p_i^{(\ell)}
\right\|_2
+
\epsilon
}.
```

其中：

```math
q_i^{(\ell)}
\in
\mathbb{R}^{k}.
```

$q_i$ 才是后续聚类使用的 semantic response token。

这一步的意义：

1. `Std` 去除类别 logit 尺度差异。
2. `L2 normalize` 让后续 dot product 等价于 cosine similarity。
3. 聚类只关注 response pattern 的方向，而不是绝对幅度。

张量形状：

```text
P:       [n, k]
mean:    [1, k]
std:     [1, k]
P_hat:   [n, k]
Q:       [n, k]
```

---

## 9. Step 5: 计算 semantic importance

仅知道两个 patch 的语义响应是否相似还不够，还需要知道某个 patch 是否重要。

一个背景 patch 可能对所有候选类别响应都差不多。它的 response vector 可能稳定，但不一定重要。一个细粒度关键 patch 可能对某个类别有明显更强响应，这种 patch 应该尽量被保留下来，或者在聚合时获得更高权重。

定义第 $i$ 个 patch 的 semantic importance：

```math
u_i
=
\max_{c \in \mathcal S}
\hat p_{i,c}^{(\ell)}
-
\frac{1}{k}
\sum_{c \in \mathcal S}
\hat p_{i,c}^{(\ell)}.
```

直觉：

- 第一项 $\max_c \hat p_{i,c}$ 表示这个 patch 对最强候选类别的响应。
- 第二项 $\frac{1}{k}\sum_c \hat p_{i,c}$ 表示这个 patch 的平均响应。
- 两者差值越大，说明该 patch 对某个候选类别有更尖锐、更有区分度的贡献。

然后对所有 patch 的 $u_i$ 再做一次标准化：

```math
\tilde u_i
=
\frac{
u_i - \operatorname{mean}(u)
}{
\operatorname{std}(u) + \epsilon
}.
```

其中：

```math
\tilde u_i
\in
\mathbb{R}.
```

这个标量表示第 $i$ 个 patch 的相对语义重要性。

后面它会用于三个地方：

1. 选择 semantic anchor tokens。
2. 初始化聚类中心时偏向重要 patch。
3. 聚合 prototype token 时给重要 patch 更高权重。

张量形状：

```text
P_hat:    [n, k]
u:        [n]
u_tilde:  [n]
```

---

## 10. Step 6: 保护 semantic anchor tokens

对于细粒度识别，某些 patch 不能被轻易平均掉。例如：

- 鸟的眼圈。
- 鸟喙边缘。
- 翅膀上的纹理。
- 车牌上的小文字。
- 人手中的小物体。
- VQA 问题里被问到的局部区域。

如果这些区域被合并进大背景 cluster，信息可能被稀释。

因此 Early-SemReduce 先选择 top-$b$ 个最重要的 patch 作为 semantic anchors：

```math
\mathcal A
=
\operatorname{TopB}
\left(
\tilde u_i, b
\right).
```

其中：

```math
|\mathcal A| = b.
```

这些 anchor tokens 直接成为 singleton clusters：

```math
C_j = \{a_j\},
\qquad
a_j \in \mathcal A,
\qquad
j = 1,\dots,b.
```

对应 prototype token 直接是原 token：

```math
r_j^{(\ell)}
=
x_{a_j}^{(\ell)}.
```

这样最重要的 $b$ 个 patch 不会被平均掉。

如果不想显式保护 anchor，可以设置：

```math
b = 0.
```

这时所有 patch 都进入普通 semantic clustering。

注意约束：

```math
0 \le b \le m.
```

如果用户给的 $b > m$，实现中应该自动裁剪为：

```math
b \leftarrow m.
```

---

## 11. Step 7: 对剩余 patch 做 semantic clustering

去掉 anchors 后，剩余 patch 集合为：

```math
\mathcal U
=
\{1,2,\dots,n\}
\setminus
\mathcal A.
```

剩余需要生成的 cluster 数量是：

```math
M = m - b.
```

目标是把 $\mathcal U$ 中的 patch 分成：

```math
D_1, D_2, \dots, D_M.
```

最终完整的 $m$ 个 cluster 是：

```math
C_1,\dots,C_b,D_1,\dots,D_M.
```

其中：

- $C_1,\dots,C_b$ 是 anchor singleton clusters。
- $D_1,\dots,D_M$ 是普通 semantic clusters。

---

## 12. Step 7.1: Semantic-aware initialization

普通 k-means 的初始化可能只考虑几何距离，但 Early-SemReduce 希望中心既重要又多样。

第一个非 anchor 中心选择剩余 patch 中最重要的 token：

```math
s_1
=
\arg\max_{i \in \mathcal U}
\tilde u_i.
```

初始化：

```math
\mu_1
=
q_{s_1}^{(\ell)}.
```

后续中心同时考虑：

- semantic importance。
- semantic diversity。

对第 $t$ 个中心：

```math
s_t
=
\arg\max_{i \in \mathcal U}
\left[
\tilde u_i
+
\lambda_{\mathrm{div}}
\min_{r < t}
\left(
1
-
\left(q_i^{(\ell)}\right)^\top
q_{s_r}^{(\ell)}
\right)
\right],
```

其中：

```math
t = 2,\dots,M.
```

然后：

```math
\mu_t
=
q_{s_t}^{(\ell)}.
```

这里：

- $\tilde u_i$ 鼓励选择重要 patch。
- $1 - q_i^\top q_{s_r}$ 是 semantic distance。
- $\min_{r<t}$ 表示该 patch 到已有中心中最近中心的距离。
- $\lambda_{\mathrm{div}}$ 控制多样性强度。

如果 $\lambda_{\mathrm{div}}$ 较大：

- 初始化中心更分散。
- 覆盖更多语义区域。

如果 $\lambda_{\mathrm{div}}$ 较小：

- 初始化更偏向高重要性 patch。
- 可能更关注判别区域。

张量形状：

```text
Q_U:       [|U|, k]
u_U:       [|U|]
centers:   [M, k]
```

---

## 13. Step 7.2: Semantic assignment

有了中心 $\mu_1,\dots,\mu_M$ 后，对每个非 anchor patch，按照 semantic response similarity 分配 cluster：

```math
g_i
=
\arg\max_{t \in \{1,\dots,M\}}
\left(q_i^{(\ell)}\right)^\top
\mu_t,
\qquad
i \in \mathcal U.
```

然后：

```math
D_t
=
\{i \in \mathcal U \mid g_i = t\}.
```

这里的 assignment 使用的是：

```math
\left(q_i^{(\ell)}\right)^\top
\mu_t,
```

不是：

```math
\left(x_i^{(\ell)}\right)^\top
c_t.
```

这就是 semantic clustering 的核心。

因为 $q_i$ 和 $\mu_t$ 都 L2 normalized，所以 dot product 就是 cosine similarity。

---

## 14. Step 7.3: Semantic center update

对于每个 cluster $D_t$，用 semantic importance 加权更新中心：

```math
\mu_t
=
\frac{
\sum_{i \in D_t}
\exp
\left(
\gamma \tilde u_i
\right)
q_i^{(\ell)}
}{
\left\|
\sum_{i \in D_t}
\exp
\left(
\gamma \tilde u_i
\right)
q_i^{(\ell)}
\right\|_2
+
\epsilon
}.
```

其中：

- $\gamma$ 控制高重要性 patch 对 cluster center 的影响。
- $\gamma = 0$ 时，所有 patch 权重相同。
- $\gamma > 0$ 时，重要 patch 对中心影响更大。

当 $\gamma=0$：

```math
\mu_t
=
\operatorname{Normalize}
\left(
\sum_{i \in D_t}
q_i^{(\ell)}
\right).
```

重复 assignment 和 update 共 $T$ 次。

通常 $T$ 不需要很大，常用：

```text
T = 3 to 8
```

因为每张图像的 token 数有限，且 reduction 是 inference-time 操作，过多迭代会增加额外开销。

---

## 15. 空 cluster 处理

semantic clustering 可能出现空 cluster：

```math
D_t = \varnothing.
```

空 cluster 会导致 prototype 数量少于 $m$，所以必须处理。

一种处理方式是重新初始化该中心：

```math
\mu_t
=
q_{i^\star}^{(\ell)}.
```

其中：

```math
i^\star
=
\arg\max_{i \in \mathcal U}
\left[
\tilde u_i
+
\min_{r \ne t}
\left(
1
-
\left(q_i^{(\ell)}\right)^\top
\mu_r
\right)
\right].
```

含义：

- 选择重要性高的 patch。
- 同时选择当前 centers 表示不充分的 patch。

工程实现中还可以采用更稳定的 repair 策略：

1. 找出空 cluster。
2. 找出当前包含多个 patch 的 donor cluster。
3. 从 donor cluster 中移出一个高重要性或低拟合度 patch。
4. 把这个 patch 分配给空 cluster。

这样能保证：

```math
\sum_{j=1}^{m}
|C_j|
=
n.
```

也就是每个原始 patch 恰好属于一个 prototype cluster。

---

## 16. Step 8: 生成 semantic reduction tokens

聚类只是决定了哪些 patch 属于同一组。最终还需要把每组 patch 合成一个视觉 token。

重要原则：

> 聚类在 semantic response space 里做，但 prototype token 必须在 original visual token space 里生成。

### 16.1 Anchor cluster

对于 anchor cluster：

```math
C_j = \{a_j\},
\qquad
j = 1,\dots,b,
```

直接令：

```math
r_j^{(\ell)}
=
x_{a_j}^{(\ell)}.
```

### 16.2 普通 semantic cluster

对于普通 cluster：

```math
D_t,
\qquad
t = 1,\dots,M,
```

先计算 cluster 内每个 patch 的聚合分数：

```math
s_i^{(t)}
=
\left(q_i^{(\ell)}\right)^\top
\mu_t
+
\lambda_{\mathrm{imp}}
\tilde u_i,
\qquad
i \in D_t.
```

其中：

- 第一项 $q_i^\top \mu_t$ 表示 patch 与 semantic center 的相似度。
- 第二项 $\lambda_{\mathrm{imp}}\tilde u_i$ 表示 patch 本身的重要性。
- $\lambda_{\mathrm{imp}}$ 控制聚合时重要性加权强度。

然后用 softmax 得到聚合权重：

```math
\alpha_i^{(t)}
=
\frac{
\exp
\left(
s_i^{(t)} / \tau
\right)
}{
\sum_{l \in D_t}
\exp
\left(
s_l^{(t)} / \tau
\right)
}.
```

满足：

```math
\sum_{i \in D_t}
\alpha_i^{(t)}
=
1,
\qquad
\alpha_i^{(t)}
\ge
0.
```

最终 prototype token：

```math
r_{b+t}^{(\ell)}
=
\sum_{i \in D_t}
\alpha_i^{(t)}
x_i^{(\ell)}.
```

注意这里加权的是原始视觉 token $x_i^{(\ell)}$，不是 $q_i^{(\ell)}$。

原因是：

- $q_i$ 是 $k$ 维 semantic response vector。
- 后续视觉 Transformer 期望输入的是 $d$ 维 hidden token。
- $q_i$ 只用于决定聚类和权重，不能替代视觉 token。

温度 $\tau$ 的作用：

- $\tau$ 小：权重更尖锐，更接近选择 cluster 中最代表性的 patch。
- $\tau$ 大：权重更平滑，更接近平均池化。

常见设置：

```text
tau = 0.05 to 0.2
```

---

## 17. Step 9: Soft position aggregation

如果后续模型需要保留 token 的空间顺序，可以为每个 prototype token 计算 soft position。

设第 $i$ 个 patch 的二维位置是：

```math
\pi_i
=
(h_i, w_i).
```

对于 anchor token：

```math
\pi_j^r
=
\pi_{a_j}.
```

对于普通 prototype：

```math
\pi_{b+t}^r
=
\sum_{i \in D_t}
\alpha_i^{(t)}
\pi_i.
```

然后按 raster order 排序：

```math
\text{top-left}
\rightarrow
\text{bottom-right}.
```

排序后的 prototype 序列为：

```math
\widetilde R^{(\ell)}
=
\operatorname{SortByPosition}
\left(
R^{(\ell)}
\right).
```

这样可以尽量保留二维空间结构。

### 17.1 Absolute position embedding 的情况

如果模型使用标准 ViT absolute position embedding，并且 position embedding 已经在输入层加进 token：

```math
x_i^{(0)}
=
\operatorname{PatchEmbed}(I)_i
+
e_i^{\mathrm{pos}},
```

那么中间层 reduction 时通常不需要重新加 position embedding，因为位置信息已经混入 hidden representation。

### 17.2 Relative position bias 或 2D RoPE 的情况

如果模型使用 relative position bias 或 2D RoPE，则 reduced tokens 之间的位置关系需要重新定义。

这时可以使用 soft position：

```math
\pi_j^r
```

来计算 reduced tokens 之间的相对位置。

例如相对位移：

```math
\Delta \pi_{a,b}^r
=
\pi_a^r
-
\pi_b^r.
```

这可以作为 relative position bias 或 RoPE 坐标的近似输入。

---

## 18. Step 10: 继续后续 Transformer 层

原始第 $\ell$ 层后的序列是：

```math
[
x_{\mathrm{cls}}^{(\ell)},
x_1^{(\ell)},
\dots,
x_n^{(\ell)}
].
```

Early-SemReduce 替换为：

```math
\widetilde X^{(\ell)}
=
[
x_{\mathrm{cls}}^{(\ell)},
\widetilde r_1^{(\ell)},
\widetilde r_2^{(\ell)},
\dots,
\widetilde r_m^{(\ell)}
].
```

其中 $\widetilde r_j$ 表示按 soft position 排序后的 prototype。

然后继续运行后续视觉层：

```math
H^{(L)}
=
F_{\ell+1:L}
\left(
\widetilde X^{(\ell)}
\right).
```

对于分类模型，最终预测可以写成：

```math
\hat y
=
\operatorname{softmax}
\left(
W_{\mathrm{cls}}
\operatorname{LN}
\left(
h_{\mathrm{cls}}^{(L)}
\right)
\right).
```

对于视觉语言模型，后续可能是：

1. 视觉 encoder 后续层。
2. multimodal projector。
3. language model 的 image placeholder replacement。
4. decoder-only language model generation。

Early-SemReduce 的接入点要根据模型结构决定。

---

## 19. 完整算法伪代码

```text
Algorithm: Early-SemReduce
Semantic Response Guided Training-Free Token Reduction

Input:
    I: input image
    F_{1:L}: frozen visual encoder with L Transformer layers
    W_cls in R^{K x d}: frozen classifier head
    ell: reduction layer
    m: target number of output patch tokens
    k: number of candidate classes
    b: number of protected semantic anchors
    T: number of clustering iterations
    tau: aggregation temperature
    lambda_imp: importance weight for final aggregation
    lambda_div: diversity weight for initialization
    gamma: importance weight for center update
    eps: numerical stability constant

Output:
    H^{(L)}: final visual representation after token reduction

1. Shallow forward:
       X_l <- F_{1:ell}(I)
       X_l = [x_cls, x_1, ..., x_n]

2. Candidate class selection:
       cls_norm <- LN(x_cls)
       g <- W_cls cls_norm
       S <- TopK(g, k)
       W_S <- rows of W_cls indexed by S

3. Patch-level semantic response:
       for i = 1 to n:
           x_i_norm <- LN(x_i)
           p_i <- W_S x_i_norm

4. Standardization:
       for each class c in S:
           mu_c <- mean_i p_i[c]
           sigma_c <- std_i p_i[c]

       for i = 1 to n:
           p_hat_i[c] <- (p_i[c] - mu_c) / (sigma_c + eps)
           q_i <- p_hat_i / (||p_hat_i||_2 + eps)

5. Semantic importance:
       for i = 1 to n:
           u_i <- max_c p_hat_i[c] - mean_c p_hat_i[c]

       u_tilde_i <- (u_i - mean(u)) / (std(u) + eps)

6. Anchor protection:
       A <- TopB(u_tilde, b)
       for j = 1 to b:
           a_j <- A[j]
           C_j <- {a_j}
           r_j <- x_{a_j}

7. Semantic clustering:
       U <- {1, ..., n} \ A
       M <- m - b

       Initialize centers mu_1, ..., mu_M:
           s_1 <- argmax_{i in U} u_tilde_i
           mu_1 <- q_{s_1}

           for t = 2 to M:
               s_t <- argmax_{i in U}
                      [u_tilde_i
                       + lambda_div * min_{r < t} (1 - q_i^T q_{s_r})]
               mu_t <- q_{s_t}

       repeat T times:
           Assignment:
               for i in U:
                   g_i <- argmax_t q_i^T mu_t

           Build clusters:
               D_t <- {i in U | g_i = t}

           Update centers:
               for t = 1 to M:
                   if D_t is empty:
                       repair or reinitialize center
                   else:
                       mu_t <- Normalize(
                                   sum_{i in D_t}
                                   exp(gamma * u_tilde_i) q_i
                               )

8. Prototype aggregation:
       for t = 1 to M:
           for i in D_t:
               score_i <- q_i^T mu_t + lambda_imp * u_tilde_i

           alpha_i <- Softmax(score_i / tau over i in D_t)

           r_{b+t} <- sum_{i in D_t} alpha_i x_i

9. Optional soft position sorting:
       compute pi_j^r for each prototype
       sort prototypes by raster order

10. Continue visual encoder:
       X_reduced <- [x_cls, r_1, ..., r_m]
       H_L <- F_{ell+1:L}(X_reduced)

11. return H_L
```

---

## 20. 紧凑数学公式

Early-SemReduce 可以浓缩为以下几行：

```math
X^{(\ell)}
=
F_{1:\ell}(I),
```

```math
p_i^{(\ell)}
=
W_{\mathcal S}
\operatorname{LN}
\left(
x_i^{(\ell)}
\right),
```

```math
q_i^{(\ell)}
=
\operatorname{Normalize}
\left(
\operatorname{Std}
\left(
p_i^{(\ell)}
\right)
\right),
```

```math
\{C_j\}_{j=1}^{m}
=
\operatorname{Cluster}
\left(
\{q_i^{(\ell)}\}_{i=1}^{n},
m
\right),
```

```math
r_j^{(\ell)}
=
\sum_{i \in C_j}
\alpha_i^{(j)}
x_i^{(\ell)}.
```

其中：

```math
\alpha_i^{(j)}
=
\frac{
\exp
\left(
\frac{
\left(q_i^{(\ell)}\right)^\top \mu_j
+
\lambda_{\mathrm{imp}}
\tilde u_i
}{
\tau
}
\right)
}{
\sum_{l \in C_j}
\exp
\left(
\frac{
\left(q_l^{(\ell)}\right)^\top \mu_j
+
\lambda_{\mathrm{imp}}
\tilde u_l
}{
\tau
}
\right)
}.
```

最终：

```math
H^{(L)}
=
F_{\ell+1:L}
\left(
[
x_{\mathrm{cls}}^{(\ell)},
r_1^{(\ell)},
\dots,
r_m^{(\ell)}
]
\right).
```

---

## 21. 为什么 Early-SemReduce 是 training-free

Early-SemReduce 不引入任何需要训练的新参数。

它使用的全部对象都是：

1. Frozen visual encoder:

```math
F_{1:L}.
```

2. Frozen classifier head:

```math
W_{\mathrm{cls}}.
```

3. 固定算子：

```text
LayerNorm
TopK
standardization
L2 normalization
argmax assignment
weighted center update
softmax aggregation
position sorting
```

这些都没有 learnable parameters。

因此，整个过程不需要：

- 反向传播。
- gradient update。
- 新数据标注。
- finetuning。
- adapter training。
- prompt tuning。

在 inference 时，对每张图像动态执行一次 semantic reduction 即可。

---

## 22. 复杂度分析

假设每层 self-attention 主要复杂度为：

```math
\mathcal{O}(n^2 d).
```

原始视觉编码器复杂度约为：

```math
\mathcal{O}(L n^2 d).
```

Early-SemReduce 复杂度约为：

```math
\mathcal{O}(\ell n^2 d)
+
\mathcal{O}((L-\ell)m^2 d)
+
\mathcal{O}(n k d)
+
\mathcal{O}(T n m k).
```

逐项解释：

1. 前 $\ell$ 层仍然处理完整 token：

```math
\mathcal{O}(\ell n^2 d).
```

2. 后续 $L-\ell$ 层只处理 $m$ 个 prototype tokens：

```math
\mathcal{O}((L-\ell)m^2 d).
```

3. 计算 patch-level semantic response：

```math
\mathcal{O}(n k d).
```

4. Semantic clustering 的 assignment/update：

```math
\mathcal{O}(T n m k).
```

当：

```math
m \ll n,
\qquad
k \ll K,
```

并且 $\ell$ 不太晚时，整体计算量会显著小于完整视觉编码器。

---

## 23. 可选增强：Mass-Aware Attention

一个 prototype token 可能代表多个原始 patch：

```math
\rho_j = |C_j|.
```

如果直接用一个 token 替代多个 token，后续 attention 可能低估该区域的总贡献。直觉上，原本 cluster 中有 $|C_j|$ 个 key/value，现在只剩一个 key/value，它在 attention softmax 中的质量变小了。

可以给每个 prototype token 一个 cluster mass：

```math
\rho_j = |C_j|.
```

在后续 self-attention 中，对 key token $r_j$ 加 log-mass bias：

```math
A_{a,j}
=
\frac{
\exp
\left(
\frac{
Q_a K_j^\top
}{
\sqrt d
}
+
\log \rho_j
\right)
}{
\sum_t
\exp
\left(
\frac{
Q_a K_t^\top
}{
\sqrt d
}
+
\log \rho_t
\right)
}.
```

直觉近似：

```math
\sum_{i \in C_j}
\exp(Q_a K_i^\top)
\approx
|C_j|
\exp(Q_a K_j^\top).
```

所以：

```math
\log |C_j|
```

可以近似补偿多个 token 被压成一个 token 后的 attention mass 损失。

这仍然是 training-free，因为 $\rho_j$ 是 cluster size，不是可学习参数。

如果不想修改 attention 实现，可以不使用这个增强，直接使用普通 prototype tokens。

---

## 24. 接入 ViT 分类模型的建议

对于标准 ViT 分类模型，最自然的接入方式是：

1. 保留 patch embedding 和 position embedding。
2. 运行前 $\ell$ 个 Transformer blocks。
3. 使用 frozen classifier head $W_{\mathrm{cls}}$ 计算 semantic response。
4. 执行 Early-SemReduce。
5. 把 `[CLS, prototypes]` 输入后续 blocks。
6. 使用原来的 norm 和 classifier head 输出 logits。

伪代码：

```python
x = patch_embed(image)
x = add_cls_and_position(x)

for block in blocks[:ell]:
    x = block(x)

cls = x[:, 0]
patches = x[:, 1:]

patches_reduced = early_semreduce(
    patch_tokens=patches,
    cls_token=cls,
    classifier=model.head.weight,
    num_prototypes=m,
    candidate_classes=k,
    num_anchors=b,
)

x = concat([cls, patches_reduced])

for block in blocks[ell:]:
    x = block(x)

logits = model.head(model.norm(x[:, 0]))
```

注意：

- `model.head.weight` 必须和 token dimension $d$ 对齐。
- 如果 head 前还有 final norm，应在计算 semantic response 时使用相同或兼容的 norm。
- 如果模型有 distillation token，需要同时保留 dist token，不能当 patch token 合并掉。

---

## 25. 接入 LLaVA/VLM 的注意事项

LLaVA 这类 VLM 通常不是标准 ImageNet classifier。它的流程大致是：

```text
image
 -> vision tower
 -> multimodal projector
 -> image features in language hidden space
 -> replace <image> placeholders in text embedding
 -> language model generation
```

严格意义上的 Early-SemReduce 最理想位置是在 vision tower 的中间层：

```text
vision block 1...ell
 -> SemReduce
 -> vision block ell+1...L
 -> projector
 -> LLM
```

这样可以真正节省 vision tower 后续层计算。

但 Hugging Face LLaVA 的公开接口通常更容易 hook 的位置是：

```text
vision tower + projector 之后
LLM 接收 image features 之前
```

这个位置虽然能减少 LLM 看到的 image tokens，但不一定节省 vision tower 本身的计算。它仍然可以用于研究 token reduction 对 hallucination、POPE yes/no bias、answer accuracy 的影响。

### 25.1 如果没有显式视觉分类头怎么办

Early-SemReduce 需要 $W_{\mathrm{cls}}$。如果模型没有 ImageNet classifier head，可以考虑使用 frozen surrogate semantic head：

1. 使用语言模型的 output embedding / LM head：

```math
W_{\mathrm{lm}} \in \mathbb{R}^{V \times d}.
```

如果 image features 已经投影到语言 hidden space，$W_{\mathrm{lm}}$ 可以作为 frozen semantic response head。

2. 使用 prompt candidate tokens 对应的 embedding rows。

例如 yes/no、object names、dataset label names、question keywords。

3. 使用 CLIP text embeddings 作为 semantic anchors。

这仍然是 training-free，只要这些 heads/embeddings 是 frozen 的。

### 25.2 image placeholder 数量必须同步

在 Hugging Face LLaVA 中，文本 prompt 里 `<image>` token 的数量必须等于 image features 的数量。

如果把 image features 从 $n$ 压缩到 $m$，也必须把 prompt 中 image placeholders 改成 $m$ 个。

否则会出现类似错误：

```text
Image features and image tokens do not match
```

因此 VLM runner 需要同时做两件事：

1. 压缩 image features。
2. 修改 input_ids 中 `<image>` placeholder 的数量。

---

## 26. 超参数解释与推荐

### 26.1 `m`: prototype token 数

`m` 决定压缩强度。

```text
m 越小: 计算更省，但信息损失更大
m 越大: 信息更完整，但计算节省更少
```

常见选择：

```text
m = 64, 96, 128, 192
```

对于 LLaVA 13B POPE 这类实验，可以先用：

```text
m = 128
```

作为稳定 baseline。

### 26.2 `k`: candidate class 数

`k` 决定 semantic response vector 的维度。

```text
k 太小: 语义空间太窄，可能漏掉重要类别
k 太大: 噪声和计算增加
```

常见选择：

```text
k = 32, 64, 128
```

### 26.3 `b`: anchor 数

`b` 决定保护多少高重要性 patch。

```text
b = 0: 不保护 anchor
b > 0: 显式保留 top-b semantic importance tokens
```

常见选择：

```text
b = 4, 8, 16
```

### 26.4 `T`: clustering iterations

`T` 决定聚类迭代次数。

```text
T 太小: cluster 不稳定
T 太大: 推理开销增加
```

常见选择：

```text
T = 3, 5, 8
```

### 26.5 `tau`: aggregation temperature

`tau` 控制 cluster 内 soft pooling 的尖锐程度。

```text
tau 小: 更接近选择最代表 token
tau 大: 更接近平均池化
```

常见选择：

```text
tau = 0.05, 0.07, 0.1, 0.2
```

### 26.6 `lambda_imp`

`lambda_imp` 控制 final aggregation 中 semantic importance 的影响。

```text
lambda_imp = 0: 只按 patch-center 相似度聚合
lambda_imp > 0: 重要 patch 权重更高
```

常见选择：

```text
lambda_imp = 0.1, 0.25, 0.5
```

### 26.7 `lambda_div`

`lambda_div` 控制初始化时的语义多样性。

```text
lambda_div 小: 更偏向高重要性
lambda_div 大: 更偏向分散覆盖
```

常见选择：

```text
lambda_div = 0.5, 1.0, 2.0
```

### 26.8 `gamma`

`gamma` 控制 center update 时重要 patch 对中心的影响。

```text
gamma = 0: 普通均值更新
gamma > 0: importance-aware center update
```

常见选择：

```text
gamma = 0.5, 1.0, 2.0
```

---

## 27. 建议消融实验

为了证明每个设计都有作用，建议做以下 ablation：

1. Visual clustering vs semantic clustering。

```text
ProtoReduce: cluster x_i
Early-SemReduce: cluster q_i
```

2. 是否使用 anchor。

```text
b = 0
b = 8
b = 16
```

3. 是否使用 importance-aware aggregation。

```text
lambda_imp = 0
lambda_imp = 0.25
```

4. 是否使用 importance-aware center update。

```text
gamma = 0
gamma = 1
```

5. candidate class 数量。

```text
k = 16, 32, 64, 128
```

6. reduction layer。

```text
ell = L/4, L/3, L/2, 2L/3
```

7. prototype token 数量。

```text
m = 64, 96, 128, 192
```

---

## 28. 可能失败的情况

Early-SemReduce 不是无条件更好，可能失败的情况包括：

1. Reduction 太早。

如果 $\ell$ 太小，patch token 尚未形成稳定语义，semantic response 噪声较大。

2. `m` 太小。

如果 prototype 数过少，小目标、细节区域可能被合并掉。

3. `k` 太小。

候选类别集合如果漏掉真实相关语义，response space 会偏。

4. 没有合适的 classifier head。

如果 $W_{\mathrm{cls}}$ 与 token space 不匹配，semantic response 不可靠。

5. VLM 中 surrogate semantic head 不完美。

用 LM head 或 token embeddings 作为 surrogate semantic head 时，它和视觉 patch 的关系可能不如真正的视觉分类头直接。

6. Position 处理不当。

如果模型强依赖 relative position，而 reduction 后没有正确处理 soft position，可能影响后续 attention。

---

## 29. 推荐报告方式

在论文或实验报告中，可以这样描述方法：

> We propose Early-SemReduce, a training-free semantic response guided token reduction method. Instead of clustering intermediate visual tokens by feature similarity, Early-SemReduce projects each patch token onto a frozen classifier head and clusters tokens in the semantic response space. This groups patches that contribute similarly to the model decision. High-importance patches are protected as semantic anchors, and each remaining cluster is converted into a prototype token by importance-aware weighted aggregation in the original visual token space. The reduced token sequence is then forwarded through the remaining Transformer layers, reducing computation while preserving discriminative visual evidence.

中文描述：

> 我们提出 Early-SemReduce，一种 training-free 的语义响应引导视觉 token 压缩方法。不同于根据视觉特征相似度聚类 patch tokens，Early-SemReduce 首先运行视觉编码器的浅层部分，并利用原模型冻结的分类头为每个中间层 patch token 计算类别语义响应。随后，方法在语义响应空间中进行聚类，使得对分类决策具有相似贡献的 patch tokens 被合并。对于高语义重要性的 patch，方法将其作为 semantic anchor 直接保留；对于其余语义簇，方法使用语义重要性加权的方式聚合原始视觉 tokens，生成 semantic prototype token。最后，仅将压缩后的 prototype tokens 输入后续 Transformer 层，从而在无需训练的情况下减少计算量，并尽可能保留细粒度识别所需的判别性区域。

---

## 30. 最核心公式

如果只能保留一行公式，Early-SemReduce 的核心可以写成：

```math
r_j^{(\ell)}
=
\sum_{i \in C_j}
\operatorname{Softmax}_{i \in C_j}
\left(
\frac{
\cos
\left(
W_{\mathcal S}
\operatorname{LN}
\left(
x_i^{(\ell)}
\right),
\mu_j
\right)
+
\lambda_{\mathrm{imp}}
\tilde u_i
}{
\tau
}
\right)
x_i^{(\ell)}.
```

其中：

```math
C_j
```

不是由视觉特征相似度决定，而是由：

```math
W_{\mathcal S}
\operatorname{LN}
\left(
x_i^{(\ell)}
\right)
```

的语义响应相似度决定。

这就是 Early-SemReduce 和普通视觉 token clustering 最大的区别。

---

## 31. 最终 takeaway

Early-SemReduce 可以理解为：

1. 先让视觉编码器浅层提取一定语义。
2. 用 frozen classifier head 把 patch token 映射到候选语义响应空间。
3. 在语义响应空间里判断哪些 patch 对决策贡献类似。
4. 显式保护最重要的细粒度 patch。
5. 把其余 patch 聚成 semantic clusters。
6. 用原始视觉 token 空间生成 prototype tokens。
7. 后续 Transformer 只处理更少的 prototype tokens。

它的目标不是简单压缩图像，而是：

> 在 training-free 的前提下，用模型自己已有的语义决策空间来指导视觉 token merge。
