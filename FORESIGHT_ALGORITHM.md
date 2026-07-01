# 非常重要：FORESIGHT 算法完整流程

> 非常重要：本文档记录 FORESIGHT 当前完整算法流程，应作为实现、复现和实验对齐时的优先参考。

本文档记录当前仓库实现的 FORESIGHT 算法。这个版本的核心是：

1. 从当前图像的 projected image tokens 中生成 image-conditioned hypothesis directions。
2. 从当前问题文本中生成 text-conditioned hypothesis directions。
3. 将 image hypotheses 和 text hypotheses 合并成一个 candidate pool。
4. 用 global support 与 local support 得到每个 candidate 的初始支持分数。
5. 用 diversity-aware selection 选出最多 `K_max` 个 candidate hypotheses。
6. 用 candidate response 得到 token-level support 和 image-level confidence。
7. 通过 multiplicative evidence 动态决定 `K_eff`。
8. 在 active hypothesis response space 中聚类。
9. 回到原始 projected visual feature space 聚合 tokens，得到 reduced visual tokens。

本版本不包含 protected tokens，也不包含 dual-space clustering。文本 hypothesis 会直接进入 candidate pool，而不是只作为 image candidate 的 reweighting。

## 1. 输入、输出和位置

输入是一张图像 `I`，以及可选的文本 instruction 或问题 `u`。

图像先经过 frozen vision tower 和 multimodal projector：

```text
X = Projector(VisionTower(I))
```

得到 projector 后的 image tokens：

```text
X = {x_i}_{i=1}^n in R^{n x d}
```

其中：

- `n` 是原始 image token 数量。
- `d` 是 projector 输出维度，也是 LLM 可以接收的 hidden dimension。

FORESIGHT 的输出是一组更短的 visual prototype tokens：

```text
R = {r_j}_{j=1}^{K_eff} in R^{K_eff x d}
```

其中 `K_eff` 是当前图像动态决定的压缩后 token 数量。

整体流程是：

```text
I -> VisionTower -> Projector -> X -> FORESIGHT -> R -> LLM
```

## 2. 对 projected image tokens 做归一化

首先对每个 projected image token 做 layer normalization：

```text
x_tilde_i = LN(x_i)
```

然后做 L2 normalization：

```text
z_i = x_tilde_i / (||x_tilde_i||_2 + eps)
```

得到：

```text
X_tilde = {x_tilde_i}_{i=1}^n
Z = {z_i}_{i=1}^n
```

然后计算整张图的平均表示：

```text
x_bar = (1 / n) * sum_i x_tilde_i
```

再将其归一化，得到 global image representation：

```text
z_img = x_bar / (||x_bar||_2 + eps)
```

这里，`x_tilde_i` 用于后续 response 计算，`z_i` 用于相似度计算。

## 3. 从当前图像生成 image-conditioned hypothesis directions

对每个 image token 都构造一个图像条件下的候选 hypothesis direction。

对于 token `i`，先计算它在全图平均方向上的投影系数：

```text
kappa_i = dot(x_tilde_i, x_bar) / (||x_bar||_2^2 + eps)
```

然后从 token 表示中去掉这部分全局平均方向：

```text
r_img_i = x_tilde_i - kappa_i * x_bar
```

再做 normalization，得到 image-conditioned hypothesis direction：

```text
d_img_i = r_img_i / (||r_img_i||_2 + eps)
```

所有 image tokens 都可以产生候选方向：

```text
D_img = {d_img_i}_{i=1}^n
D_img in R^{n x d}
```

每个 `d_img_i` 表示当前图像中一个由局部 visual token 诱导出来的 latent hypothesis direction。

## 4. 从当前文本生成 text-conditioned hypothesis directions

如果当前输入包含文本 instruction 或问题 `u`，设它的 text token embeddings 为：

```text
H = {h_t}_{t=1}^m
```

其中 `m` 是文本 token 数量，`h_t in R^d`。

去掉特殊 tokens，例如 image placeholder、开始符、结束符等。剩余有效文本 token 的 index 集合记为 `U`。

对于每个有效文本 token，先做 layer normalization：

```text
h_tilde_t = LN(h_t)
```

再做 L2 normalization：

```text
d_text_t = h_tilde_t / (||h_tilde_t||_2 + eps)
```

得到 text-conditioned hypothesis directions：

```text
D_text = {d_text_t}_{t in U}
```

如果没有文本 instruction，则：

```text
D_text = empty
```

## 5. 构造当前输入的 hypothesis candidate pool

将 image-conditioned directions 和 text-conditioned directions 合并：

```text
D = concat_rows(D_img, D_text)
```

设合并后的 candidate 数量为 `M`：

```text
D = {d_j}_{j=1}^M
D in R^{M x d}
```

这里的每一个 `d_j` 都是当前输入生成的 candidate hypothesis direction。

## 6. 计算每个 candidate hypothesis 的初始图像支持分数

对于每个 candidate hypothesis `d_j`，计算它的 global support：

```text
g_j = dot(z_img, d_j)
```

这个值表示整张图的全局表示和 hypothesis `d_j` 的相似度。

然后计算它的 local support：

```text
l_j = max_i dot(z_i, d_j)
```

这个值表示图像中最支持 hypothesis `d_j` 的局部 token response。

分别对 global support 和 local support 做 softmax：

```text
p_g_j = exp(g_j) / sum_q exp(g_q)
p_l_j = exp(l_j) / sum_q exp(l_q)
```

然后将两者相乘并归一化，得到 candidate hypothesis 的初始图像支持分数：

```text
omega_j = ((p_g_j + eps) * (p_l_j + eps)) /
          sum_q ((p_g_q + eps) * (p_l_q + eps))
```

`omega_j` 越大，说明 candidate hypothesis `d_j` 同时获得了较强的全局支持和局部支持。

## 7. Diversity-aware candidate hypothesis selection

设最大 candidate hypothesis 数为：

```text
K_b = min(K_max, n, M)
```

其中：

- `K_max` 是最大 token budget。
- `n` 是原始 image token 数量。
- `M` 是 candidate pool 的大小。

首先选择初始支持分数最高的 candidate：

```text
j_1 = argmax_j omega_j
S_1 = {j_1}
```

之后逐个选择新的 candidate hypothesis。假设当前已经选择了：

```text
S_{t-1} = {j_1, j_2, ..., j_{t-1}}
```

对于一个未选择的 candidate `j`，计算它和已有 candidates 的最小距离：

```text
d_div_j = min_{q in S_{t-1}} (1 - dot(d_j, d_q))
```

然后选择：

```text
j_t = argmax_{j not in S_{t-1}} omega_j * d_div_j
S_t = S_{t-1} union {j_t}
```

重复这个过程，直到选出 `K_b` 个 candidate hypotheses。

最终得到 candidate hypothesis set：

```text
B_c = {d_j : j in S_{K_b}}
B_c in R^{K_b x d}
```

## 8. 计算 candidate response matrix

使用 candidate hypothesis set `B_c`，计算每个 image token 对每个 candidate hypothesis 的 response：

```text
Y_c = X_tilde * B_c^T
Y_c in R^{n x K_b}
```

元素 `Y_c[i, j]` 表示第 `i` 个 image token 对第 `j` 个 candidate hypothesis 的 response。

## 9. 对 candidate responses 做列标准化

对于 `Y_c` 的第 `j` 列，计算均值和标准差：

```text
mu_c_j = (1 / n) * sum_i Y_c[i, j]
sigma_c_j = sqrt((1 / n) * sum_i (Y_c[i, j] - mu_c_j)^2)
```

然后标准化：

```text
Yhat_c[i, j] = (Y_c[i, j] - mu_c_j) / (sigma_c_j + eps)
```

得到标准化后的 candidate response matrix：

```text
Yhat_c in R^{n x K_b}
```

## 10. 计算 token-level support

对每个 token `i`，在所有 candidate hypotheses 上做 softmax：

```text
A[i, j] = exp(Yhat_c[i, j]) / sum_l exp(Yhat_c[i, l])
```

`A[i, j]` 表示 token `i` 分配给 hypothesis `j` 的支持比例。

然后对所有 tokens 求平均，得到 hypothesis `j` 的 token-level support：

```text
s_j = (1 / n) * sum_i A[i, j]
```

因此：

```text
s_j >= 0
sum_j s_j = 1
```

## 11. 计算 candidate hypothesis 的 image-level confidence

前面已经得到所有 candidate directions 的初始支持分数 `omega_j`。对于被选入 `B_c` 的 candidates，重新归一化它们的 `omega` 分数：

```text
p_j = omega_j / (sum_{l in S_{K_b}} omega_l + eps)
```

这里 `p_j` 表示第 `j` 个 candidate hypothesis 的 image-level confidence。

## 12. 计算 evidence score

对每个 candidate hypothesis，融合 image-level confidence 和 token-level support：

```text
e_j = ((p_j + eps) * (s_j + eps)) /
      sum_l ((p_l + eps) * (s_l + eps))
```

其中 `e_j` 是第 `j` 个 candidate hypothesis 的最终 evidence score。

这个分数越大，说明该 hypothesis 越应该被保留到后续 response space 中。

## 13. 动态决定输出 token 数 K_eff

将 evidence scores 从大到小排序：

```text
e_(1) >= e_(2) >= ... >= e_(K_b)
```

然后找到最小的 `K_rho`，使得前 `K_rho` 个 hypotheses 的累计 evidence 达到比例 `rho`：

```text
K_rho = min { k : sum_{j=1}^k e_(j) >= rho }
```

定义最小 token 数：

```text
K_min_bar = min(K_min, K_b)
```

最终输出 token 数为：

```text
K_eff = clip(K_rho, K_min_bar, K_b)
```

也就是：

```text
K_eff = max(K_min_bar, min(K_rho, K_b))
```

## 14. 选择 active hypotheses

根据 evidence score 选择 top `K_eff` 个 hypotheses：

```text
J_a = TopK(e, K_eff)
```

得到 active hypothesis set：

```text
B_a = {B_c[j] : j in J_a}
B_a in R^{K_eff x d}
```

## 15. 构建最终 response space

使用 active hypotheses 重新计算 response matrix：

```text
Y = X_tilde * B_a^T
Y in R^{n x K_eff}
```

对 `Y` 的每一列做标准化：

```text
mu_j = (1 / n) * sum_i Y[i, j]
sigma_j = sqrt((1 / n) * sum_i (Y[i, j] - mu_j)^2)
Yhat[i, j] = (Y[i, j] - mu_j) / (sigma_j + eps)
```

第 `i` 个 token 的 response pattern 是：

```text
Yhat_i = [Yhat[i, 1], Yhat[i, 2], ..., Yhat[i, K_eff]]
```

将它 L2 normalize：

```text
q_i = Yhat_i / (||Yhat_i||_2 + eps)
q_i in R^{K_eff}
```

`q_i` 表示 token `i` 在 active hypothesis response space 中的表示。

## 16. 计算 token importance

对于每个 token `i`，计算它对 active hypotheses 的最大 response：

```text
max_response_i = max_j Yhat[i, j]
```

再计算它对所有 active hypotheses 的平均 response：

```text
mean_response_i = (1 / K_eff) * sum_j Yhat[i, j]
```

token importance 定义为：

```text
v_i = max_response_i - mean_response_i
```

然后对所有 token importance 做标准化：

```text
v_tilde_i = (v_i - mean(v)) / (std(v) + eps)
```

`v_tilde_i` 后续用于 cluster center update 和 prototype aggregation。

## 17. 初始化 clusters

需要形成 `K_eff` 个 clusters，每个 active hypothesis 初始化一个 cluster。

对于第 `j` 个 active hypothesis，选择对它 standardized response 最大的 token 作为 seed：

```text
a_j = argmax_{i not in A_{j-1}} Yhat[i, j]
A_j = A_{j-1} union {a_j}
```

第 `j` 个 cluster 的初始 center 是：

```text
mu_j^(0) = q_{a_j}
```

## 18. 在 response space 中进行 clustering

聚类使用的是 `q_i`，不是原始 projected feature `x_i`。

最大迭代次数根据平均 cluster size 决定：

```text
T_max = 1 + ceil(log2(n / K_eff))
```

因为 `K_eff <= n`，所以 `n / K_eff >= 1`。

### 18.1 Token assignment

在第 `t` 轮，对于每个 token `i`，将它分配给最相似的 cluster center：

```text
c_i^(t) = argmax_j dot(q_i, mu_j^(t))
```

第 `j` 个 cluster 的 token 集合为：

```text
C_j^(t) = {i : c_i^(t) = j}
```

### 18.2 Empty cluster repair

如果某个 cluster 为空，则需要修复。

对于每个非空 cluster，计算内部离散程度：

```text
D_j^(t) = (1 / |C_j^(t)|) * sum_{i in C_j^(t)} (1 - dot(q_i, mu_j^(t)))
```

选择最分散且 token 数大于 1 的 cluster：

```text
j_star = argmax_{j : |C_j^(t)| > 1} D_j^(t)
```

在该 cluster 中选择最不匹配当前中心的 token：

```text
i_star = argmin_{i in C_j_star^(t)} dot(q_i, mu_j_star^(t))
```

将 `i_star` 从 cluster `j_star` 移到空 cluster 中。

### 18.3 Cluster center update

每个 token 的 center update 权重为：

```text
w_i = exp(v_tilde_i)
```

对于第 `j` 个 cluster，更新中心：

```text
mu_j^(t+1) =
    sum_{i in C_j^(t)} w_i * q_i /
    (||sum_{i in C_j^(t)} w_i * q_i||_2 + eps)
```

重复 assignment、empty cluster repair 和 center update，直到完成 `T_max` 轮。

最终得到每个 token 的 cluster assignment `c_i`，以及最终 cluster center `mu_j`。

## 19. 在原始 projector feature space 中聚合 tokens

虽然聚类是在 response space 中完成的，但最终输出 token 必须来自原始 projected visual feature space。

对于第 `j` 个 cluster，它包含的 token 集合为：

```text
C_j = {i : c_i = j}
```

对于 cluster `j` 中的每个 token `i`，计算 aggregation logit：

```text
ell_ij = dot(q_i, mu_j) + v_tilde_i
```

其中：

- `dot(q_i, mu_j)` 表示 token `i` 和 cluster center 的 response-space similarity。
- `v_tilde_i` 表示 token importance。

在 cluster 内做 softmax：

```text
alpha_ij = exp(ell_ij) / sum_{k : c_k = j} exp(ell_kj)
```

然后使用这些权重聚合原始 projected image tokens：

```text
r_j = sum_{i : c_i = j} alpha_ij * x_i
```

这里的 `r_j` 仍然是 `d`-dimensional projected visual token。

## 20. 得到 reduced visual token sequence

对所有 clusters 执行聚合，得到：

```text
R = {r_j}_{j=1}^{K_eff}
R in R^{K_eff x d}
```

## 21. 根据 soft spatial position 排序

如果原始 image tokens 有空间位置：

```text
Pi = {pi_i}_{i=1}^n
pi_i = (x_pos_i, y_pos_i)
```

那么每个 prototype token 的 soft position 为：

```text
pi_j = sum_{i : c_i = j} alpha_ij * pi_i
```

然后按照 raster order 对 prototypes 排序：

1. 先按纵向位置从上到下排序。
2. 同一行内按横向位置从左到右排序。

排序后的 prototype sequence 记为 `R_sorted`。

如果没有空间位置信息，则直接令：

```text
R_sorted = R
```

代码实现中，如果没有显式传入 positions，但 `n` 是平方数，会自动构造 `sqrt(n) x sqrt(n)` 网格位置。LLaVA-1.5 常见的 576 visual tokens 会被视为 `24 x 24` 网格。

## 22. 替换原始 image tokens

原始送入 LLM 的序列可以表示为：

```text
[text tokens; X]
```

FORESIGHT 处理后变成：

```text
[text tokens; R_sorted]
```

其中：

```text
X in R^{n x d}
R_sorted in R^{K_eff x d}
```

最终，LLM 接收压缩后的 visual prototype tokens，并继续完成后续生成或推理。
