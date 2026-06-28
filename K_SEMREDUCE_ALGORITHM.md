# K-SemReduce 算法说明

本文档说明当前仓库实现的 **K-SemReduce: Class-Prototype Guided
Training-Free Visual Token Reduction**。

本版本的核心变化是把原来分开的 `k`、`m`、`b` 统一成一个核心变量
`K`：

```text
K = Top-K candidate semantic classes 的数量
K = semantic response 维度
K = 聚类中心数量
K = 最终输出 prototype tokens 数量
```

因此当前版本不再有：

```text
m
b
TopB
protected anchors
lambda_div
```

## 1. 输入和输出

视觉编码器某一层产生 token 序列：

```text
[CLS, x1, x2, ..., xn]
```

其中：

```text
n  = 原始 patch token 数量
xi = 第 i 个 patch token
d  = hidden dimension
```

K-SemReduce 把 `n` 个 patch tokens 压缩成 `K` 个 prototype tokens：

```text
[CLS, r1, r2, ..., rK]
```

每个 `rj` 仍然是视觉 token，不是类别 token。它只是由第 `j` 个候选语义类别引导出的一个 visual prototype token。

## 2. 符号

```text
I:
    输入图像

F_{1:L}:
    frozen visual encoder，一共有 L 层 Transformer

ell:
    插入 token reduction 的层数

C:
    classifier head 的总类别数

K:
    Top-K candidate semantic classes 数量

W_cls:
    frozen classifier head，shape 是 [C, d]

T:
    聚类迭代次数

tau:
    prototype aggregation 的 softmax temperature

lambda_imp:
    聚合阶段 semantic importance 的权重

gamma:
    center update 阶段 semantic importance 的权重

eps:
    数值稳定项
```

## 3. 总体流程

```text
I
-> 前 ell 层视觉编码器
-> 得到 [CLS, x1, ..., xn]
-> 用 CLS token 选择 Top-K candidate semantic classes
-> 计算每个 patch 对这 K 个类别的 semantic response
-> 每个 candidate class 选择一个最强响应 patch 作为 seed
-> 得到 K 个 non-duplicate class-guided seeds
-> 用 seeds 初始化 K 个 semantic centers
-> 所有 patch tokens 参与 semantic clustering
-> 每个 cluster 聚合成一个 visual prototype token
-> 得到 [CLS, r1, ..., rK]
-> 输入后续视觉编码器或 VLM 后续模块
```

## 4. 选择 Top-K candidate semantic classes

先对 CLS token 做 normalization：

```text
x_cls_norm = LN(x_cls)
```

用 frozen classifier head 得到图像级语义响应：

```text
g = W_cls x_cls_norm
```

其中：

```text
W_cls: [C, d]
g:     [C]
```

取 Top-K 类别：

```text
S = TopK(g, K)
```

再取这些类别对应的 classifier rows：

```text
W_S = W_cls[S]
```

所以：

```text
W_S: [K, d]
```

## 5. Patch-level semantic response

对每个 patch token：

```text
x_i_norm = LN(x_i)
p_i = W_S x_i_norm
```

其中：

```text
p_i: [K]
```

把所有 patch 放在一起：

```text
P: [n, K]
```

`P[i, j]` 表示第 `i` 个 patch 对第 `j` 个 candidate semantic class 的响应。

两个 patch 是否应该合并，不直接看 `xi` 和 `xj` 的原始视觉相似度，而是看它们在当前 Top-K semantic classes 上的响应模式是否相似。

## 6. 标准化和 L2 normalization

不同类别的 logit 尺度可能不同，所以先对每个 semantic dimension 做标准化：

```text
mean_j = mean over i of P[i, j]
std_j  = std over i of P[i, j]

P_hat[i, j] = (P[i, j] - mean_j) / (std_j + eps)
```

得到：

```text
P_hat: [n, K]
```

然后对每个 patch 的 semantic response vector 做 L2 normalize：

```text
q_i = P_hat[i] / (||P_hat[i]||_2 + eps)
```

所有 patch 合起来：

```text
Q: [n, K]
```

后续聚类使用 `q_i`，不是原始视觉 token `x_i`。

## 7. Semantic importance

虽然没有 TopB 和 protected anchors，semantic importance 仍然保留。它用于 center update 和 prototype aggregation。

对每个 patch：

```text
u_i = max_j P_hat[i, j] - mean_j P_hat[i, j]
```

直觉是：如果一个 patch 对某个候选类别特别强，而不是对所有类别都差不多强，那么它更有判别性。

然后对所有 patch 的 `u_i` 做标准化：

```text
u_tilde_i = (u_i - mean(u)) / (std(u) + eps)
```

## 8. Non-duplicate class-guided seed selection

对每个 candidate class，都选一个最支持它的 patch 作为 seed。

普通写法是：

```text
a_j = argmax_i P_hat[i, j]
```

但不同 classes 可能选到同一个 patch。当前版本要求 seed 数必须等于 K，所以使用 non-duplicate seed selection：

```text
A = empty set

for j = 1 to K:
    a_j = argmax over i not in A of P_hat[i, j]
    A = A union {a_j}
```

这样得到：

```text
A = {a1, a2, ..., aK}
```

并且：

```text
|A| = K
```

如果用户设置 `K > n`，代码会自动使用：

```text
K = n
```

因为不能从 `n` 个 patch 里选出超过 `n` 个不同 seeds。

## 9. 初始化 K 个 semantic centers

每个 seed patch 对应一个 semantic center：

```text
mu_j = q_{a_j}
```

所以：

```text
Q:       [n, K]
centers: [K, K]
```

第一个 K 是 cluster 数量，第二个 K 是 semantic response vector 的维度。

## 10. Semantic clustering

对每个 patch，计算它和每个 center 的相似度：

```text
sim(i, j) = q_i dot mu_j
```

然后分配到最相似的 center：

```text
cluster_id(i) = argmax_j q_i dot mu_j
```

得到 K 个 clusters：

```text
C_1, C_2, ..., C_K
```

所有 patch tokens 都参与 clustering，包括被选为 seeds 的 patches。当前版本没有 singleton anchors。

## 11. Center update

普通 k-means 会直接平均 cluster 内的 `q_i`。K-SemReduce 使用 semantic importance 加权：

```text
w_i = exp(gamma * u_tilde_i)
mu_j = Normalize(sum over i in C_j of w_i * q_i)
```

重复下面过程 `T` 次：

```text
assign patches to centers
repair empty clusters
update centers
```

默认：

```text
T = 3
```

## 12. Empty cluster repair

如果某个 cluster 没有 patch：

```text
C_j = empty
```

最终就会少于 K 个 prototype tokens，所以必须修复。

当前实现使用 donor-cluster repair：

```text
1. 找到 patch 数大于 1 的 donor cluster。
2. 选择内部分散程度最大的 donor。
3. 从 donor 中移动一个最不适合 donor center 的 patch。
4. 把这个 patch 分给 empty cluster。
```

这样保证：

```text
每个 cluster 非空
每个 patch 只属于一个 cluster
```

## 13. Prototype aggregation

聚类在 semantic response space 完成，但输出 token 必须回到原始 visual hidden space。

对 cluster `C_j` 内每个 patch：

```text
score_i = q_i dot mu_j + lambda_imp * u_tilde_i
```

在 cluster 内做 softmax：

```text
alpha_i = Softmax(score_i / tau)
```

最后聚合原始视觉 token：

```text
r_j = sum over i in C_j of alpha_i * x_i
```

所以：

```text
r_j: [d]
R:   [K, d]
```

注意 `q_i` 是 K 维 semantic vector，`x_i` 是 d 维 visual token，最终输出必须是 d 维的 `r_j`。

## 14. Soft position aggregation

如果没有提供 patch positions，代码会自动构造 positions：

```text
如果 n 是平方数，构造 2D grid。
否则构造一行 positions。
```

每个 prototype 的 soft position 是：

```text
pi_j = sum over i in C_j of alpha_i * pi_i
```

最后按照 raster order 排序：

```text
top-left -> bottom-right
```

## 15. 当前代码里的 LLaVA 实现

理论算法写的是在视觉编码器第 `ell` 层插入 reduction：

```text
X_l = F_{1:ell}(I)
X_reduced = [x_cls, r1, ..., rK]
H_L = F_{ell+1:L}(X_reduced)
```

当前 Hugging Face LLaVA runner 的实际插入点是：

```text
after vision tower and multimodal projector
```

原因是 LLaVA-1.5 的公开接口稳定暴露的是 projector 之后的 image features。代码会先得到这些 image features，再运行 K-SemReduce，然后把 prompt 里的 `<image>` token 数改成 reduced token 数。

LLaVA 没有 ImageNet 风格的视觉 classifier head。因此 runner 使用和 image feature hidden dim 匹配的 frozen language embedding 或 LM head 作为 surrogate semantic head。算法流程仍然是：

```text
Top-K semantic rows
-> K class-guided seeds
-> K semantic centers
-> K clusters
-> K visual prototype tokens
```

## 16. 当前默认超参数

```text
K = 64
T = 3
tau = 0.1
lambda_imp = 0.25
gamma = 1.0
eps = 1e-6
```

推荐消融：

```text
K = 32, 64, 96, 128
T = 1, 2, 3, 5
lambda_imp = 0, 0.25
```

## 17. 最短伪代码

```text
function KSemReduce(X, W_cls, K):
    x_cls = mean(X) if no CLS token is available
    S = TopK(W_cls LN(x_cls), K)
    W_S = W_cls[S]

    P = LN(X) W_S^T
    P_hat = standardize each column of P
    Q = L2 normalize each row of P_hat
    u = standardize(max(P_hat, dim=class) - mean(P_hat, dim=class))

    seeds = []
    for class j in 1..K:
        seed_j = argmax over unused patch i of P_hat[i, j]
        seeds.append(seed_j)

    centers = Q[seeds]

    repeat T times:
        assignments = argmax_j Q centers^T
        assignments = repair empty clusters
        centers = importance weighted center update

    for each cluster:
        scores = Q_members center + lambda_imp * u_members
        weights = softmax(scores / tau)
        prototype = weighted sum of original visual tokens

    sort prototypes by soft position
    return prototypes
```

