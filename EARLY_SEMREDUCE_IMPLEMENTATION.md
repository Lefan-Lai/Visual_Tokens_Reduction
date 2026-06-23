# Early-SemReduce Code Implementation Notes

本文档专门说明 **代码里到底怎样实现 Early-SemReduce**。它和
`EARLY_SEMREDUCE_ALGORITHM.md` 的区别是：

- `EARLY_SEMREDUCE_ALGORITHM.md` 解释方法的数学动机、公式和理论流程。
- 本文档解释代码结构、函数调用链、张量形状、每个 helper 做什么、LLaVA
  实验脚本怎样把 token reduction 接进模型 forward。

当前实现代码位于 `Lefan-Lai/SEER` 的
`Visual_Token_Merge` 目录：

```text
Visual_Token_Merge/
  early_semreduce/
    __init__.py
    reducer.py
    vit_wrapper.py
  scripts/
    demo_semreduce.py
  tests/
    test_early_semreduce.py
  run_llava13b_pope.py
  README.md
```

最核心的文件是：

```text
early_semreduce/reducer.py
```

它实现了训练无关的 semantic response guided token merge。LLaVA 13B POPE
实验入口是：

```text
run_llava13b_pope.py
```

这个脚本负责加载 POPE、加载 LLaVA 13B、运行 vanilla 和 Early-SemReduce
两种方法、保存 JSONL 和 summary 指标。

---

## 1. 实现整体分层

代码可以分成四层：

```text
Layer 1: 核心 token reduction
    early_semreduce/reducer.py

Layer 2: 标准 ViT 插入 helper
    early_semreduce/vit_wrapper.py

Layer 3: LLaVA 13B POPE runner
    run_llava13b_pope.py

Layer 4: demo 和 tests
    scripts/demo_semreduce.py
    tests/test_early_semreduce.py
```

各层职责如下。

### 1.1 `reducer.py`

这是算法主体。它不依赖 LLaVA，也不依赖 timm。只要输入 patch tokens 和一个
frozen semantic classifier weight，它就可以把 `n` 个 patch tokens 合成 `m`
个 prototype tokens。

核心 API 有两个：

```python
class EarlySemReduce(nn.Module):
    def forward(
        self,
        sequence_tokens,
        classifier,
        positions=None,
        cls_index=0,
    ) -> SemReduceResult:
        ...
```

以及函数式接口：

```python
def early_semreduce(
    patch_tokens,
    classifier,
    cls_token=None,
    config=None,
    token_norm=None,
    positions=None,
    **config_overrides,
) -> SemReduceResult:
    ...
```

二者的区别：

- `EarlySemReduce.forward` 接收的是完整 sequence，也就是 `[CLS, patch_1,
  ..., patch_n]`。
- `early_semreduce` 接收的是 patch-only tokens，也就是 `[patch_1, ...,
  patch_n]`，可选传入 `cls_token`。

### 1.2 `vit_wrapper.py`

这个文件负责把 Early-SemReduce 插进 timm 风格 ViT 模型的某一层之后。它的
目标是真正的 early reduction：

```text
patch_embed -> position tokens -> blocks[:ell] -> EarlySemReduce -> blocks[ell:] -> head
```

核心函数是：

```python
forward_timm_vit_with_semreduce(
    model,
    images,
    reducer,
    reduction_layer,
    classifier=None,
    return_info=False,
)
```

这个 helper 适合标准 ViT 分类模型，因为标准 ViT 通常有：

```text
model.patch_embed
model.blocks
model.norm
model.head
```

### 1.3 `run_llava13b_pope.py`

这个文件是实验 runner。它不是纯算法文件，而是把算法接进 Hugging Face
LLaVA 的推理流程中。

LLaVA 的公开接口不容易直接在 vision tower 中间层插入 reduction，所以这个
runner 的接入点是：

```text
vision tower + multimodal projector 之后
language model 接收 image features 之前
```

也就是说，对于 LLaVA runner，这个实现主要减少 language model 看到的 image
tokens 数量，并用于比较 hallucination / POPE 指标；它不等价于在 LLaVA
vision tower 内部某个 block 后做真正的 early vision reduction。

这点非常重要：

```text
vit_wrapper.py:
    支持真正的 ViT block-level early reduction。

run_llava13b_pope.py:
    对 Hugging Face LLaVA 的 image features 做后处理压缩。
```

---

## 2. 核心配置对象 `SemReduceConfig`

代码用 dataclass 保存所有超参数：

```python
@dataclass(frozen=True)
class SemReduceConfig:
    num_prototypes: int
    candidate_classes: int | None = 64
    num_anchors: int = 8
    iterations: int = 5
    temperature: float = 0.07
    lambda_importance: float = 0.25
    lambda_diversity: float = 1.0
    gamma: float = 1.0
    eps: float = 1e-6
    sort_by_position: bool = True
```

每个字段在代码中的具体作用如下。

### 2.1 `num_prototypes`

目标输出 patch prototype 数量，也就是压缩后的 `m`。

如果输入 patch token 数是 `n`，代码里会做：

```python
target = min(int(cfg.num_prototypes), num_patches)
```

所以如果 `num_prototypes >= n`，不会强行扩增 token，而是直接进入
no-reduction 分支。

### 2.2 `candidate_classes`

候选语义类别数量，也就是从 frozen classifier head 里选 top-k 行。

代码行为：

```python
if candidate_classes is None or int(candidate_classes) >= num_classes:
    selected = torch.arange(num_classes, device=patch_tokens.device)
else:
    selected = torch.topk(logits, k=max(1, int(candidate_classes)), dim=-1).indices
```

含义：

- 如果是 `None`，使用所有类别。
- 如果大于等于总类别数，也使用所有类别。
- 否则根据 CLS token 的 logits 选 top-k。

### 2.3 `num_anchors`

保留的 semantic anchor token 数量。代码会把它裁剪到合法范围：

```python
num_anchors = min(max(int(cfg.num_anchors), 0), target)
```

所以：

- 小于 0 会被当成 0。
- 大于 target 会被裁成 target。

anchor 的作用是把最重要的 patch 直接作为 singleton cluster，不参与平均合并。

### 2.4 `iterations`

semantic clustering 的 hard assignment / center update 轮数。

代码里使用：

```python
for _ in range(max(0, int(iterations))):
    ...
```

如果设成 0，仍然会完成初始化、最后 assignment 和 empty cluster repair。

### 2.5 `temperature`

最终 cluster 内 soft pooling 的温度。

代码位置：

```python
weights = torch.softmax(scores / float(temperature), dim=0)
```

温度越小，权重越尖锐；温度越大，越接近平均池化。

### 2.6 `lambda_importance`

最终聚合 prototype 时，semantic importance 对权重的影响。

代码位置：

```python
scores = semantic_scores + float(lambda_importance) * importance[member_indices]
```

如果设为 0，prototype 聚合只看 patch 和 cluster center 的 semantic
similarity。

### 2.7 `lambda_diversity`

初始化 semantic centers 时，多样性项的权重。

代码位置：

```python
scores = u_subset + float(lambda_diversity) * min_distance
```

这里 `u_subset` 是 patch importance，`min_distance` 是该 patch 到已有
centers 的最近语义距离。这个设计让初始化既偏向重要 patch，又避免所有中心挤
在同一种语义模式附近。

### 2.8 `gamma`

更新 cluster center 时，importance weighting 的强度。

代码位置：

```python
weights = torch.exp(float(gamma) * u_subset[member_mask]).unsqueeze(-1)
weighted = (weights * q_subset[member_mask]).sum(dim=0, keepdim=True)
```

如果 `gamma = 0`，所有 cluster 内成员权重相同；如果 `gamma > 0`，重要
patch 对 center 更新影响更大。

### 2.9 `eps`

数值稳定项，用在：

- response 标准化的分母。
- L2 normalize。
- importance 标准化。

### 2.10 `sort_by_position`

是否根据 soft position 对 prototype token 排序。

如果为 `True`，且有位置坐标，代码会：

```python
order = _position_order(soft_positions)
prototypes = prototypes[order]
masses = masses[order]
soft_positions = soft_positions[order]
assignments = _remap_assignments(assignments, order)
```

排序后 prototype 顺序更接近原图 raster order。

---

## 3. 输出对象 `SemReduceResult`

reducer 不只返回压缩后的 tokens，还返回完整 bookkeeping 信息：

```python
@dataclass
class SemReduceResult:
    sequence: torch.Tensor | None
    patch_tokens: torch.Tensor
    assignments: torch.Tensor
    centers: torch.Tensor
    selected_classes: torch.Tensor
    anchors: torch.Tensor
    masses: torch.Tensor
    soft_positions: torch.Tensor | None
    prototype_order: torch.Tensor
```

每个字段含义如下。

### 3.1 `sequence`

如果输入是完整 sequence，也就是 `[CLS, patches]`，则返回：

```text
[CLS, prototype_1, ..., prototype_m]
```

shape 通常是：

```text
[B, 1 + m, D]
```

如果用的是 patch-only 函数 `early_semreduce`，这里是 `None`。

### 3.2 `patch_tokens`

压缩后的 prototype patch tokens。

shape：

```text
[m, D]
```

或者 batch 输入时：

```text
[B, m, D]
```

### 3.3 `assignments`

每个原始 patch 被分配到哪个 prototype。

shape：

```text
[n]
```

或者：

```text
[B, n]
```

它用于调试和可视化。例如可以看哪些原始 patch 被合并成同一个 prototype。

### 3.4 `centers`

普通 semantic clusters 的最终中心。

注意它不是视觉 token 空间里的中心，而是 semantic response space 里的中心。

shape：

```text
[m - b, k]
```

其中 `b` 是 anchor 数，`k` 是 candidate class 数。

### 3.5 `selected_classes`

本张图像选出来的候选类别 index。

shape：

```text
[k]
```

或者 batch：

```text
[B, k]
```

### 3.6 `anchors`

被保护为 singleton cluster 的原始 patch index。

shape：

```text
[b]
```

### 3.7 `masses`

每个 prototype 代表多少个原始 patch。

shape：

```text
[m]
```

所有 mass 加起来应该等于原始 patch 数：

```text
masses.sum() == n
```

测试里专门检查了这一点。

### 3.8 `soft_positions`

每个 prototype 的 soft 2D 坐标。如果没有显式传入 positions，代码会自动构造：

- 如果 `n` 是平方数，构造二维 grid。
- 否则构造一维行坐标。

shape：

```text
[m, 2]
```

### 3.9 `prototype_order`

如果启用了 position sorting，记录排序前到排序后的顺序。

---

## 4. `EarlySemReduce.forward` 的实际执行流程

入口：

```python
result = reducer(sequence_tokens, classifier)
```

输入可以是：

```text
[N + 1, D]
[B, N + 1, D]
```

其中第一维 token 必须是 CLS。

代码先检查：

```python
if cls_index != 0:
    raise ValueError("EarlySemReduce currently expects the CLS token at index 0")
```

所以当前实现不支持 CLS 在其它位置。

### 4.1 单张图像

如果 `sequence_tokens.ndim == 2`，直接调用：

```python
return self._forward_sequence_single(sequence_tokens, classifier, positions)
```

在 `_forward_sequence_single` 里：

```python
cls_token = sequence_tokens[0]
patch_tokens = sequence_tokens[1:]
```

然后调用核心函数：

```python
reduced = early_semreduce(
    patch_tokens=patch_tokens,
    classifier=classifier,
    cls_token=cls_token,
    config=self.config,
    token_norm=self._norm,
    positions=positions,
)
```

最后把 CLS token 拼回去：

```python
reduced.sequence = torch.cat([cls_token.unsqueeze(0), reduced.patch_tokens], dim=0)
```

### 4.2 batch 输入

如果输入是 `[B, N + 1, D]`，代码没有把整个 batch 混在一起聚类，而是逐张图像
独立 reduce：

```python
per_batch = []
for batch_index, batch_tokens in enumerate(sequence_tokens):
    batch_positions = None if positions is None else positions[batch_index]
    per_batch.append(self._forward_sequence_single(batch_tokens, classifier, batch_positions))
return _stack_results(per_batch, include_sequence=True)
```

这样做的原因是每张图像的 semantic response、top-k 类别、anchors 和 clusters
都应该独立计算，不能跨图像聚类。

---

## 5. `early_semreduce` 函数式接口

函数式接口适合已经拿到了 patch tokens 的情况：

```python
result = early_semreduce(
    patch_tokens=features,
    classifier=classifier,
    config=config,
)
```

它支持：

```text
[N, D]
[B, N, D]
```

和 sequence 版本一样，batch 输入会逐张图像处理，然后 `_stack_results`。

如果没有传 `config`，则必须通过 keyword overrides 给出 `num_prototypes`：

```python
early_semreduce(
    patch_tokens,
    classifier,
    num_prototypes=128,
    candidate_classes=64,
)
```

内部由 `_resolve_config` 合成 `SemReduceConfig`。

---

## 6. `_early_semreduce_single`: 核心主流程

真正的单样本算法主流程在：

```python
def _early_semreduce_single(...):
    ...
```

它按下面顺序执行。

### 6.1 配置和输入检查

先调用：

```python
_validate_config(cfg)
```

检查：

```text
num_prototypes > 0
temperature > 0
candidate_classes is None or candidate_classes > 0
```

然后检查 patch tokens：

```python
if patch_tokens.ndim != 2:
    raise ValueError(...)
if patch_tokens.shape[0] == 0:
    raise ValueError("cannot reduce an empty patch-token sequence")
```

### 6.2 计算目标 token 数

代码：

```python
num_patches = int(patch_tokens.shape[0])
target = min(int(cfg.num_prototypes), num_patches)
```

`target` 是最终 prototype 数。

如果 `target == num_patches`，说明不需要压缩，直接返回原 tokens。这个分支很重要，
因为它避免了在没有压缩需求时仍然做聚类。

no-reduction 分支返回：

```python
patch_tokens=patch_tokens
assignments=torch.arange(num_patches)
centers=q_tokens
masses=torch.ones(num_patches)
prototype_order=torch.arange(num_patches)
```

其中 `centers=q_tokens` 只是为了保持 result 结构完整。

### 6.3 准备 norm 和 classifier weight

代码：

```python
norm_fn = _make_norm(token_norm)
weight = _classifier_weight(classifier, patch_tokens.device)
```

`_make_norm` 的行为：

- 如果用户传了模型自己的 norm，就使用它。
- 如果没传，就使用 parameter-free layer norm：

```python
F.layer_norm(values.float(), values.shape[-1:])
```

`_classifier_weight` 的行为：

- 如果传入的是 `nn.Module`，读取 `.weight`。
- 如果传入的是 Tensor，直接使用。
- 检查必须是二维 `[K, D]`。
- 转到当前 device，并转成 `float32`。

这里转成 float32 是为了让 response 计算和 clustering 更稳定。

---

## 7. `_semantic_response`: 语义响应是怎样算出来的

这是整个方法和普通 ProtoReduce 最大的实现区别。

入口：

```python
q_tokens, p_hat, importance, selected_classes = _semantic_response(...)
```

输入：

```text
patch_tokens:       [n, d]
classifier_weight: [K, d]
cls_token:          [d] or None
```

输出：

```text
q_tokens:          [n, k]
p_hat:             [n, k]
importance:        [n]
selected_classes:  [k]
```

### 7.1 如果没有 CLS token

如果调用方没有传 `cls_token`，代码用 patch mean 代替：

```python
if cls_token is None:
    cls_token = patch_tokens.mean(dim=0)
```

这个 fallback 主要服务于 patch-only 特征输入，比如 LLaVA image features。

### 7.2 用 CLS 选候选类别

先归一化 CLS：

```python
norm_cls = token_norm(cls_token.unsqueeze(0)).to(dtype=torch.float32).squeeze(0)
```

然后计算 CLS 到 classifier rows 的 logits：

```python
logits = norm_cls @ classifier_weight.T
```

shape：

```text
norm_cls:           [d]
classifier_weight: [K, d]
logits:             [K]
```

接着选择 candidate classes：

```python
selected = torch.topk(logits, k=max(1, int(candidate_classes)), dim=-1).indices
```

如果 `candidate_classes=None`，则不 top-k，直接使用所有类别。

### 7.3 计算 patch-level semantic responses

取出候选类别对应的权重：

```python
selected_weight = classifier_weight[selected]
```

shape：

```text
selected_weight: [k, d]
```

归一化 patch tokens：

```python
norm_patch = token_norm(patch_tokens).to(dtype=torch.float32)
```

计算 response：

```python
responses = norm_patch @ selected_weight.T
```

shape：

```text
norm_patch:       [n, d]
selected_weight: [k, d]
responses:       [n, k]
```

`responses[i, c]` 表示第 `i` 个 patch 对第 `c` 个候选语义类别的响应。

### 7.4 标准化 response

代码对每个候选类别，在 patch 维度上做 mean/std：

```python
mean = responses.mean(dim=0, keepdim=True)
std = responses.std(dim=0, keepdim=True, unbiased=False)
p_hat = (responses - mean) / (std + eps)
```

这一步避免某些类别因为 logit 尺度大而支配聚类。

### 7.5 L2 normalize 成 clustering token

代码：

```python
q_tokens = F.normalize(p_hat, p=2, dim=-1, eps=eps)
```

`q_tokens` 是后面聚类用的 token，不是最终输出的视觉 token。

它的 shape 是：

```text
[n, k]
```

注意这里的维度已经从视觉 hidden dim `d` 变成候选类别数 `k`。

### 7.6 计算 semantic importance

代码：

```python
importance = p_hat.max(dim=-1).values - p_hat.mean(dim=-1)
importance = (importance - importance.mean()) / (importance.std(unbiased=False) + eps)
```

它的含义是：

- 如果一个 patch 在某个候选类别上有尖锐响应，它的重要性高。
- 如果一个 patch 对所有候选类别都差不多，它的重要性低。

最后再把 importance 标准化，使它在不同图像之间尺度更稳定。

---

## 8. Anchor 选择和 non-anchor 划分

回到 `_early_semreduce_single`，semantic response 算完之后，代码先选择 anchors：

```python
num_anchors = min(max(int(cfg.num_anchors), 0), target)
anchor_indices = _topk_indices(importance, num_anchors)
```

`_topk_indices` 很简单：

```python
return torch.topk(values, k=k, dim=0).indices
```

如果 `k <= 0`，返回空 tensor。

接着构造 non-anchor mask：

```python
non_anchor_mask = torch.ones(num_patches, dtype=torch.bool, device=patch_tokens.device)
if num_anchors > 0:
    non_anchor_mask[anchor_indices] = False
non_anchor_indices = torch.nonzero(non_anchor_mask, as_tuple=False).squeeze(-1)
```

普通 cluster 数量：

```python
num_clusters = target - num_anchors
```

因此最终输出由两部分组成：

```text
前 b 个: anchor singleton prototypes
后 M 个: semantic clusters 聚合出来的 prototypes
```

---

## 9. `_initialize_centers`: semantic-aware 初始化

如果 `num_clusters > 0`，代码会初始化 ordinary cluster centers：

```python
centers = _initialize_centers(
    q_tokens=q_tokens,
    importance=importance,
    indices=non_anchor_indices,
    num_centers=num_clusters,
    lambda_diversity=cfg.lambda_diversity,
)
```

### 9.1 只在 non-anchor tokens 里初始化

函数开始：

```python
q_subset = q_tokens[indices]
u_subset = importance[indices]
```

所以 anchors 不参与普通聚类。

### 9.2 第一个 center 取最重要 patch

代码：

```python
first = int(torch.argmax(u_subset).item())
selected_local.append(first)
```

这说明第一个普通 center 是 non-anchor 中 semantic importance 最高的 patch。

### 9.3 后续 center 兼顾重要性和多样性

先计算每个 patch 到已有 center 的最小语义距离：

```python
min_distance = 1.0 - (q_subset @ q_subset[first])
```

后续每次选择：

```python
scores = u_subset + float(lambda_diversity) * min_distance
```

并且已经选过的 center 会被排除：

```python
scores[torch.tensor(selected_local, device=scores.device)] = -torch.inf
```

如果所有候选都不可选，代码 fallback 到：

```python
candidate = int(torch.argmax(min_distance).item())
```

否则：

```python
candidate = int(torch.argmax(scores).item())
```

选择新 center 后更新每个 patch 到 selected centers 的最近距离：

```python
candidate_distance = 1.0 - (q_subset @ q_subset[candidate])
min_distance = torch.minimum(min_distance, candidate_distance)
```

最终返回：

```python
return q_subset[selected_local].clone()
```

---

## 10. `_cluster_semantic_tokens`: hard assignment 和 center update

中心初始化后，执行 semantic clustering：

```python
centers, local_assignments = _cluster_semantic_tokens(...)
```

输入：

```text
q_tokens:     [n, k]
importance:  [n]
indices:     [n - b]
centers:     [M, k]
```

函数内部先取 non-anchor subset：

```python
q_subset = q_tokens[indices]
u_subset = importance[indices]
```

### 10.1 Assignment

每轮先算 similarity：

```python
assignments = (q_subset @ centers.T).argmax(dim=-1)
```

因为 `q_subset` 和 `centers` 都是 L2 normalized，所以 dot product 就是 cosine
similarity。

### 10.2 Center update

对每个 center：

```python
member_mask = assignments == center_index
```

如果 cluster 非空，则 importance-aware 更新：

```python
weights = torch.exp(float(gamma) * u_subset[member_mask]).unsqueeze(-1)
weighted = (weights * q_subset[member_mask]).sum(dim=0, keepdim=True)
updated.append(F.normalize(weighted, p=2, dim=-1, eps=eps).squeeze(0))
```

这一步仍然在 semantic response space 里更新 center。

### 10.3 空 cluster 临时重初始化

如果某个 cluster 当前没有成员：

```python
updated.append(_reinitialize_empty_center(q_subset, u_subset, centers, center_index))
```

`_reinitialize_empty_center` 会选择：

```text
importance + diversity
```

都比较高的 patch 作为新 center。

具体代码：

```python
other_centers = centers[other_indices]
diversity = (1.0 - q_subset @ other_centers.T).min(dim=-1).values
candidate = torch.argmax(u_subset + diversity)
return q_subset[candidate]
```

### 10.4 最终 assignment 和 repair

循环结束后，代码再做一次最终 assignment：

```python
assignments = (q_subset @ centers.T).argmax(dim=-1)
assignments = _repair_empty_assignments(q_subset, u_subset, centers, assignments)
```

为什么还需要 repair？

因为即使更新 center 时处理了空 cluster，最后一次 assignment 后仍可能出现空
cluster。`_repair_empty_assignments` 保证每个 ordinary cluster 至少拿到一个
patch。

---

## 11. `_repair_empty_assignments`: 怎样修空 cluster

函数先统计每个 center 的成员数量：

```python
counts = torch.bincount(assignments, minlength=num_centers)
empty_centers = torch.nonzero(counts == 0, as_tuple=False).flatten()
```

如果没有空 cluster，直接返回。

如果有空 cluster，则循环修复：

```python
for empty_center in empty_centers.tolist():
    donor_mask = counts[repaired] > 1
    donor_indices = torch.nonzero(donor_mask, as_tuple=False).flatten()
```

只从成员数大于 1 的 donor cluster 里拿 patch，避免把另一个 cluster 拿空。

然后计算 donor patch 的当前拟合程度：

```python
current_similarity = (q_subset * centers[repaired]).sum(dim=-1)
donor_scores = u_subset - current_similarity
```

这个分数越高，表示：

- patch 本身比较重要。
- 但它对当前 center 的 similarity 不是特别高。

于是最适合被挪出去单独补空 cluster：

```python
moved_local = donor_indices[torch.argmax(donor_scores[donor_indices])]
```

然后更新 assignment 和 counts：

```python
counts[repaired[moved_local]] -= 1
repaired[moved_local] = int(empty_center)
counts[empty_center] += 1
```

这个 repair 逻辑的目标不是重新找到全局最优聚类，而是保证输出 prototype 数量
稳定等于 `target`。

---

## 12. `_aggregate_prototypes`: 怎样生成最终视觉 tokens

聚类完成后，代码调用：

```python
prototypes, masses, assignments, soft_positions = _aggregate_prototypes(...)
```

这个函数非常关键，因为它把 semantic clustering 结果转回 original visual token
space。

### 12.1 准备 positions

先调用：

```python
prepared_positions = _prepare_positions(positions, num_patches, device)
```

如果用户没传 positions：

```python
side = math.isqrt(num_patches)
if side * side == num_patches:
    构造 side x side grid
else:
    构造一维 row=0, col=0..n-1
```

所以对常见 ViT patch 数，比如 196，代码会自动构造 14 x 14 grid。

### 12.2 Anchor prototypes

对于每个 anchor patch：

```python
prototypes.append(patch_tokens[patch_index])
masses.append(torch.tensor(1))
assignments[patch_index] = output_index
```

如果有 positions，则直接使用 anchor 原位置：

```python
soft_positions.append(prepared_positions[patch_index])
```

anchor 不做平均、不做 softmax，就是原 patch token。

### 12.3 普通 cluster prototypes

对每个 ordinary cluster：

```python
member_mask = local_assignments == center_index
member_indices = non_anchor_indices[member_mask]
```

先算每个成员和 semantic center 的相似度：

```python
semantic_scores = q_tokens[member_indices] @ centers[center_index]
```

再加 importance：

```python
scores = semantic_scores + float(lambda_importance) * importance[member_indices]
```

然后 softmax：

```python
weights = torch.softmax(scores / float(temperature), dim=0).to(dtype=patch_tokens.dtype)
```

最终 prototype 是原始视觉 token 的加权和：

```python
prototype = (patch_tokens[member_indices] * weights.unsqueeze(-1)).sum(dim=0)
```

注意这里用的是：

```text
patch_tokens: [n, d]
```

而不是：

```text
q_tokens: [n, k]
```

这保证输出仍然是后续模型期望的 hidden dimension `d`。

### 12.4 mass 和 assignments

每个 ordinary prototype 的 mass 是 cluster 成员数：

```python
masses.append(torch.tensor(member_indices.numel()))
assignments[member_indices] = output_index
```

这样可以追踪每个 prototype 代表多少原始 patch。

### 12.5 soft position

如果有 positions，普通 cluster 的 position 是 weighted average：

```python
soft_positions.append(
    (prepared_positions[member_indices] * pos_weights.unsqueeze(-1)).sum(dim=0)
)
```

这里的 `pos_weights` 和聚合视觉 token 的 `weights` 是同一套权重。

---

## 13. Position sorting 和 assignment remap

如果配置里：

```python
sort_by_position=True
```

且有 `soft_positions`，代码会对 prototypes 排序：

```python
order = _position_order(soft_positions)
```

排序 key：

```python
width = positions[:, 1].max().clamp(min=0.0) + 1.0
return torch.argsort(positions[:, 0] * width + positions[:, 1], stable=True)
```

这相当于按：

```text
row-major raster order
```

排序。

排序之后，prototype index 变了，所以 assignments 也必须 remap：

```python
assignments = _remap_assignments(assignments, order)
```

`_remap_assignments` 做的是构造 inverse permutation：

```python
inverse = torch.empty_like(order)
inverse[order] = torch.arange(order.numel(), device=order.device)
return inverse[assignments]
```

如果不 remap，那么 `assignments` 会指向排序前的 prototype index，调试信息就错了。

---

## 14. dtype 和 device 处理

实现里有几个重要细节。

### 14.1 classifier weight 转 float32

```python
return weight.to(device=device, dtype=torch.float32)
```

这样可以避免 fp16/bf16 下 response 标准化和 clustering 数值不稳定。

### 14.2 token norm 输出转 float32

```python
norm_patch = token_norm(patch_tokens).to(dtype=torch.float32)
```

semantic response 和 clustering 在 float32 中进行。

### 14.3 最终 prototype 转回原 dtype

函数返回前：

```python
patch_tokens=prototypes.to(dtype=patch_tokens.dtype)
```

这样后续模型收到的 dtype 和原 image features 一致。

### 14.4 所有新 tensor 放在原 device

例如：

```python
torch.arange(num_patches, device=patch_tokens.device)
torch.ones(num_patches, dtype=torch.bool, device=patch_tokens.device)
```

这避免 CPU/GPU 混用导致 runtime error。

---

## 15. `vit_wrapper.py`: ViT 里怎样插入 Early-SemReduce

`forward_timm_vit_with_semreduce` 的流程是：

```python
x = _patch_embed(model, images)
x = _add_position_tokens(model, x)

for block in blocks[:reduction_layer]:
    x = block(x)

classifier = classifier if classifier is not None else _default_classifier(model)
reduction = reducer(x, classifier=classifier)
x = reduction.sequence

for block in blocks[reduction_layer:]:
    x = block(x)

logits, features = _forward_head(model, x)
```

具体解释：

1. `_patch_embed` 把图像变成 patch sequence。
2. `_add_position_tokens` 加 CLS token、position embedding、dropout、pre-norm 等。
3. 前 `reduction_layer` 个 blocks 仍然处理完整 token。
4. 在这一层后调用 `EarlySemReduce`。
5. 后续 blocks 只处理 `[CLS, prototypes]`。
6. 最后走原模型 head。

如果用户没有显式传 classifier，代码默认使用：

```python
model.head.weight
```

因此它最适合标准分类 ViT。

---

## 16. `run_llava13b_pope.py`: LLaVA 实验 runner 总流程

脚本入口：

```python
if __name__ == "__main__":
    main()
```

`main()` 做下面几件事：

```text
1. parse_args()
2. load_pope_examples(...)
3. 初始化 LlavaRunner
4. 对每个 POPE example 循环
5. 对每个 method 循环: vanilla / early_semreduce
6. 保存 method.jsonl
7. 计算 summary.json
```

支持的主要参数：

```text
--model-id
--dataset-name
--split
--category
--methods
--limit
--prototype-tokens
--candidate-classes
--semantic-anchors
--cluster-iters
--temperature
--lambda-importance
--lambda-diversity
--gamma
--load-in-4bit
```

### 16.1 方法检查

代码只允许：

```text
vanilla
early_semreduce
```

如果传入其它 method，会直接报错：

```python
unknown = [method for method in methods if method not in {"vanilla", "early_semreduce"}]
if unknown:
    raise SystemExit(...)
```

### 16.2 数据集加载

`load_pope_examples` 使用 Hugging Face datasets：

```python
dataset = load_dataset(dataset_name, split=split)
```

然后按 category 过滤：

```python
dataset = dataset.filter(lambda row: str(row.get("category", "")).lower() == category.lower())
```

再 shuffle 和 limit：

```python
dataset = dataset.shuffle(seed=seed)
dataset = dataset.select(range(min(int(limit), len(dataset))))
```

每条样本封装成：

```python
Example(
    question_id=...,
    question=...,
    label=normalize_label(row["answer"]),
    image=row["image"],
    image_source=...,
    category=...,
)
```

### 16.3 输出文件

每个 method 单独写：

```text
vanilla.jsonl
early_semreduce.jsonl
```

每条 record 包含：

```text
question_id
image_source
category
question
label
method
prediction
raw_text
confidence
correct
elapsed_sec
meta
```

最后写：

```text
summary.json
```

如果同时跑了 vanilla 和 Early-SemReduce，会额外写：

```text
delta_early_semreduce_minus_vanilla
```

---

## 17. `LlavaRunner`: 模型加载和配置

初始化时：

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
```

加载 processor：

```python
self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

如果开启 4bit：

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
```

否则按 `--dtype` 解析 dtype：

```python
resolve_dtype(dtype)
```

然后加载模型：

```python
self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
self.model.eval()
```

Early-SemReduce 配置在 runner 初始化时固定：

```python
self.semreduce_config = SemReduceConfig(
    num_prototypes=prototype_tokens,
    candidate_classes=candidate_classes,
    num_anchors=semantic_anchors,
    iterations=cluster_iters,
    temperature=temperature,
    lambda_importance=lambda_importance,
    lambda_diversity=lambda_diversity,
    gamma=gamma,
)
```

---

## 18. LLaVA 中的 semantic classifier 从哪里来

标准 LLaVA 没有 ImageNet classifier head，所以代码实现了 surrogate semantic
classifier。

函数：

```python
def _semantic_classifier_for(self, device, hidden_dim):
    ...
```

它会优先检查：

```python
output_embeddings = self.model.get_output_embeddings()
input_embeddings = self.model.get_input_embeddings()
```

候选 weight：

```python
output_embeddings.weight
input_embeddings.weight
```

然后找最后一维等于 image feature hidden dim 的 weight：

```python
if int(weight.shape[-1]) == int(hidden_dim):
    cached = weight.detach().to(device=device, dtype=torch.float32)
    self._classifier_cache[cache_key] = cached
    return cached
```

也就是说，LLaVA runner 用 frozen LM head / embedding table 作为语义响应头。

这不是标准 ImageNet 分类头，而是 VLM 场景下的 surrogate head。它的作用是让
image feature 在语言模型 vocabulary embedding 空间里产生语义响应。

为了避免每次都复制 weight，代码用 cache：

```python
self._classifier_cache: dict[tuple[str, int], torch.Tensor] = {}
```

cache key 是：

```text
(device string, hidden_dim)
```

---

## 19. LLaVA hook: `_semreduce_context`

Early-SemReduce 通过 context manager 临时 patch LLaVA 的
`get_image_features`。

入口：

```python
context = self._semreduce_context() if use_semreduce else nullcontext()
with context:
    ...
```

### 19.1 为什么要 patch 两层

Hugging Face LLaVA 可能在外层 model 或内层 model 上实现 `get_image_features`。
所以代码构造：

```python
candidates = [self.model]
inner_model = getattr(self.model, "model", None)
if inner_model is not None:
    candidates.append(inner_model)
```

然后对每个有 `get_image_features` 的 candidate patch：

```python
original = candidate.get_image_features
candidate.get_image_features = make_wrapped_get_image_features(original)
patched.append((candidate, original))
```

### 19.2 wrapper 做什么

wrapper 先调用原始 image feature 函数：

```python
outputs = original(*args, **kwargs)
```

再调用：

```python
return self._reduce_image_features(outputs)
```

所以模型其余 forward 逻辑不变，只是 image features 被压缩了。

### 19.3 context 退出时恢复原函数

代码：

```python
finally:
    for candidate, original in patched:
        candidate.get_image_features = original
```

这保证 patch 只在 Early-SemReduce 方法 forward/generate 期间生效，不会污染后续
vanilla 方法。

---

## 20. `_reduce_image_features`: 对各种返回类型递归压缩

LLaVA 的 `get_image_features` 返回值可能是 Tensor，也可能是带属性的对象、list
或 tuple。代码都做了兼容。

### 20.1 Tensor

如果是 tensor：

```python
if torch.is_tensor(features):
    if features.ndim in {2, 3}:
        classifier = self._semantic_classifier_for(features.device, int(features.shape[-1]))
        return early_semreduce(
            patch_tokens=features,
            classifier=classifier,
            config=self.semreduce_config,
        ).patch_tokens
    return features
```

也就是说：

- `[n, d]` 会压缩。
- `[B, n, d]` 会压缩。
- 其它维度不处理。

这里没有传 `cls_token`，所以 `early_semreduce` 内部会用 mean patch token 代替 CLS
来选 candidate classes。

### 20.2 有 `pooler_output` 的对象

如果返回对象有 `pooler_output`：

```python
features.pooler_output = self._reduce_image_features(features.pooler_output)
return features
```

这兼容部分 Transformers 版本。

### 20.3 list 和 tuple

递归处理：

```python
if isinstance(features, list):
    return [self._reduce_image_features(item) for item in features]
if isinstance(features, tuple):
    return tuple(self._reduce_image_features(item) for item in features)
```

---

## 21. 为什么必须同步 image placeholder 数量

Hugging Face LLaVA 会要求：

```text
文本里的 image token 数量 == image features 数量
```

如果 Early-SemReduce 把 image features 从原始 `n` 压到 `m`，prompt 中 `<image>`
placeholder 也必须变成 `m` 个。

否则会出现类似错误：

```text
Image features and image tokens do not match
```

代码通过：

```python
force_image_placeholder_count(...)
```

解决。

### 21.1 具体实现

函数先找到每行 input ids 中 image token 的位置：

```python
image_positions = torch.where(ids == image_token_id)[0]
```

取第一段 image token 的边界：

```python
first = int(image_positions[0].item())
last = int(image_positions[-1].item())
```

保留 image tokens 前后的文本：

```python
before = ids[:first]
after = ids[last + 1:]
```

创建新的 image token 序列：

```python
image_ids = torch.full(
    (int(target_count),),
    int(image_token_id),
    dtype=ids.dtype,
    device=ids.device,
)
```

拼回：

```python
rows.append(torch.cat([before, image_ids, after], dim=0))
```

attention mask 同步拼接：

```python
masks.append(torch.cat([before_mask, image_mask, after_mask], dim=0))
```

最后 batch 内 padding 到同一长度：

```python
max_len = max(int(row.numel()) for row in rows)
```

pad token 使用：

```python
pad_token_id
```

mask 的 padding 部分为 0。

---

## 22. `ask_yes_no`: 一条样本怎样推理

调用：

```python
answer = runner.ask_yes_no(image, question, use_semreduce=True)
```

执行流程：

```text
1. build_prompt
2. processor(images=image, text=prompt, return_tensors="pt")
3. move_inputs_to_model
4. 如果 use_semreduce，先同步 image placeholders
5. 如果 use_semreduce，进入 _semreduce_context
6. 计算 yes/no confidence
7. model.generate
8. decode 新生成文本
9. normalize_yes_no
10. 返回 prediction/raw_text/confidence/meta
```

### 22.1 prompt

prompt 要求模型只回答 yes/no：

```python
instruction = f"{question}\n\nAnswer exactly one word: yes or no. Do not add explanation."
```

如果 processor 支持 chat template，则使用：

```python
processor.apply_chat_template(...)
```

否则 fallback：

```text
USER: <image>
{instruction}
ASSISTANT:
```

### 22.2 confidence

`_yes_no_confidence` 会先做一次普通 forward：

```python
logits = self.model(**inputs).logits[:, -1, :]
probs = torch.softmax(logits.float(), dim=-1)
```

然后把 yes/no 相关 token 的概率加起来：

```python
yes = probs[:, self.yes_token_ids].sum(dim=-1)
no = probs[:, self.no_token_ids].sum(dim=-1)
```

最后归一化到 yes/no 二分类空间：

```python
yes / (yes + no)
no / (yes + no)
```

### 22.3 generation

生成时使用：

```python
generated = self.model.generate(
    **inputs,
    max_new_tokens=self.max_new_tokens,
    do_sample=False,
)
```

也就是 greedy decoding，不采样。

### 22.4 prediction normalization

生成文本经过：

```python
normalize_yes_no(text)
```

规则：

- 以 yes 开头，或者文本里有独立 yes，预测 yes。
- 以 no 开头，或者文本里有独立 no，预测 no。
- 否则 fallback 为 no。

---

## 23. 指标计算

`compute_metrics` 统计 yes/no 二分类结果：

```python
tp = label yes and pred yes
tn = label no and pred no
fp = label no and pred yes
fn = label yes and pred no
```

然后计算：

```text
accuracy  = (tp + tn) / total
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)
yes_ratio = predicted yes count / total
```

当分母为 0 时，对应指标返回 0，避免除零。

---

## 24. demo 是怎样验证 shape 的

`scripts/demo_semreduce.py` 构造随机输入：

```python
sequence = torch.randn(batch_size, patches + 1, dim)
classifier = torch.randn(classes, dim)
```

然后创建 reducer：

```python
reducer = EarlySemReduce(
    SemReduceConfig(
        num_prototypes=args.prototype_tokens,
        candidate_classes=args.candidate_classes,
        num_anchors=args.anchors,
        iterations=args.iterations,
    )
)
```

运行后打印：

```text
input_sequence_shape
reduced_sequence_shape
selected_classes_shape
assignments_shape
mass_per_sample
```

这个 demo 主要验证 API 能跑通和 shape 是否正确，不代表真实模型精度。

---

## 25. tests 覆盖了什么

`tests/test_early_semreduce.py` 有三个核心测试。

### 25.1 `test_reduce_sequence_shape_and_assignments`

构造：

```python
sequence = torch.randn(2, 65, 32)
classifier = torch.randn(20, 32)
num_prototypes = 16
candidate_classes = 12
num_anchors = 4
```

检查：

```text
result.sequence shape == [2, 17, 32]
result.patch_tokens shape == [2, 16, 32]
assignments shape == [2, 64]
selected_classes shape == [2, 12]
CLS token 没变
masses.sum == 64
assignments 范围在 [0, 15]
```

### 25.2 `test_patch_only_function_supports_positions`

构造 7 x 7 positions：

```python
y, x = torch.meshgrid(torch.arange(7), torch.arange(7), indexing="ij")
positions = torch.stack([y.flatten(), x.flatten()], dim=-1).float()
```

调用 patch-only `early_semreduce`。

检查：

```text
result.sequence is None
result.patch_tokens shape == [9, 24]
soft_positions is not None
soft_positions 已经按 raster order 排序
masses.sum == 49
```

### 25.3 `test_no_reduction_returns_original_tokens`

输入有 4 个 patch，`num_prototypes=4`。

检查：

```text
result.sequence == sequence
assignments == [0, 1, 2, 3]
```

这个测试保证 no-reduction 分支不会改动 tokens。

---

## 26. 训练无关性在代码里怎样体现

整个实现没有任何 learnable parameter。

具体体现：

1. `SemReduceConfig` 只是超参数，不是 `nn.Parameter`。
2. `EarlySemReduce` 里没有新增线性层、卷积层或可训练 embedding。
3. classifier weight 来自 frozen model，并且用 `.detach()` 或直接作为 weight 使用。
4. LLaVA runner 在推理时使用：

```python
with torch.inference_mode():
    ...
```

5. 所有聚类和聚合都是即时 tensor operation。

因此它是 inference-time dynamic token merge，而不是训练一个新的 reducer。

---

## 27. 当前实现和理论 Early-SemReduce 的对应关系

| 理论步骤 | 代码位置 |
| --- | --- |
| 运行前若干视觉层得到中间 tokens | `vit_wrapper.py` 的 `blocks[:reduction_layer]` |
| 用 CLS 选择候选类别 | `_semantic_response` |
| 计算 patch semantic response | `_semantic_response` |
| response 标准化和 L2 normalize | `_semantic_response` |
| 计算 semantic importance | `_semantic_response` |
| 选择 semantic anchors | `_early_semreduce_single` + `_topk_indices` |
| 初始化 semantic centers | `_initialize_centers` |
| semantic assignment | `_cluster_semantic_tokens` |
| importance-aware center update | `_cluster_semantic_tokens` |
| empty cluster repair | `_reinitialize_empty_center` + `_repair_empty_assignments` |
| 原视觉空间 prototype 聚合 | `_aggregate_prototypes` |
| soft position sorting | `_prepare_positions` + `_position_order` |
| 后续 Transformer | `vit_wrapper.py` 的 `blocks[reduction_layer:]` |
| LLaVA image feature 压缩 | `run_llava13b_pope.py` 的 `_semreduce_context` |

---

## 28. 当前实现的限制

### 28.1 LLaVA runner 不是 vision tower 中间层插入

`run_llava13b_pope.py` 里的 LLaVA 接入点在 image features 产出之后。它能减少
LLM 看到的 image token 数，但不减少 vision tower 内部前向计算。

真正 block-level early reduction 的实现路径是 `vit_wrapper.py`。

### 28.2 LLaVA 使用 surrogate semantic head

标准分类模型可以直接用 `model.head.weight`。LLaVA 没有视觉分类头，所以 runner
使用 LM output embedding 或 input embedding 作为 frozen semantic classifier。

这能提供语言语义空间中的 response，但它和真正视觉分类头不是同一个东西。

### 28.3 batch 内每张图单独聚类

这是设计选择。优点是每张图像都有自己的 candidate classes、anchors 和 clusters。
缺点是 batch 维度上没有完全向量化，极大 batch 时会有 Python loop overhead。

### 28.4 当前 CLS 位置固定为 0

`EarlySemReduce.forward` 当前只支持：

```text
[CLS, patch_1, ..., patch_n]
```

如果模型有 distillation token 或多个特殊 token，需要扩展代码。

---

## 29. 调试时最应该看的字段

如果结果不符合预期，建议先打印 `SemReduceResult` 的这些字段：

```python
result.patch_tokens.shape
result.assignments.shape
result.masses
result.masses.sum()
result.selected_classes
result.anchors
result.soft_positions
```

重点检查：

```text
1. patch_tokens.shape[-2] 是否等于 num_prototypes
2. masses.sum 是否等于原始 patch 数
3. assignments.min/max 是否在合法 prototype index 范围
4. selected_classes 数量是否等于 candidate_classes
5. anchors 数量是否等于 num_anchors
6. soft_positions 是否按空间顺序排序
```

LLaVA 里最常见的问题是 image placeholder 数量不一致。需要检查：

```text
input_ids 中 image_token_id 的数量 == prototype_tokens
_reduce_image_features 返回的 image feature 数量 == prototype_tokens
```

---

## 30. 一次完整 LLaVA Early-SemReduce 调用链

从命令行开始：

```bash
python run_llava13b_pope.py \
  --model-id llava-hf/llava-1.5-13b-hf \
  --methods vanilla,early_semreduce \
  --limit 100 \
  --category adversarial \
  --prototype-tokens 128 \
  --candidate-classes 64 \
  --semantic-anchors 8 \
  --cluster-iters 5 \
  --load-in-4bit
```

对应代码调用链：

```text
main
  parse_args
  load_pope_examples
  LlavaRunner.__init__
    load processor
    load model
    build SemReduceConfig
    find image token id
    prepare yes/no token ids
  for each example
    LlavaRunner.ask_yes_no
      build_prompt
      processor(...)
      move_inputs_to_model
      force_image_placeholder_count        # only early_semreduce
      _semreduce_context                   # only early_semreduce
        patch get_image_features
        model forward / generate
          original get_image_features
          _reduce_image_features
            _semantic_classifier_for
            early_semreduce
              _early_semreduce_single
                _semantic_response
                _initialize_centers
                _cluster_semantic_tokens
                _aggregate_prototypes
                optional position sorting
        restore get_image_features
      decode generated text
      normalize_yes_no
      return record
  compute_metrics
  write summary.json
```

这就是当前代码里从数据集样本到最终 yes/no prediction 的完整路径。

---

## 31. 最短实现理解

如果只用一句话概括代码实现：

> `reducer.py` 先把 patch tokens 投影到 frozen semantic classifier 的 response
> space，在这个低维语义响应空间里选择 anchors、初始化 centers、做 hard
> clustering，然后回到原始 visual hidden space 里用 importance-aware softmax
> 加权求和生成 prototype tokens；`run_llava13b_pope.py` 则通过临时 patch
> LLaVA 的 `get_image_features`，把生成前的 image features 替换成这些
> prototype tokens，并同步 `<image>` placeholder 数量来保证 Hugging Face
> LLaVA forward 能正常运行。
