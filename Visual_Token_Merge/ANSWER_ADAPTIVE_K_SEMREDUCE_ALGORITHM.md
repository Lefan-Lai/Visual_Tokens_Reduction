# Answer-Adaptive K-SemReduce 当前代码实现详解

本文档说明 `Visual_Token_Merge` 目录里当前已经实现的算法。重点是当前实验使用的 **Answer-Adaptive K-SemReduce**，也就是“答案自适应语义假设引导的视觉 token 压缩”。

本文档只使用普通 Markdown 和代码式公式，例如 `K_Q = len(H_Q) * multiplier`，避免 GitHub 或其他网页渲染器因为 LaTeX 宏不兼容而显示失败。

## 1. 当前实现是否完成了算法

当前代码已经完成了这条主路径：

1. 对每个 MME yes/no 问题，从问题文本和 MME 类别中提取语义假设集合 `H_Q`。
2. 不再设置最小 hypotheses 数量，也不再设置最大 hypotheses 数量。
3. 不再用固定 `extra_visual_topk` 去补齐 image-activated vocabulary。
4. 用 `hypothesis_multiplier` 决定最终 K：
   - `multiplier = 1` 时，`K_Q = len(H_Q) * 1`。
   - `multiplier = 2` 时，`K_Q = len(H_Q) * 2`。
   - `multiplier = 3` 时，`K_Q = len(H_Q) * 3`。
5. 把每个 expanded hypothesis 转成 LLaVA 语言 embedding 空间里的一个 frozen semantic vector。
6. 用这些 semantic vectors 作为 K-SemReduce 的动态 classifier head。
7. 在 LLaVA image features 层进行 token reduction。
8. 根据 reduction 后的 token 数量，重写 prompt 里的 `<image>` placeholder 数量。
9. 用相同的 LLaVA-1.5-13B 模型同时跑 vanilla 和 reduced 方法。
10. 输出 MME overall、dimension-level、category-level 结果。

当前代码没有训练任何参数，也没有 fine-tune 模型。所有压缩都是 inference-time、training-free 的。

## 2. 相关代码文件

当前算法主要由这些文件组成：

| 文件 | 作用 |
|---|---|
| `early_semreduce/k_reducer.py` | 固定 K 的 K-SemReduce 核心实现。 |
| `early_semreduce/answer_adaptive_reducer.py` | Answer-Adaptive K-SemReduce 包装器，把动态 hypothesis matrix 传给 K-SemReduce。 |
| `run_llava13b_mme.py` | LLaVA-1.5-13B + MME 实验入口，包含 hypothesis 提取、multiplier 扩展、LLaVA image feature hook、指标输出。 |
| `run_llava13b_pope.py` | POPE runner，支持 `k_semreduce`。 |
| `early_semreduce/__init__.py` | 导出 K-SemReduce 和 Answer-Adaptive K-SemReduce API。 |
| `tests/test_k_semreduce.py` | K-SemReduce 单元测试。 |
| `tests/test_answer_adaptive_k_semreduce.py` | Answer-Adaptive K-SemReduce 和 hypothesis multiplier 单元测试。 |
| `README.md` | 简要使用说明和命令示例。 |

## 3. 算法总览

整体 pipeline 可以写成：

```text
image + question
  -> LLaVA processor
  -> vision tower and projector 得到 image features X
  -> 从 question/category 提取 H_Q
  -> 用 multiplier 扩展 H_Q 得到 expanded hypotheses
  -> 把 expanded hypotheses 转成 W_H
  -> K-SemReduce(X, W_H)
  -> reduced image features
  -> 根据 reduced token 数重写 <image> placeholders
  -> LLaVA language model 生成 yes/no
  -> MME metrics
```

其中最重要的变量是：

| 名称 | 含义 |
|---|---|
| `X` | 原始 image patch tokens，shape 是 `[N, D]` 或 `[B, N, D]`。 |
| `N` | 原始 patch token 数量。 |
| `D` | LLaVA hidden dimension。 |
| `H_Q` | 从当前问题和类别里提取出的基础 semantic hypotheses。 |
| `multiplier` | `--adaptive-hypothesis-multiplier`，当前实验取 `1, 2, 3`。 |
| `K_Q` | 当前问题请求的输出 token 数，`K_Q = len(H_Q) * multiplier`。 |
| `W_H` | expanded hypotheses 的 embedding matrix，shape 是 `[K_Q, D]`。 |
| `q_i` | 第 `i` 个 patch 的 normalized semantic response vector。 |
| `u_i` | 第 `i` 个 patch 的 semantic importance score。 |
| `mu_j` | 第 `j` 个 semantic cluster center。 |

## 4. 与旧版本的区别

旧版本里曾经有这些参数：

```text
--adaptive-min-hypotheses
--adaptive-max-hypotheses
--adaptive-extra-visual-topk
```

当前版本已经不再使用这些参数。也就是说：

1. 不再人为规定 `H_Q` 至少有多少个 hypotheses。
2. 不再人为规定 `H_Q` 最多有多少个 hypotheses。
3. 不再额外加入固定数量的 visual vocabulary terms。
4. 当前 K 只由问题提取出的 hypotheses 数量和 multiplier 决定。

当前代码里的有效规则是：

```text
base_hypotheses = extract_question_hypotheses(question, category)
expanded_hypotheses = expand_hypotheses(base_hypotheses, multiplier)
K_Q = len(expanded_hypotheses)
```

对于 `multiplier = 1, 2, 3`，在没有重复项被去重之后：

```text
K_Q = len(H_Q) * 1
K_Q = len(H_Q) * 2
K_Q = len(H_Q) * 3
```

底层 K-SemReduce 仍然会做一个必要的安全截断：

```text
actual_K = min(K_Q, number_of_patch_tokens, number_of_classifier_rows)
```

在 Answer-Adaptive 路径里，`number_of_classifier_rows = K_Q`，所以主要安全截断来自 `number_of_patch_tokens`。这不是最大 hypotheses 设计，而是为了避免要求输出 token 数超过输入 patch token 数。

## 5. Step 1: 从问题中提取基础 hypotheses

实现位置：

```text
run_llava13b_mme.py
function: extract_question_hypotheses(question, category)
```

输入是一个 MME 问题和它的 category，例如：

```text
question = "Is there a red car in the image? Please answer yes or no."
category = "color"
```

输出是一个去重后的 list，例如：

```text
["color", "red", "red object", "car"]
```

### 5.1 文本预处理

代码先把问题变成小写：

```text
text = question.lower()
```

然后去掉 MME yes/no instruction 里的固定句子：

```text
"please answer yes or no"
```

这样做的目的，是避免把 `please`、`answer`、`yes`、`no` 这些非视觉词当成 hypotheses。

### 5.2 加入 category hints

代码内有一个 `CATEGORY_HINTS` 字典。不同 MME category 会加入不同的先验视觉概念：

| MME category | 例子 |
|---|---|
| `color` | `color` |
| `count` | `count`, `number`, `object` |
| `OCR` | `text`, `word`, `letter`, `number`, `sign` |
| `position` | `position`, `relation`, `location` |
| `code_reasoning` | `code`, `python`, `program`, `output`, `number` |
| `scene` | `scene`, `place`, `background` |

例如 category 是 `OCR`，即使问题文本很短，算法也会显式加入 `text`、`word`、`letter`、`number`、`sign` 这些与 OCR 任务相关的 hypotheses。

### 5.3 检测已知视觉短语

代码会检查 `KNOWN_VISUAL_PHRASES`，例如：

```text
baseball bat
cell phone
dining table
fire hydrant
traffic light
wine glass
```

如果这些 phrase 出现在问题里，就直接加入 hypotheses。

这样做是因为简单按空格切词会把 `traffic light` 拆成 `traffic` 和 `light`，但它们合在一起才是更完整的视觉概念。

### 5.4 检测关系词

代码会检查 `RELATION_PHRASES`：

```text
above
behind
below
beside
between
holding
inside
near
next to
on top of
under
wearing
```

如果问题中有这些关系词，就加入类似：

```text
"near relation"
"holding relation"
"wearing relation"
```

这让 position、relation、commonsense 类问题在 semantic response 中有专门的关系维度。

### 5.5 检测颜色

代码会检查常见颜色：

```text
black, blue, brown, gray, green, orange, pink, purple, red, white, yellow
```

如果问题中出现颜色，例如 `red`，会加入两个 hypotheses：

```text
"red"
"red object"
```

这样既保留颜色本身，也保留“带有这种颜色的物体”这个视觉概念。

### 5.6 提取引号中的答案概念

如果问题里出现单引号或双引号包住的词，代码会提取它们：

```text
'cat'
"python"
"stop sign"
```

如果提取到的值不是 `yes` 或 `no`，就加入 hypotheses。

这对一些 MME 问题很重要，因为问题可能问：

```text
Does the sign contain the word "STOP"?
```

这时 `"STOP"` 就是视觉文本证据。

### 5.7 普通 token 提取和 stopword 过滤

最后，代码用正则提取所有英文和数字 token：

```text
[a-z0-9]+
```

然后过滤掉 stopwords，例如：

```text
a, an, and, are, is, the, yes, no, answer, image, picture
```

长度小于 3 的非数字 token 也会过滤掉。

剩下的 token 会经过一个非常轻量的 singularize：

```text
cars -> car
babies -> baby
```

### 5.8 fallback

如果最后没有提取到任何 concept，代码会使用 fallback：

```text
["object", "scene", "visual evidence"]
```

这保证 `H_Q` 不会为空。

### 5.9 去重规则

去重函数是：

```text
dedupe_preserve_order(values)
```

它会：

1. strip 前后空格。
2. lower-case。
3. 把多个空格压成一个空格。
4. 保留第一次出现的位置。
5. 删除后续重复项。

所以 `["Cat", " cat ", "red object"]` 会变成：

```text
["cat", "red object"]
```

## 6. Step 2: 用 multiplier 扩展 hypotheses

实现位置：

```text
run_llava13b_mme.py
function: expand_hypotheses(base, multiplier)
```

当前实验核心就在这里。

输入：

```text
base = H_Q
multiplier = 1, 2, or 3
```

输出：

```text
expanded_hypotheses
```

代码逻辑是：

```text
base_hypotheses = dedupe_preserve_order(base)
multiplier = max(1, int(multiplier))

templates = [
    "{label}",
    "visual evidence of {label}",
    "detailed {label} region",
]

expanded = []
for template in templates[:multiplier]:
    for label in base_hypotheses:
        expanded.append(template.format(label=label))
```

因此，如果：

```text
H_Q = ["cat", "red object"]
```

那么 `multiplier = 1`：

```text
expanded_hypotheses = [
    "cat",
    "red object",
]
K_Q = 2
```

`multiplier = 2`：

```text
expanded_hypotheses = [
    "cat",
    "red object",
    "visual evidence of cat",
    "visual evidence of red object",
]
K_Q = 4
```

`multiplier = 3`：

```text
expanded_hypotheses = [
    "cat",
    "red object",
    "visual evidence of cat",
    "visual evidence of red object",
    "detailed cat region",
    "detailed red object region",
]
K_Q = 6
```

如果之后想测试 `multiplier > 3`，代码也支持，会自动加入：

```text
"supporting visual cue 4 for {label}"
"supporting visual cue 5 for {label}"
...
```

但当前实验计划是 `1, 2, 3`。

## 7. Step 3: 把 hypotheses 转成 semantic head

实现位置：

```text
run_llava13b_mme.py
method: LlavaMMERunner._answer_adaptive_semantic_head(...)
method: LlavaMMERunner._phrase_embedding(...)
```

当前代码对每个 expanded hypothesis 做 phrase embedding：

```text
embedding(label) = mean(input_embedding(tokenize(" " + label)))
```

具体步骤：

1. 从 LLaVA 模型里取 input embedding weight。
2. 先尝试 tokenizing `" " + phrase`，因为很多语言模型词表对带前导空格的 token 更稳定。
3. 如果没有有效 token id，就再尝试 tokenizing `phrase`。
4. 如果还是没有有效 token id，就返回一个 fallback vector。
5. 如果有多个 token，就取这些 token embedding 的平均值。
6. 最后对平均向量做 L2 normalize。

得到的矩阵是：

```text
W_H = stack([embedding(h) for h in expanded_hypotheses])
```

shape 是：

```text
[K_Q, D]
```

其中：

```text
K_Q = len(expanded_hypotheses)
D = LLaVA hidden dimension
```

这个 `W_H` 就是 Answer-Adaptive K-SemReduce 的动态 classifier head。

## 8. Step 4: Answer-Adaptive 包装器

实现位置：

```text
early_semreduce/answer_adaptive_reducer.py
```

这个文件里的核心思想很简单：

```text
Answer-Adaptive K-SemReduce = K-SemReduce + per-question W_H
```

`AnswerAdaptiveKSemReduceConfig` 里没有固定 K，因为 K 由 `hypothesis_embeddings.shape[0]` 决定。

配置项是：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `iterations` | `3` | 聚类迭代次数。 |
| `temperature` | `0.1` | 聚合时 softmax 的温度。 |
| `lambda_importance` | `0.25` | 聚合时 semantic importance 的权重。 |
| `gamma` | `1.0` | 更新 cluster center 时 importance 权重的指数系数。 |
| `eps` | `1e-6` | 数值稳定项。 |
| `sort_by_position` | `True` | 是否按 soft spatial position 排序输出 prototype tokens。 |

包装器最终构造：

```text
KSemReduceConfig(
    num_semantic_classes = hypothesis_embeddings.shape[0],
    iterations = adaptive_iterations,
    temperature = adaptive_temperature,
    lambda_importance = adaptive_lambda_importance,
    gamma = adaptive_gamma,
    eps = adaptive_eps,
    sort_by_position = adaptive_sort_by_position,
)
```

然后调用：

```text
k_semreduce(
    patch_tokens = X,
    classifier = W_H,
    config = k_config,
)
```

因此，Answer-Adaptive 不是另写一套聚类算法，而是用动态问题语义头驱动同一个 K-SemReduce 核心。

## 9. Step 5: K-SemReduce 核心算法

实现位置：

```text
early_semreduce/k_reducer.py
```

K-SemReduce 的输入是：

```text
patch_tokens: X, shape [N, D]
classifier: W, shape [C, D]
num_semantic_classes: K_request
```

输出是：

```text
prototype_tokens: shape [actual_K, D]
assignments: 每个原始 patch 属于哪个 prototype
masses: 每个 prototype 聚合了多少原始 patch
selected_classes: 使用了哪些 semantic rows
soft_positions: 每个 prototype 的 soft spatial position
```

### 9.1 K 的安全确定

代码里先计算：

```text
actual_K = min(K_request, N, C)
```

对 Answer-Adaptive 来说：

```text
K_request = K_Q
C = K_Q
```

所以一般是：

```text
actual_K = min(K_Q, N)
```

如果 `K_Q <= N`，输出 token 数就是 `K_Q`。

如果 `K_Q > N`，输出 token 数会被截到 `N`，因为不可能把 `N` 个输入 patch 压缩成比 `N` 更多的 prototype tokens。

### 9.2 semantic response

实现依赖：

```text
early_semreduce/reducer.py
function: _semantic_response(...)
```

如果没有显式传入 CLS token，代码用所有 patch 的平均值作为 pseudo CLS：

```text
cls = mean(X)
```

然后归一化 CLS：

```text
norm_cls = layer_norm(cls)
```

计算 CLS 对 classifier rows 的 logit：

```text
logits = norm_cls @ W.T
```

如果是固定 K-SemReduce，会从 classifier 里选 top K rows。

如果是 Answer-Adaptive，`W = W_H` 本身就只有 `K_Q` rows，所以实际等价于使用所有 hypotheses rows。

然后对 patch tokens 做同样的 token norm：

```text
norm_patch = layer_norm(X)
```

计算 patch-level semantic response：

```text
responses = norm_patch @ W_selected.T
```

shape 是：

```text
[N, actual_K]
```

其中 `responses[i, j]` 表示第 `i` 个 patch 对第 `j` 个 semantic hypothesis/class 的响应。

### 9.3 response 标准化

对每个 semantic dimension 单独做标准化：

```text
p_hat[:, j] = (responses[:, j] - mean(responses[:, j])) / (std(responses[:, j]) + eps)
```

这样可以减少不同 hypothesis row 的尺度差异。

### 9.4 patch semantic vector

对每个 patch 的 standardized response vector 做 L2 normalize：

```text
q_i = normalize(p_hat_i)
```

因此每个 patch 现在不再只看原始视觉 hidden state，而是看它在 semantic hypothesis space 里的位置。

### 9.5 semantic importance

每个 patch 的 importance 是：

```text
raw_importance_i = max(p_hat_i) - mean(p_hat_i)
```

直觉是：

1. 如果某个 patch 对某个 hypothesis 特别强，那么 `max(p_hat_i)` 会很大。
2. 如果它对所有 hypothesis 都差不多，那么 `max - mean` 不会特别突出。
3. 因此这个值可以衡量 patch 是否有“尖锐的语义证据”。

然后 importance 再做一次 z-score：

```text
u_i = (raw_importance_i - mean(raw_importance)) / (std(raw_importance) + eps)
```

这个 `u_i` 后面会用于 cluster center 更新和 prototype 聚合。

### 9.6 class-guided seed selection

K-SemReduce 不随机初始化 cluster centers。

对每个 semantic class/hypothesis 维度 `j`，代码选择：

```text
seed_j = argmax_i p_hat[i, j]
```

但它有一个约束：不同 class 不能选同一个 patch 做 seed。

实现方式：

1. 初始化一个 `unavailable` mask。
2. 第 1 个 class 选响应最高的 patch。
3. 选中过的 patch 标为 unavailable。
4. 第 2 个 class 只能从剩下的 patch 里选。
5. 一直到选出 `actual_K` 个不同 seed。

这样每个 semantic hypothesis 至少有一个不同的初始 patch anchor。

### 9.7 cluster assignment

初始 centers 是 seed 对应的 semantic vectors：

```text
mu_j = q_seed_j
```

每次 assignment 使用 cosine-style dot product：

```text
assignment_i = argmax_j dot(q_i, mu_j)
```

因为 `q_i` 和 `mu_j` 都是 normalized vector，dot product 就是 semantic similarity。

### 9.8 empty cluster repair

聚类过程中可能出现某些 cluster 没有 patch。代码不会允许空 cluster 进入聚合。

修复逻辑：

1. 找出所有 empty clusters。
2. 找出所有 patch 数量大于 1 的 donor clusters。
3. 选择最分散的 donor cluster。
4. 从 donor cluster 里拿出最不适合该 donor center 的 patch。
5. 把这个 patch 移到 empty cluster。

选择 donor cluster 时，代码计算每个 donor 的平均 dispersion：

```text
dispersion = mean(1 - dot(q_i, mu_donor))
```

dispersion 最大的 cluster 说明内部最松散，拿走一个 patch 的代价相对低。

选择 moved patch 时，代码找 donor 内与 donor center 最不相似的 patch：

```text
moved_patch = argmin_i dot(q_i, mu_donor)
```

这样做可以保证所有 cluster 都非空。

### 9.9 cluster center update

每个 cluster center 更新时，不是简单平均，而是 importance-weighted average：

```text
weight_i = exp(gamma * u_i)
mu_j = normalize(sum_i weight_i * q_i)
```

其中 `i` 只遍历属于 cluster `j` 的 patch。

`gamma` 控制 importance 对 center 的影响：

| `gamma` | 效果 |
|---:|---|
| `0` | 所有 patch 权重相同，退化成普通平均。 |
| `1.0` | 当前默认，语义更强的 patch 对 center 影响更大。 |
| 更大 | 更偏向 high-importance patch。 |

### 9.10 迭代次数

当前默认：

```text
iterations = 3
```

每次迭代包括：

```text
assign -> repair_empty_clusters -> update_centers
```

循环结束后，代码还会再做一次 final：

```text
assign -> repair_empty_clusters -> update_centers
```

这样可以保证最终 assignments 与 centers 是一致的，并且没有空 cluster。

### 9.11 prototype token 聚合

聚类结束后，每个 cluster 会被聚合成一个 prototype visual token。

对 cluster `j` 内的每个 patch `i`，先算：

```text
semantic_score_i = dot(q_i, mu_j)
score_i = semantic_score_i + lambda_importance * u_i
```

然后做 softmax：

```text
alpha_i = softmax(score_i / temperature)
```

最后在原始视觉 hidden space 里聚合：

```text
prototype_j = sum_i alpha_i * x_i
```

注意这里最终聚合用的是原始 patch token `x_i`，不是 semantic vector `q_i`。也就是说，semantic response 只负责决定怎么分组和怎么加权，最终输出 token 仍然保留 LLaVA image feature space 的表示。

### 9.12 soft position 和排序

如果没有显式传入 patch positions，代码会根据 patch 数量自动构造位置：

1. 如果 `N` 是平方数，例如 576，就构造 2D grid。
2. 如果不是平方数，就构造一行 positions。

每个 prototype 的 soft position 是 cluster 内 patch positions 的加权平均：

```text
prototype_position_j = sum_i alpha_i * position_i
```

如果 `sort_by_position = True`，最终 prototype tokens 会按 soft position 从上到下、从左到右排序。

这样做是为了尽量保持 LLaVA 后续语言模型看到的视觉 token 顺序仍然接近图像空间顺序。

## 10. Step 6: LLaVA-1.5-13B 接入方式

实现位置：

```text
run_llava13b_mme.py
class LlavaMMERunner
```

### 10.1 vanilla baseline

如果 method 是：

```text
vanilla
```

代码不做 token reduction，直接让 LLaVA 正常处理图像和问题。

### 10.2 reduced method

如果 method 是：

```text
answer_adaptive_k_semreduce
```

代码会先调用：

```text
_precompute_dynamic_image_features(inputs, method, question, category)
```

这个函数会让 LLaVA 的 vision tower 和 projector 先计算 image features，然后在 image feature level 调用：

```text
_reduce_image_features(features, method, question, category)
```

当前 reduction stage 写在 metadata 里：

```text
image_feature_level_after_vision_tower_and_projector
```

也就是压缩发生在 vision tower 和 multimodal projector 之后，进入 language model 之前。

### 10.3 为什么要重写 `<image>` placeholders

LLaVA prompt 里有若干 `<image>` token placeholders。原始 image features 有多少个 token，prompt 里就需要对应多少个 image placeholders。

压缩后，image features 的 token 数变成了动态的 `actual_K`。所以代码必须调用：

```text
force_image_placeholder_count(...)
```

把 input ids 里的 `<image>` token 数量改成 `actual_K`。

否则 language model 会发现：

```text
number_of_image_placeholders != number_of_image_features
```

然后报错或行为不一致。

### 10.4 预计算 features 的上下文 hook

代码会用 context manager 临时 patch 模型的 image feature 方法，让模型 forward/generate 时直接复用已经压缩好的 image features。

这样可以避免：

1. 第一次为了压缩算了一遍 image features。
2. generate 时又重新算一遍未压缩 image features。

在 context 退出后，原始模型方法会被恢复。

## 11. MME 数据和指标

实现位置：

```text
run_llava13b_mme.py
```

### 11.1 数据加载

默认数据集：

```text
lmms-lab/MME
```

默认 split：

```text
test
```

默认采样：

```text
--sampling stratified
```

默认 image 数量：

```text
--limit-images 300
```

你当前要跑 100 张图时，命令里使用：

```text
--limit-images 100
```

### 11.2 dimension 划分

代码把 MME category 分成：

```text
perception
cognition
unknown
```

perception categories：

```text
existence, count, position, color, posters, celebrity, scene, landmark, artwork, OCR
```

cognition categories：

```text
commonsense_reasoning, numerical_calculation, text_translation, code_reasoning
```

### 11.3 yes/no 输出解析

prompt 强制要求：

```text
Answer exactly one word: yes or no. Do not add explanation.
```

生成后，代码用 `normalize_yes_no` 解析：

1. 如果回答以 `yes` 开头，预测为 `yes`。
2. 如果回答以 `no` 开头，预测为 `no`。
3. 如果无法识别，默认预测为 `no`。

同时，代码还会用下一 token logits 计算 `yes_prob` 和 `no_prob`，保存到 metadata。

### 11.4 输出文件

每次运行会写：

```text
vanilla.jsonl
answer_adaptive_k_semreduce.jsonl
summary.json
category_metrics.csv
dimension_metrics.csv
```

每条 jsonl 记录包含：

```text
image_id
question_id
category
dimension
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

Answer-Adaptive 的 `meta` 还包含：

```text
adaptive_hypothesis_multiplier
base_hypothesis_count
hypothesis_multiplier
requested_hypothesis_count
selected_hypothesis_count
hypotheses
selected_hypotheses
original_image_tokens
reduced_image_tokens
mean_region_mass
max_region_mass
min_region_mass
```

其中：

```text
base_hypothesis_count = len(H_Q)
requested_hypothesis_count = len(H_Q) * multiplier
selected_hypothesis_count = actual output token count
```

### 11.5 MME metrics

代码计算：

| 指标 | 含义 |
|---|---|
| `accuracy` | 单问题 yes/no 准确率。 |
| `paired_accuracy` | 同一 image 的两条 yes/no 问题是否都答对。MME 通常把它记作 `acc+`。 |
| `mme_score` | `100 * accuracy + 100 * paired_accuracy`。 |
| `precision` | yes 类 precision。 |
| `recall` | yes 类 recall。 |
| `f1` | yes 类 F1。 |
| `yes_ratio` | 预测 yes 的比例。 |

这些指标会分别输出到：

1. overall。
2. dimension-level。
3. category-level。

## 12. 当前超参数说明

### 12.1 MME runner 参数

| 参数 | 默认值 | 当前实验建议 | 说明 |
|---|---:|---:|---|
| `--model-id` | `llava-hf/llava-1.5-13b-hf` | 同默认 | 使用 LLaVA-1.5-13B。 |
| `--dataset-name` | `lmms-lab/MME` | 同默认 | MME 数据集。 |
| `--split` | `test` | 同默认 | MME test split。 |
| `--methods` | `vanilla,answer_adaptive_k_semreduce` | 同默认 | 同时跑 baseline 和压缩方法。 |
| `--limit-images` | `300` | `100` 或 `300` | 限制图片数量。 |
| `--sampling` | `stratified` | `stratified` | 分类别均衡采样。 |
| `--seed` | `0` | `0` | 采样随机种子。 |
| `--device-map` | `auto` | `auto` | transformers 自动放置模型。 |
| `--dtype` | `auto` | `auto` | 自动选择 dtype。 |
| `--max-new-tokens` | `8` | `8` | yes/no 生成长度。 |
| `--load-in-4bit` | off | on | 13B 模型通常需要 4-bit。 |

### 12.2 Answer-Adaptive 参数

| 参数 | 默认值 | 当前实验值 | 作用 |
|---|---:|---:|---|
| `--adaptive-cluster-iters` | `3` | `3` | K-SemReduce 聚类迭代次数。 |
| `--adaptive-temperature` | `0.1` | `0.1` | 聚合 softmax 温度。 |
| `--adaptive-lambda-importance` | `0.25` | `0.25` | 聚合时 importance 权重。 |
| `--adaptive-gamma` | `1.0` | `1.0` | 更新 centers 时 importance 指数权重。 |
| `--adaptive-hypothesis-multiplier` | `1` | `1, 2, 3` | 当前实验的核心自变量。 |

### 12.3 已删除的旧参数

当前命令不应该再使用：

```text
--adaptive-min-hypotheses
--adaptive-max-hypotheses
--adaptive-extra-visual-topk
```

如果命令里还有这些参数，说明命令是旧版本的。

## 13. 当前推荐实验命令

在远端服务器上：

```bash
ssh llai933@lais10.cer.auckland.ac.nz
cd ~/Visual_Token_Merge
```

跑 100 张 MME 图片，分别测试 `multiplier = 1, 2, 3`：

```bash
for M in 1 2 3; do
  PYTHONPATH=. ./venv/bin/python run_llava13b_mme.py \
    --model-id llava-hf/llava-1.5-13b-hf \
    --methods vanilla,answer_adaptive_k_semreduce \
    --limit-images 100 \
    --sampling stratified \
    --adaptive-cluster-iters 3 \
    --adaptive-temperature 0.1 \
    --adaptive-lambda-importance 0.25 \
    --adaptive-gamma 1.0 \
    --adaptive-hypothesis-multiplier ${M} \
    --load-in-4bit \
    --output-dir ~/results/llava13b_answer_adaptive_k_semreduce_mme100_hypx${M}
done
```

跑 300 张 MME 图片时，只需要把 `--limit-images 100` 改成 `--limit-images 300`，并把 output dir 改成 `mme300`。

## 14. 为什么这个算法是 answer-adaptive

普通 K-SemReduce 使用固定 classifier rows 或固定 top K class prototypes。对于所有问题，它的 K 和 semantic axes 通常相同。

Answer-Adaptive K-SemReduce 的区别是：

1. 每个问题都有自己的 `H_Q`。
2. 每个问题都有自己的 `W_H`。
3. 每个问题都有自己的 `K_Q`。
4. 压缩时保留的 prototype tokens 会围绕当前问题的语义证据组织。

例如：

```text
Question A: Is there a red car?
H_Q may include: color, red, red object, car
```

而：

```text
Question B: Is the word "STOP" visible on the sign?
H_Q may include: text, word, sign, stop
```

这两个问题使用的 semantic axes 不同，所以即使输入图片相同，压缩结果也可能不同。

## 15. 为什么 multiplier 有意义

`multiplier` 控制每个基础 hypothesis 被展开成多少个语义视角。

`multiplier = 1`：

```text
"cat"
```

只保留最直接的 object/concept 语义。

`multiplier = 2`：

```text
"cat"
"visual evidence of cat"
```

额外强调视觉证据，不只是词本身。

`multiplier = 3`：

```text
"cat"
"visual evidence of cat"
"detailed cat region"
```

进一步强调局部区域和细节。

这三个设置的实验问题是：

```text
更多 hypothesis variants 是否能保留更多与回答相关的视觉证据？
```

同时它也会带来更高的 token 数：

```text
K_Q increases linearly with multiplier
```

所以实验需要比较：

1. MME accuracy 是否提升。
2. paired accuracy 是否提升。
3. perception 和 cognition 两个 dimension 是否表现不同。
4. token reduction ratio 是否下降。
5. 推理时间是否增加。

## 16. 当前实现的边界和注意事项

1. 当前 hypothesis extraction 是规则式的，不调用额外 language model。
2. 当前 phrase embedding 使用 LLaVA input embedding 平均，不训练新的 text encoder。
3. 当前 reduction 发生在 LLaVA image feature level，不是在 vision tower 内部早期层。
4. 当前代码保留了部分 semantic vocabulary helper，但当前 Answer-Adaptive 主路径不再调用 fixed `extra_visual_topk`。
5. 当前没有使用 min/max hypotheses。
6. 当前没有把 yes/no 当成 visual hypotheses。
7. 当前 K-SemReduce 是 training-free，不会改变 LLaVA 权重。
8. 当前 LLaVA prompt 强制 one-word yes/no，所以适合 MME yes/no 格式。

## 17. 单元测试覆盖点

当前测试覆盖了：

1. `k_semreduce` 输出 exact K。
2. 当 K 大于 patch 数时，K 会 clamp 到 patch 数。
3. K-SemReduce seed selection 不会重复选同一个 patch。
4. Answer-Adaptive 的输出 token 数等于 hypothesis embedding rows。
5. Answer-Adaptive 也会在 hypothesis rows 多于 patch 数时安全 clamp。
6. `expand_hypotheses(["cat", "cat", "red object"], 3)` 会先去重，再得到 6 个 expanded hypotheses。

测试命令：

```bash
cd Visual_Token_Merge
python -m pytest tests
```

## 18. 最短伪代码

```text
function AnswerAdaptiveKSemReduce(image, question, category, multiplier):
    X = LLaVA_vision_tower_and_projector(image)

    H_Q = extract_question_hypotheses(question, category)
    H_expanded = expand_hypotheses(H_Q, multiplier)
    W_H = phrase_embeddings(H_expanded)

    K_Q = len(H_expanded)
    actual_K = min(K_Q, number_of_patches(X))

    R = semantic_response(X, W_H)
    P = standardize_each_semantic_dimension(R)
    Q = normalize_each_patch_response(P)
    U = semantic_importance(P)

    seeds = class_guided_unique_seeds(P, actual_K)
    centers = Q[seeds]

    repeat iterations times:
        assignments = assign_by_dot_similarity(Q, centers)
        assignments = repair_empty_clusters(Q, centers, assignments)
        centers = importance_weighted_center_update(Q, U, assignments)

    assignments = assign_by_dot_similarity(Q, centers)
    assignments = repair_empty_clusters(Q, centers, assignments)
    centers = importance_weighted_center_update(Q, U, assignments)

    prototypes = []
    for each cluster:
        scores = dot(Q_members, center) + lambda_importance * U_members
        weights = softmax(scores / temperature)
        prototype = weighted_sum(original_X_members, weights)
        prototypes.append(prototype)

    prototypes = sort_by_soft_position(prototypes)
    return prototypes
```

## 19. 结论

当前代码实现的核心是：

```text
K is no longer fixed.
K is no longer bounded by manually chosen min/max hypotheses.
K is determined by the question:
K_Q = len(extract_question_hypotheses(question, category)) * hypothesis_multiplier
```

然后用这个问题自适应的 `K_Q` 和对应的 semantic hypothesis matrix `W_H`，驱动 K-SemReduce 对 LLaVA image features 做 training-free token compression。

