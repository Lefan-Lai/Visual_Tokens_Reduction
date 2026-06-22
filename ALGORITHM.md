# ProtoReduce Visual Token Reduction Algorithm

本文档说明当前 `Visual_Token_Reduction` 代码里的算法设计、实现位置、LLaVA 13B 接入方式，以及实验结果应该如何解读。

## 1. 目标

LLaVA 这类 VLM 会把图像切成许多 patch token，再把这些视觉 token 送入语言模型。视觉 token 数量越多，推理时的上下文长度、显存占用和注意力计算成本越高。

ProtoReduce 的目标是在不训练模型、不改模型权重的前提下，把原始 patch tokens 压缩成更少的 prototype tokens：

```text
原始视觉 tokens:     X = {x_i}_{i=1}^n, x_i in R^d
压缩后 prototype:   R = {r_j}_{j=1}^m, r_j in R^d
其中 m << n
```

当前实验脚本默认把 LLaVA 13B 的视觉 token 压缩到 `m = 128` 个 prototype tokens，然后比较：

- `vanilla`: 不压缩视觉 token，直接用 LLaVA 13B 推理。
- `proto_reduce`: 对视觉 token 做 ProtoReduce 后再推理。

## 2. 核心思想

ProtoReduce 不是简单平均池化，也不是随机丢弃 token。它把 patch tokens 看成一组视觉特征点，先在单位球面上聚类，再对每个 cluster 内的原始 token 做 soft weighted pooling。

这样做有两个动机：

1. 聚类可以尽量覆盖图像里不同语义/视觉区域，避免只保留局部高响应 token。
2. 最终 pooling 使用原始 token `x_i`，而不是归一化后的 `z_i`，保留原本特征尺度和模型投影空间。

## 3. 算法步骤

输入：

```text
X = {x_i}_{i=1}^n, patch tokens, x_i in R^d
m = prototype token 数量
T = clustering iterations
tau = softmax temperature
```

输出：

```text
R = {r_j}_{j=1}^m, prototype tokens
```

### 3.1 Token 归一化

每个视觉 token 先做 L2 normalization：

```text
z_i = x_i / (||x_i||_2 + eps)
```

后续聚类使用 cosine similarity：

```text
sim(i, j) = z_i^T mu_j
```

这样可以减少 token magnitude 对聚类的影响，让聚类更关注方向，也就是视觉特征相似性。

### 3.2 初始化 centers

当前实现支持两种初始化：

- `farthest`: farthest-first 初始化，默认使用。
- `kmeans++`: k-means++ 风格初始化。

默认 `farthest` 的流程是：

1. 计算所有归一化 token 的平均方向。
2. 选择与平均方向最相似的 token 作为第一个 center。
3. 后续每次选择距离已有 centers 最远的 token。

这样可以让初始 centers 尽量分散，覆盖更多视觉区域。

### 3.3 Spherical k-means 聚类

重复 `T` 轮：

1. 对每个 token 找最相似的 center：

```text
g_i = argmax_j z_i^T mu_j
```

2. 对每个 cluster 更新 center：

```text
C_j = {i | g_i = j}
mu_j = Normalize(mean_{i in C_j} z_i)
```

如果某个 cluster 为空，当前实现会选一个“最不被已有 centers 表示”的 token 重新初始化它：

```text
fallback = argmin_i max_j z_i^T mu_j
```

这可以避免空 cluster 让 prototype 数量变少。

### 3.4 Cluster 内 soft pooling

聚类完成后，每个 prototype token 不是直接取 center，而是回到原始 token 空间做加权求和：

```text
s_i = z_i^T mu_j
alpha_i = exp(s_i / tau) / sum_{k in C_j} exp(s_k / tau)
r_j = sum_{i in C_j} alpha_i x_i
```

其中：

- `tau` 越小，越接近选 cluster 中最接近 center 的 token。
- `tau` 越大，越接近 cluster 内平均池化。

当前默认：

```text
T = 5
tau = 0.07
m = 128
```

## 4. 代码实现

核心实现位于：

```text
proto_reduce.py
```

主要函数：

```python
proto_reduce(tokens, num_prototypes, iterations=5, temperature=0.07)
```

输入 shape 支持：

```text
[n, d]
[batch, n, d]
```

输出 shape：

```text
[m, d]
[batch, m, d]
```

如果 `m >= n`，函数直接返回原 tokens，不做压缩。

## 5. LLaVA 13B 接入方式

实验脚本位于：

```text
run_llava13b_pope.py
```

当前使用 Hugging Face 版本：

```text
llava-hf/llava-1.5-13b-hf
```

LLaVA forward 的关键流程大致是：

```text
image -> vision tower -> multimodal projector -> image features
text prompt 中的 <image> placeholders -> language model input embeddings
image features 替换 <image> placeholders 对应位置
```

ProtoReduce 插入的位置是：

```text
vision tower + multimodal projector 之后
language model 接收 image features 之前
```

也就是说压缩发生在已经投影到 LLaVA 语言模型 hidden size 的视觉特征空间里。

### 5.1 为什么要同步 image placeholder 数量

Hugging Face 的 LLaVA 实现会检查：

```text
text prompt 里的 image token 数量 == image features 数量
```

所以如果把视觉特征从原始数量压缩成 128 个，也必须把文本输入里的 `<image>` placeholder 数量改成 128 个。

脚本里对应函数是：

```python
force_image_placeholder_count(...)
```

它会把原始 prompt 中连续的 image token placeholders 替换成指定数量。

### 5.2 兼容当前 Transformers 版本

当前服务器上的 Transformers 版本中，LLaVA 实际调用的是内层模型：

```text
self.model.get_image_features(...)
```

因此脚本会同时 patch 外层和内层模型的 `get_image_features`，并对返回对象里的 `pooler_output` 做压缩。这个兼容逻辑在：

```python
LlavaRunner._proto_context()
```

## 6. 实验流程

安装依赖后可以运行：

```bash
python run_llava13b_pope.py \
  --model-id llava-hf/llava-1.5-13b-hf \
  --load-in-4bit \
  --methods vanilla,proto_reduce \
  --prototype-tokens 128 \
  --cluster-iters 5 \
  --temperature 0.07 \
  --limit 100 \
  --output-dir /home/llai933/results/llava13b_proto_reduce_100_4bit
```

脚本会对 POPE adversarial 样本逐条运行两个方法，并输出：

```text
vanilla.jsonl
proto_reduce.jsonl
summary.json
```

每条 JSONL 包含：

- question id
- question
- label
- prediction
- confidence
- correct
- elapsed time
- method metadata

`summary.json` 包含：

- accuracy
- precision
- recall
- F1
- yes ratio
- `delta_proto_minus_vanilla`

其中 `delta_proto_minus_vanilla` 表示：

```text
ProtoReduce 指标 - Vanilla 指标
```

如果 delta 为正，说明 ProtoReduce 在该指标上更好；如果为负，说明压缩造成了下降。

## 7. 当前实现的特点

当前 ProtoReduce 是 training-free 方法：

- 不训练新模型。
- 不更新 LLaVA 权重。
- 不需要额外标注。
- 每张图像推理时动态聚类。
- 可以通过 `--prototype-tokens` 控制压缩强度。

主要超参数：

```text
--prototype-tokens: prototype token 数量，默认 128
--cluster-iters: 聚类迭代次数，默认 5
--temperature: cluster 内 soft pooling 温度，默认 0.07
--init: center 初始化方式，默认 farthest
```

## 8. 预期影响

理论上，减少视觉 token 数量可以降低语言模型处理视觉 token 的成本，尤其是在 attention 计算中。不过当前脚本每个 method 都独立 forward，并且 ProtoReduce 本身也有聚类开销，因此端到端时间不一定立刻变快。

这个实验主要先回答两个问题：

1. 视觉 token 压缩后，POPE adversarial 上的准确率/F1 会不会下降。
2. 在相同模型和数据上，`proto_reduce` 与 `vanilla` 的 yes/no 偏置是否有明显变化。

## 9. 已验证状态

当前代码已在服务器上通过 1 条样本 smoke test：

```text
method: proto_reduce
limit: 1
prediction: yes
correct: true
```

并修复了 Hugging Face LLaVA 中常见的 placeholder mismatch 问题：

```text
Image features and image tokens do not match
```

修复方式是同时压缩实际 forward 使用的 image features，并同步 prompt 中的 image token placeholder 数量。
