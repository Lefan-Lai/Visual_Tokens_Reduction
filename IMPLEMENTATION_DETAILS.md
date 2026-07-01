# FORESIGHT 实现细节、超参数和实验设置

本文档说明当前仓库代码如何实现 `FORESIGHT_ALGORITHM.md` 中的算法，以及 LLaVA-1.5-7B 在 POPE 上运行时使用的超参数。

## 1. 代码文件对应关系

核心文件如下：

- `foresight.py`
  - 实现 projected visual tokens 的压缩。
  - 入口函数是 `foresight_reduce(...)`。
  - 该函数只接收已经过 vision tower 和 multimodal projector 的 image features。
  - 它不修改 vision tower，不训练任何参数，也不使用固定 semantic bank。

- `run_llava7b_pope.py`
  - 加载 `llava-hf/llava-1.5-7b-hf`。
  - 加载 `lmms-lab/POPE`。
  - 在每个样本生成前，先调用模型的 `get_image_features(...)` 得到 projected image tokens。
  - 对 image tokens 调用 `foresight_reduce(...)`。
  - 将 prompt 中 image placeholder 的数量改成 `K_eff`。
  - monkey patch 当前模型的 `get_image_features(...)`，让 LLM forward 使用压缩后的 visual tokens。

- `launch_llava7b_pope.sh`
  - 服务器运行脚本。
  - 默认全量运行 POPE `category=all`。
  - 默认模型为 LLaVA-1.5-7B。

## 2. 算法开始的位置

FORESIGHT 的处理位置是：

```text
after frozen vision tower and multimodal projector
```

也就是说，代码不是在原始图像 patch 上做 merge，也不是在 LLM 内部某一层做 merge，而是在 LLaVA 的 projected image tokens 上做压缩。

具体数据流是：

```text
image
-> vision tower
-> multimodal projector
-> projected image tokens X
-> FORESIGHT
-> reduced visual tokens R_sorted
-> LLM
```

因此，`foresight_reduce(...)` 中的 `patch_tokens` 实际含义是：

```text
X in R^{n x d}
```

对 LLaVA-1.5-7B，常见情况下：

```text
n = 576
d = 4096
```

## 3. 当前实现是否完成了算法

是。当前实现覆盖了本文算法中的关键步骤：

1. 对 projected image tokens 做 LayerNorm 和 L2 normalize。
2. 计算 `x_bar` 和 `z_img`。
3. 对每个 image token 生成 residual hypothesis direction `d_img_i`。
4. 对问题文本 token embeddings 生成 `d_text_t`。
5. 将 `D_img` 和 `D_text` 合并为 candidate pool `D`。
6. 计算 global support `g_j` 和 local support `l_j`。
7. 用 `softmax(g)` 和 `softmax(l)` 的乘积得到 `omega_j`。
8. 用 `omega_j * diversity` 逐个选择 `K_b` 个 candidate hypotheses。
9. 计算 candidate response matrix `Y_c` 并按列标准化。
10. 计算 token-level support `s_j`。
11. 对 selected candidates 的 `omega_j` 重新归一化为 image-level confidence `p_j`。
12. 计算 multiplicative evidence `e_j`。
13. 通过累计 evidence 和 `rho` 动态决定 `K_eff`。
14. 选择 top `K_eff` active hypotheses。
15. 重新构建 active response space，得到 `q_i`。
16. 计算 token importance `v_tilde_i`。
17. 以每个 active hypothesis 的最大 response token 初始化 cluster seed。
18. 在 response space 中做 clustering。
19. 修复 empty cluster。
20. 用 `exp(v_tilde_i)` 加权更新 cluster centers。
21. 用 `dot(q_i, mu_j) + v_tilde_i` 聚合原始 projected image tokens。
22. 如果能推断空间位置，则按 soft spatial position 做 raster ordering。

## 4. 超参数总表

| 超参数 | 代码默认值 | LLaVA-7B POPE 使用值 | 作用 |
| --- | ---: | ---: | --- |
| `k_min` | 32 | 32 | `K_eff` 的最小值。 |
| `k_max` | 128 | 128 | candidate budget 上限，也是 `K_eff` 的最大值。 |
| `k_text` | 64 | 64 | 最多使用多少个有效文本 token embeddings 作为 text hypotheses。 |
| `rho` | 0.90 | 0.90 | 累计 evidence 阈值，用来决定 `K_rho`。 |
| `eps` | `1e-6` | `1e-6` | 所有归一化和除法的数值稳定项。 |
| `temperature` | 1.0 | 1.0 | prototype aggregation softmax 的温度。 |
| `lambda_importance` | 1.0 | 1.0 | aggregation logit 中 token importance 的权重。 |
| `sort_by_position` | True | True | 是否按 soft spatial position 对输出 prototype tokens 排序。 |
| `max_new_tokens` | 8 | 8 | POPE yes/no 生成的最大新 token 数。 |
| `load_in_4bit` | False | True in launch script | 是否用 4-bit 加载 LLaVA-7B，降低显存占用。 |

## 5. `K_b` 和 `K_eff` 如何确定

Candidate hypothesis budget 是：

```text
K_b = min(k_max, n, M)
```

其中：

- `k_max = 128`
- `n` 是 image token 数量，LLaVA-1.5-7B POPE 中通常是 576
- `M = n + m_text_used`

如果一个 POPE 问题有 9 个有效文本 tokens，则：

```text
M = 576 + 9 = 585
K_b = min(128, 576, 585) = 128
```

动态输出 token 数是：

```text
K_rho = smallest k such that cumulative top-k evidence >= rho
K_eff = max(min(k_min, K_b), min(K_rho, K_b))
```

也就是说，当前实现不是固定输出 128 个 tokens，而是根据 evidence 自动选择。已有 LLaVA-7B POPE full run 中，平均 visual tokens 约为 109。

## 6. 文本 hypothesis 的具体实现

文本不经过额外语言模型 forward。代码直接从 LLaVA text embedding table 中取问题文本 token embeddings：

```text
token_ids = tokenizer.encode(question, add_special_tokens=False)
h_t = embedding_table[token_id]
```

然后过滤掉特殊 token：

- pad token
- eos token
- bos token
- unk token
- image token

最多保留 `k_text = 64` 个有效文本 tokens。

然后在 `foresight_reduce(...)` 内部做：

```text
h_tilde_t = LayerNorm(h_t)
d_text_t = L2Normalize(h_tilde_t)
```

这些 `d_text_t` 会直接进入 candidate pool：

```text
D = concat(D_img, D_text)
```

这点非常重要：当前版本的文本 hypothesis 是 candidate 本身，不只是 reweight image candidates。

## 7. 图像 hypothesis 的具体实现

每个 projected image token 都会生成一个 image-conditioned hypothesis。

代码中：

```text
x_tilde = LayerNorm(patch_tokens)
z_tokens = L2Normalize(x_tilde)
x_bar = mean(x_tilde)
```

然后：

```text
projection_scale_i = dot(x_tilde_i, x_bar) / (dot(x_bar, x_bar) + eps)
residual_i = x_tilde_i - projection_scale_i * x_bar
d_img_i = L2Normalize(residual_i)
```

如果 residual norm 过小，就 fallback 到 `z_i`。

## 8. Diversity selection 的实现

第一个 candidate 是：

```text
argmax omega_j
```

之后每一步，对于未选择 candidate：

```text
diversity_j = min_{q in selected} (1 - dot(d_j, d_q))
score_j = omega_j * diversity_j
```

选择 `score_j` 最大的 candidate。

这一步输出 `K_b` 个 candidate hypotheses。

## 9. Clustering 的实现

Clustering 发生在 active response space：

```text
q_i in R^{K_eff}
```

不是在原始 `x_i in R^d` 上做。

初始化时，第 `j` 个 active hypothesis 选择 response 最大且未被占用的 token：

```text
seed_j = argmax_i Yhat[i, j]
center_j = q_seed_j
```

迭代次数：

```text
T_max = 1 + ceil(log2(n / K_eff))
```

对于 LLaVA-1.5-7B 常见样本，如果 `n = 576`、`K_eff = 110`：

```text
T_max = 1 + ceil(log2(576 / 110))
      = 1 + ceil(log2(5.236))
      = 4
```

这和已有实验日志中的 `iterations_used = 4` 对应。

## 10. Prototype aggregation 的实现

聚类完成后，每个 cluster 内计算：

```text
logit_ij = dot(q_i, center_j) + v_tilde_i
alpha_ij = softmax(logit_ij over tokens in cluster j)
r_j = sum_i alpha_ij * x_i
```

注意最终聚合的是原始 projected image tokens `x_i`，不是 `q_i`。这样输出仍然是 LLaVA LLM 可以接收的 hidden dimension。

## 11. Spatial ordering 的实现

如果调用者没有传入 positions，但 image token 数量是平方数，则自动构造二维网格。

例如：

```text
n = 576
sqrt(n) = 24
```

则 token 被视作 `24 x 24` 网格。每个 prototype 的 soft position 是 cluster 内 token position 的加权平均。最后按 raster order 排序：

```text
sort_key = y * width + x
```

这样压缩后的 visual token 顺序仍尽量接近图像空间顺序。

## 12. FLOPs 记录说明

代码记录以下估算字段：

- `estimated_llm_prefill_flops`
- `estimated_llm_decode_flops`
- `estimated_reduction_flops`
- `estimated_total_flops`

其中：

```text
estimated_total_flops =
    estimated_llm_prefill_flops
  + estimated_llm_decode_flops
  + estimated_reduction_flops
```

这些不是完整端到端 LLaVA FLOPs。它们没有完整包含 vision tower 图像编码、数据加载、图像预处理、GPU kernel 实际开销和显存访问开销。它们主要用于比较 visual token reduction 后 LLM 侧计算量的变化。

## 13. LLaVA-7B POPE 默认运行配置

默认启动脚本使用：

```text
model_id = llava-hf/llava-1.5-7b-hf
dataset_name = lmms-lab/POPE
split = test
category = all
k_min = 32
k_max = 128
k_text = 64
rho = 0.90
max_new_tokens = 8
load_in_4bit = true
```

输出目录中会保存：

- `foresight.jsonl`: 每条样本的预测和完整 metadata。
- `per_sample_efficiency.csv`: 每条样本的时间、token 数、FLOPs 和显存统计。
- `summary.json`: overall accuracy、precision、recall、F1、平均 visual tokens、平均吞吐等。

