# flash_prefill_bnsd_topk_from_score

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 待验证 |
| `torch_native` | 待验证 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

两个已实现后端仍需分别在各代芯片上验证。

## 函数原型

`auto` 默认选择 `triton`；省略 `impl` 与此相同。运行时不会自动回退到其他后端。

```python
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_topk_from_score

def flash_prefill_bnsd_topk_from_score(
    score,
    per_query_seq_lens,
    topk,
    block_size_k,
    *,
    impl="auto",
):
    ...
```

## 功能说明

在 prefill 阶段，从每个 `(head, query)` 的块分数中选出 `topk` 个 KV 逻辑块，并把该 query 自己
所在的因果块追加到最后一列，供下游 sparse attention 使用。

令 `K = score.shape[-1]`。对第 `t` 个 query：

- `q_pos_t = max(per_query_seq_lens[t] - 1, 0)`，即该 query 可见的最后一个 KV 位置；
- `local_t = min(floor(q_pos_t / block_size_k), K - 1)`，即该 query 所在的逻辑块。

候选块按 `(score[h, t, b] 降序, b 升序)` 排列，记为 `b_0, b_1, ...`。前 `topk` 个输出槽（下标
`r = 0..topk-1`）写第 `r` 个候选；该候选同时满足以下两个条件时写块号，否则写 `-1`：

- `score[h, t, b_r] > -inf`（输入约定为有效块使用有限分数、不可用块使用 `-inf`）；
- `b_r * block_size_k <= q_pos_t`（不晚于该 query 的因果位置）。

最后一列（下标 `topk`）：若 `local_t` 已出现在前 `topk` 槽中则写 `-1`，否则写 `local_t`。

$$
b_0, b_1, \ldots = \operatorname{argsort}_{b}\big(-\text{score}[h,t,b],\; b\big)
$$

$$
\text{topk\_idx}[h,t,r] =
\begin{cases}
b_r, & r < \text{topk},\ \text{score}[h,t,b_r] > -\infty,\ b_r \cdot \text{block\_size\_k} \le q\_pos_t \\
-1, & \text{otherwise}
\end{cases}
$$

$$
\text{topk\_idx}[h,t,\text{topk}] =
\begin{cases}
-1, & local_t \in \{b_0, \ldots, b_{\text{topk}-1}\} \\
local_t, & \text{otherwise}
\end{cases}
$$

前 `topk` 槽不压缩：有分数但尚未到因果位置的候选会被写成 `-1`，因此 `-1` 可以出现在任意
位置，而不是只出现在尾部。末槽总会被写入。输出形状为
`[num_kv_heads, total_q, topk + 1]`，dtype 为 `int32`；`topk > max_seqblock_k` 时多出的槽写 `-1`。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `score` | `torch.Tensor` | `[num_kv_heads, total_q, max_seqblock_k]` | `float32` | 必填 | 块分数，通常来自 `flash_prefill_bnsd_score`；不可用块为 `-inf`。Triton 按 stride 读取，支持非连续输入。 |
| `per_query_seq_lens` | `torch.Tensor` | `[total_q]` | `int32` | 必填 | 每个 query 可见的 KV 数（inclusive 因果长度），由 SGLang 的 `prefix + within + 1` 生成；取值应不超过该 request 的 `seq_lens`（调用方约定）。 |
| `topk` | `int` | — | — | 必填 | 每个 query 选出的块数，必须大于 0（被断言）。 |
| `block_size_k` | `int` | — | — | 必填 | 每个 KV 逻辑块包含的 token 数，必须为正，并与生成 `score` 时使用的块大小一致（调用方约定）。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `topk_idx` | `torch.Tensor` | `[num_kv_heads, total_q, topk + 1]` | `int32` | 在 `score` 所在设备新建的连续张量。前 `topk` 槽可为 `-1`，末槽为 local 块或 `-1`；输入不被修改。 |

实现会断言 `score` 为 `float32`、`per_query_seq_lens` 为 `int32` 且长度等于 `total_q`，以及
`topk > 0`。调用方还需保证 `block_size_k > 0`、score 块维度非空、输入位于同一设备、每个
query 长度不超过所属 request 的 KV 长度，并以有限分数或 `-inf` 表示可用块和不可用块；算子
不会检查所有这些约定。

## triton 后端

实现根据 score 宽度使用面向 UB 的 query 分块启发式（每个 program 最多处理 16 个 query）。
`topk` 是编译期参数，不同取值可能分别编译 kernel。launch 固定 `num_warps=4`、
`num_stages=1`。实现按 stride 读取 score，因此不要求连续；wrapper 不调用 `.item()`，不会与
host 同步。

## torch_native 后端

独立的 PyTorch 参考实现，不调用 Triton。它逐轮取 `max` 并用最小列号解决并列，和 Triton 的
选择顺序一一对应；不使用 `torch.topk`/`argsort` 的并列顺序。逐轮循环和 `score.clone()` 使它的
显存与耗时都高于 Triton 版本，只用于精度对比。

## 调用示例

需要安装包含此算子的包及 NPU 运行环境。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_topk_from_score

score = torch.tensor(
    [[[-1.0, -2.0, 5.0], [-1.0, -2.0, 0.5]]],
    dtype=torch.float32,
    device="npu",
)
per_query_seq_lens = torch.tensor([3, 5], dtype=torch.int32, device="npu")

out = flash_prefill_bnsd_topk_from_score(
    score,
    per_query_seq_lens,
    topk=2,
    block_size_k=2,
)
print(out.tolist())  # [[[-1, 0, 1], [2, 0, -1]]]
```

第 0 行块 2 分数最高，但 `2 * 2 > q_pos = 2`，因此首槽写 `-1`；块 0 入选，`local = 1` 未入选，
末槽写 1。第 1 行块 2、0 入选，`local = 2` 已入选，末槽写 `-1`。
