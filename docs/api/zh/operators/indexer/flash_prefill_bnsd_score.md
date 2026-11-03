# flash_prefill_bnsd_score

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
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score

def flash_prefill_bnsd_score(
    q,
    k_cache_bnsd,
    cu_seqlens,
    seq_lens,
    req_to_token,
    req_pool_indices,
    block_size_q,
    block_size_k,
    sm_scale,
    score_type="max",
    num_score_chunks=None,
    qblock_mappings=None,
    *,
    impl="auto",
):
    ...
```

## 功能说明

在 prefill 阶段，为每个 query 计算各 KV 逻辑块的分数，供后续 top-k 选择使用；每个索引头分别计算一组分数。

request `r` 的打包 query 行范围为 `cu_seqlens[r]:cu_seqlens[r + 1]`。令 `P = block_size_k`、`L_r = seq_lens[r]`、`B_r = ceil(L_r / P)`。逻辑块 `b` 中的位置为 `p = b * P + j`，其中 `0 <= j < P`。索引头 `h` 使用 KV 头 `floor(h / gqa_group_size)`，其中 `gqa_group_size = num_idx_heads / num_kv_heads`。实现使用 `q` 中的打包行号 `t` 与该 request 内的 KV 位置比较：当 `p >= L_r` 或 `t < p` 时屏蔽该 key，再对块内 logits 归约：

$$
x_{t,p} = (q_t \cdot k_p)\,\text{sm\_scale}\,\log_2 e
$$

$$
\text{score}[h,t,b] =
\begin{cases}
\max_j x_{t,p}, & \text{score\_type} = \text{"max"} \\
m + \log_2 \sum_j 2^{x_{t,p}-m}, & \text{score\_type} = \text{"lse"},\quad m=\max_j x_{t,p}
\end{cases}
$$

分数以 log2 为单位。整块被屏蔽时分数为 `-inf`（包括 `lse`）。输出形状为 `[num_idx_heads, total_q, ceil(max(seq_lens) / block_size_k)]`；超出各 request 块数的列为 `-inf`。若没有 query 块，返回张量的内容未定义，但 shape、dtype 和 device 有定义。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | `float16`、`bfloat16` 或 `float32` | 必填 | 索引 query。Triton 按 stride 读取，并要求 tensor 位于 NPU。`num_idx_heads` 必须能被 KV 头数整除（被断言）。 |
| `k_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | 与 `q` 相同 | 必填 | BNSD 布局的分页索引 K cache。`page_size == block_size_k`；映射中的物理页号必须有效（调用方约定）。 |
| `cu_seqlens` | `torch.Tensor` | `[bs + 1]` | `int32` 或 `int64` | 必填 | 标记各 request 打包 query 行范围的前缀和；从 0 开始且单调不减（调用方约定）。 |
| `seq_lens` | `torch.Tensor` | `[bs]` | `int32` 或 `int64` | 必填 | 每个 request 的 KV 总长度，决定各自块数和输出宽度。实现要求 `bs > 0`。 |
| `req_to_token` | `torch.Tensor` | `[num_reqs, max_context]` | `int32` 或 `int64` | 必填 | request token 位置到 token slot 的映射；仅在 `qblock_mappings` 为 `None` 时使用。列宽需覆盖算子读取的逻辑块起始列（调用方约定）。 |
| `req_pool_indices` | `torch.Tensor` | `[bs]` | `int32` 或 `int64` | 必填 | `req_to_token` 中对应的行号；仅在 `qblock_mappings` 为 `None` 时使用。 |
| `block_size_q` | `int` | — | — | 必填 | 必须为正数的 query 分组大小；允许非 2 的幂，Triton 内部会向上取整用于分块。 |
| `block_size_k` | `int` | — | — | 必填 | 每个 KV 逻辑块包含的 token 数，必须为正数且等于 `page_size`。 |
| `sm_scale` | `float` | — | — | 必填 | logits 缩放系数，通常为 `head_dim ** -0.5`。 |
| `score_type` | `str` | — | — | `"max"` | 块内归约方式：`"max"` 或 `"lse"`；其他取值会触发 `AssertionError`。 |
| `num_score_chunks` | `int` 或 `None` | — | — | `None` | Triton KV 调度参数；为 `None` 时自动选择，否则限制在可用块数范围内。`torch_native` 忽略该参数；它不改变分数。 |
| `qblock_mappings` | `tuple` 或 `None` | — | — | `None` | 可选的六元组：`qb_to_qstart`、`qb_to_qblock`、`qb_seq_lens`、`qb_qend`（均为长度 `all_seqblock_q` 的 `int32` 数组）、`block_table`（`int32`、`[all_seqblock_q, max_blocks]`，保存物理页号）和 Python `int` 类型的 `all_seqblock_q`。省略时由算子构建。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `score` | `torch.Tensor` | `[num_idx_heads, total_q, max_seqblock_k]` | `float32` | 在 `q` 所在设备新建的张量。不修改输入。未写入的位置为 `-inf`；没有 query 块时内容未定义。 |

显式断言仅包括：GQA 头数整除关系、`score_type` 取值，以及 Triton 专用环境变量 `MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP` 必须为 `1`、`2` 或 `4`（默认 `1`）。实现不会检查所有 shape、dtype、设备及页号关系；调用方需保证输入兼容、位于同一设备且映射有效。

## triton 后端

Triton 支持非连续的 `q`，并通过环境变量 `MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP` 控制一次处理的连续 KV 块数。该参数只改变分块，不改变分数。实现会调用 `seq_lens.max().item()`，导致设备与主机同步，因此不适合用于图捕获。

## torch_native 后端

这是独立的 PyTorch 实现，不调用 Triton。它按 request 逐个 KV 块计算，避免展开完整的 query × context logits 矩阵。`num_score_chunks` 会被忽略；未提供 `qblock_mappings` 时，`block_size_q` 只影响映射构建。该后端用于正确性参考，不用于替代高性能实现。

## 调用示例

需要安装包含此算子的包及 NPU 运行环境。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score

page_size = 128
num_pages = 3  # ceil(257 / 128)
q = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="npu")
k_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")

# 逻辑块 b 映射到物理页 b。
req_to_token = torch.zeros(1, num_pages * page_size, dtype=torch.int32, device="npu")
req_to_token[0, ::page_size] = torch.arange(num_pages, dtype=torch.int32, device="npu") * page_size

score = flash_prefill_bnsd_score(
    q,
    k_cache,
    cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device="npu"),
    seq_lens=torch.tensor([257], dtype=torch.int32, device="npu"),
    req_to_token=req_to_token,
    req_pool_indices=torch.tensor([0], dtype=torch.int32, device="npu"),
    block_size_q=4,
    block_size_k=page_size,
    sm_scale=128**-0.5,
)
print(score.shape, score.dtype)  # torch.Size([1, 4, 3]), torch.float32
```
