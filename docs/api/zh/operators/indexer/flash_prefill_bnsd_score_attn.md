# flash_prefill_bnsd_score_attn

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
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

def flash_prefill_bnsd_score_attn(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    cu_seqlens,
    seq_lens,
    req_to_token,
    req_pool_indices,
    block_size_q,
    block_size_k,
    sm_scale,
    score_type="max",
    qblock_mappings=None,
    *,
    impl="auto",
):
    ...
```

## 功能说明

在 prefill 阶段同时完成两件事：为每个 query 计算各 KV 逻辑块的分数（供后续 top-k 选块），
并对每个索引头计算一次 causal 的索引注意力输出。两个输出都按索引头组织。

request `r` 的打包 query 行范围为 `cu_seqlens[r]:cu_seqlens[r + 1]`。令
`P = block_size_k`、`L_r = seq_lens[r]`、`B_r = ceil(L_r / P)`。逻辑块 `b` 中的位置为
`p = b * P + j`（`0 <= j < P`）。索引头 `h` 使用 KV 头
`floor(h / gqa_group_size)`，其中 `gqa_group_size = num_idx_heads / num_kv_heads`。

对 query 打包行 `t` 与 key 位置 `p`，先算 log2 域 logits：

$$
x_{t,p} = (q_t \cdot k_p)\,\text{sm\_scale}\,\log_2 e
$$

块分数按 `score_type` 归约：

$$
\text{score}[h,t,b] =
\begin{cases}
\max_{p \in b} x_{t,p}, & \text{score\_type} = \text{"max"} \\
m + \log_2 \sum_{p \in b} 2^{x_{t,p}-m}, & \text{score\_type} = \text{"lse"},\quad m=\max_{p \in b} x_{t,p}
\end{cases}
$$

索引注意力输出对所有合法 key 做 softmax（与 `score_type` 无关）：

$$
\text{idx\_o}[t,h,:] =
\frac{\sum_{p} 2^{x_{t,p}-m_t}\,v_p}{\sum_{p} 2^{x_{t,p}-m_t}},
\quad m_t = \max_{p} x_{t,p}
$$

其中 `p` 需满足 `p < L_r` 且通过实现的因果判断。注意：**当前实现的因果判断使用 `q` 的全局
打包行号 `t` 与 request 内 KV 位置 `p` 比较**，因此只有 `cu_seqlens[r] == 0` 时等价于
request 内因果；批次中 `r > 0` 的请求会看到更多 key。该行为与迁移前实现一致，未在本版本修改。

`score` 形状为 `[num_idx_heads, total_q, ceil(max(seq_lens) / block_size_k)]`，
未写入的位置为 `-inf`。`idx_o` 形状为 `[total_q, num_idx_heads, head_dim]`，
未写入的位置为 `0`。若没有 query 块（`all_seqblock_q == 0`），直接返回全 `-inf` 的 `score`
和全 `0` 的 `idx_o`，shape、dtype、device 均有定义。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | `float16`、`bfloat16`（`float32` 未验证） | 必填 | 索引 query。Triton 按 stride 读取，要求 tensor 位于 NPU。`num_idx_heads` 必须能被 KV 头数整除（被断言）。 |
| `k_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | 与 `q` 相同 | 必填 | BNSD 布局的分页索引 K cache。`page_size == block_size_k`；物理页号必须有效（调用方约定）。 |
| `v_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | 与 `k_cache_bnsd` 相同 | 必填 | BNSD 布局的分页索引 V cache；与 K 使用同一 `block_table` 和 KV 头。 |
| `cu_seqlens` | `torch.Tensor` | `[bs + 1]` | `int32` 或 `int64` | 必填 | 各 request 打包 query 行的前缀和；从 0 开始、单调不减（调用方约定）。 |
| `seq_lens` | `torch.Tensor` | `[bs]` | `int32` 或 `int64` | 必填 | 每个 request 的 KV 总长度，决定块数和输出宽度。实现要求 `bs > 0`。 |
| `req_to_token` | `torch.Tensor` | `[num_reqs, max_context]` | `int32` 或 `int64` | 必填 | request token 位置到 token slot 的映射；仅在 `qblock_mappings` 为 `None` 时使用。列宽需覆盖逻辑块起始列（调用方约定）。 |
| `req_pool_indices` | `torch.Tensor` | `[bs]` | `int32` 或 `int64` | 必填 | `req_to_token` 中的行号；仅在 `qblock_mappings` 为 `None` 时使用。 |
| `block_size_q` | `int` | — | — | 必填 | 正数 query 分组大小；允许非 2 的幂，Triton 内部向上取整用于分块。 |
| `block_size_k` | `int` | — | — | 必填 | 每个 KV 逻辑块包含的 token 数，必须为正数、等于 `page_size`，且为 2 的幂（调用方约定；生产为 128）。 |
| `sm_scale` | `float` | — | — | 必填 | logits 缩放系数，通常为 `head_dim ** -0.5`。 |
| `score_type` | `str` | — | — | `"max"` | 块分数归约方式：`"max"` 或 `"lse"`；其他取值触发 `AssertionError`。只影响 `score`，不影响 `idx_o`。 |
| `qblock_mappings` | `tuple` 或 `None` | — | — | `None` | 可选预计算六元组：`qb_to_qstart`、`qb_to_qblock`、`qb_seq_lens`、`qb_qend`（均为长度 `all_seqblock_q` 的 `int32` 数组）、`block_table`（`int32`、`[all_seqblock_q, max_blocks]`，物理页号）和 Python `int` 类型的 `all_seqblock_q`。省略时由算子构建。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `score` | `torch.Tensor` | `[num_idx_heads, total_q, max_seqblock_k]` | `float32` | 在 `q` 所在设备新建。未写入位置为 `-inf`；没有 query 块时全为 `-inf`。不修改输入。 |
| `idx_o` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | 与 `q` 相同 | 在 `q` 所在设备新建。未被 kernel 写入的行保持 `0`；没有 query 块时全为 `0`。不修改输入。 |

显式断言仅包括：`num_idx_heads % num_kv_heads == 0` 和 `score_type ∈ {"max", "lse"}`。
实现不会检查所有 shape、dtype、设备、页号及 `block_size_k` 的 2 的幂关系；
调用方需保证输入兼容、位于同一设备且映射有效。

## triton 后端

- 读取 `q`/`k_cache_bnsd`/`v_cache_bnsd` 的 strides，支持非连续输入；`score`、`idx_o` 为新分配的连续张量。
- 每个 program 处理一个 query-block × 一个 KV 头，遍历该 request 的全部 KV 逻辑块；
  没有 `num_score_chunks` 或 BPS 参数。
- `qblock_mappings` 可跨层复用；其列宽可以大于 `max_seqblock_k`，多余列被忽略。
- 空输入（`all_seqblock_q == 0`）在分配之后直接返回，所以 `score` 全 `-inf`、`idx_o` 全 `0`。
- 调用 `seq_lens.max().item()` 会产生一次设备到主机同步，不适合图捕获。
- 因果判断使用全局打包行号，具体语义见“功能说明”；这是迁移前既有行为，不在本版本修复。

## torch_native 后端

独立的 PyTorch 实现，不调用 Triton。它按 request、KV 块、KV 头逐个块计算 `score` 与
online-softmax `idx_o`，避免展开完整的 query × context logits 矩阵；`p` 会先 cast 到
`v_cache_bnsd.dtype` 再与 V 相乘，以贴近 kernel 的精度路径。未提供 `qblock_mappings` 时，
它调用同一份 `_build_qblock_mappings` 副本构建映射。该后端用于正确性参考，不作为性能替代。

## 调用示例

需要安装包含此算子的包及 NPU 运行环境。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

page_size = 128
num_pages = 3  # ceil(257 / 128)
q = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="npu")
k_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")
v_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")

# 逻辑块 b 映射到物理页 b。
req_to_token = torch.zeros(1, num_pages * page_size, dtype=torch.int32, device="npu")
req_to_token[0, ::page_size] = torch.arange(num_pages, dtype=torch.int32, device="npu") * page_size

score, idx_o = flash_prefill_bnsd_score_attn(
    q,
    k_cache,
    v_cache,
    cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device="npu"),
    seq_lens=torch.tensor([257], dtype=torch.int32, device="npu"),
    req_to_token=req_to_token,
    req_pool_indices=torch.tensor([0], dtype=torch.int32, device="npu"),
    block_size_q=4,
    block_size_k=page_size,
    sm_scale=128**-0.5,
)
print(score.shape, score.dtype)  # torch.Size([1, 4, 3]), torch.float32
print(idx_o.shape, idx_o.dtype)  # torch.Size([4, 1, 128]), torch.bfloat16
```
