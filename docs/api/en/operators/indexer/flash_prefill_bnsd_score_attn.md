# flash_prefill_bnsd_score_attn

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Pending verification | Pending verification |
| `torch_native` | Pending verification | Pending verification |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

Both implemented backends still need hardware verification on each generation.

## Signature

`auto` selects `triton`; omitting `impl` has the same effect. There is no runtime fallback.

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

## Description

During prefill this operator does two things at once: it computes a score for every logical KV
block of every query (for a later top-k selection), and it computes a causal index-attention
output per index head. Both outputs are organised by index head.

For request `r`, its packed query rows are `cu_seqlens[r]:cu_seqlens[r + 1]`. Let
`P = block_size_k`, `L_r = seq_lens[r]`, and `B_r = ceil(L_r / P)`. Logical block `b` contains
key positions `p = b * P + j`, where `0 <= j < P`. Index head `h` uses KV head
`floor(h / gqa_group_size)`, where `gqa_group_size = num_idx_heads / num_kv_heads`.

For packed query row `t` and key position `p`, the log2-domain logit is

$$
x_{t,p} = (q_t \cdot k_p)\,\text{sm\_scale}\,\log_2 e
$$

The block score is reduced according to `score_type`:

$$
\text{score}[h,t,b] =
\begin{cases}
\max_{p \in b} x_{t,p}, & \text{score\_type} = \text{"max"} \\
m + \log_2 \sum_{p \in b} 2^{x_{t,p}-m}, & \text{score\_type} = \text{"lse"},\quad m=\max_{p \in b} x_{t,p}
\end{cases}
$$

The index-attention output is a softmax over all legal keys and does not depend on `score_type`:

$$
\text{idx\_o}[t,h,:] =
\frac{\sum_{p} 2^{x_{t,p}-m_t}\,v_p}{\sum_{p} 2^{x_{t,p}-m_t}},
\quad m_t = \max_{p} x_{t,p}
$$

Here `p` must satisfy `p < L_r` and pass the implementation's causal test. Note that the current
implementation compares the **global packed query-row index `t`** with the request-local key
position `p`; therefore it is equivalent to request-local causality only when
`cu_seqlens[r] == 0`. Requests with `r > 0` in a packed batch see more keys. This is the
pre-existing behavior and is not changed in this version.

`score` has shape `[num_idx_heads, total_q, ceil(max(seq_lens) / block_size_k)]`; unwritten
positions are `-inf`. `idx_o` has shape `[total_q, num_idx_heads, head_dim]`; unwritten
positions are `0`. When there are no query blocks (`all_seqblock_q == 0`), the operator returns
an all-`-inf` `score` and an all-`0` `idx_o`; shape, dtype, and device are defined.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | `float16`, `bfloat16` (`float32` unverified) | Required | Index queries. The Triton backend reads by stride and requires an NPU tensor. `num_idx_heads` must be divisible by the number of KV heads (asserted). |
| `k_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | Same as `q` | Required | Paged index-key cache in BNSD layout. `page_size == block_size_k`; page ids must be valid (caller convention). |
| `v_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | Same as `k_cache_bnsd` | Required | Paged index-value cache in BNSD layout; it shares the block table and KV heads with K. |
| `cu_seqlens` | `torch.Tensor` | `[bs + 1]` | `int32` or `int64` | Required | Prefix sums delimiting each request's packed query rows; starts at zero and is non-decreasing (caller convention). |
| `seq_lens` | `torch.Tensor` | `[bs]` | `int32` or `int64` | Required | Total KV length per request; determines the block count and output width. The implementation requires `bs > 0`. |
| `req_to_token` | `torch.Tensor` | `[num_reqs, max_context]` | `int32` or `int64` | Required | Maps request token positions to token slots; used only when `qblock_mappings` is `None`. Its width must cover the logical block-start columns read by the operator (caller convention). |
| `req_pool_indices` | `torch.Tensor` | `[bs]` | `int32` or `int64` | Required | Row indices into `req_to_token`; used only when `qblock_mappings` is `None`. |
| `block_size_q` | `int` | — | — | Required | Positive query grouping size. Non-powers of two are accepted and rounded up internally for Triton tiling. |
| `block_size_k` | `int` | — | — | Required | Tokens per logical KV block; positive, equal to `page_size`, and a power of two (caller convention; production uses 128). |
| `sm_scale` | `float` | — | — | Required | Logit scale, commonly `head_dim ** -0.5`. |
| `score_type` | `str` | — | — | `"max"` | Block-score reduction: `"max"` or `"lse"`; other values raise `AssertionError`. It affects only `score`, not `idx_o`. |
| `qblock_mappings` | `tuple` or `None` | — | — | `None` | Optional precomputed six-tuple: `qb_to_qstart`, `qb_to_qblock`, `qb_seq_lens`, and `qb_qend` (`int32`, each length `all_seqblock_q`), `block_table` (`int32`, `[all_seqblock_q, max_blocks]`, physical page ids), and Python `int` `all_seqblock_q`. Built internally when omitted. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `score` | `torch.Tensor` | `[num_idx_heads, total_q, max_seqblock_k]` | `float32` | New tensor on `q`'s device. Unwritten positions are `-inf`; with no query blocks every position is `-inf`. Inputs are not modified. |
| `idx_o` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | Same as `q` | New tensor on `q`'s device. Rows not written by the kernel stay `0`; with no query blocks every position is `0`. Inputs are not modified. |

The only explicit assertions are `num_idx_heads % num_kv_heads == 0` and
`score_type ∈ {"max", "lse"}`. The implementations do not validate all shape, dtype, device,
page-id, or power-of-two `block_size_k` relationships; callers must provide compatible tensors
on the same device and valid mappings.

## triton backend

- Reads `q`/`k_cache_bnsd`/`v_cache_bnsd` by stride, so non-contiguous inputs are supported;
  `score` and `idx_o` are newly allocated contiguous tensors.
- Each program handles one query-block × one KV head and walks all KV blocks of that request;
  there is no `num_score_chunks` or BPS parameter.
- `qblock_mappings` is reusable across layers; its column count may exceed `max_seqblock_k`,
  and extra columns are ignored.
- With `all_seqblock_q == 0`, the function returns after allocation, so `score` is all `-inf`
  and `idx_o` is all `0`.
- It calls `seq_lens.max().item()`, which synchronizes the device with the host and makes this
  path unsuitable for graph capture.
- The causal test uses the global packed row index; see "Description". This is pre-existing
  behavior and is not fixed in this version.

## torch_native backend

This independent PyTorch implementation does not call Triton. It walks requests, KV blocks, and
KV heads one block at a time, computing both `score` and the online-softmax `idx_o` without
materializing the full query-by-context logits matrix. It casts `p` to `v_cache_bnsd.dtype`
before the V matmul to mirror the kernel's precision path. When `qblock_mappings` is omitted it
uses its own copy of `_build_qblock_mappings`. It is intended as a correctness reference, not a
performance substitute.

## Example

Requires an installed package containing this operator and the NPU runtime.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

page_size = 128
num_pages = 3  # ceil(257 / 128)
q = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="npu")
k_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")
v_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")

# Logical block b maps to physical page b.
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
