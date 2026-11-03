# flash_prefill_bnsd_score

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

## Description

During prefill, compute, for each query, a score for every logical KV block so a later top-k step can select blocks. Scores are produced per index head.

For request `r`, its packed query rows are `cu_seqlens[r]:cu_seqlens[r + 1]`. Let `P = block_size_k`, `L_r = seq_lens[r]`, and `B_r = ceil(L_r / P)`. Logical block `b` contains key positions `p = b * P + j`, where `0 <= j < P`. Index head `h` uses KV head `floor(h / gqa_group_size)`, where `gqa_group_size = num_idx_heads / num_kv_heads`. The implementation compares the packed row index `t` in `q` with this request-local key position. It masks a key when `p >= L_r` or `t < p`, then reduces the remaining logits within each block:

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

The scores use log2 units. A fully masked block has score `-inf` (including the `lse` case). The output shape is `[num_idx_heads, total_q, ceil(max(seq_lens) / block_size_k)]`; columns beyond a request's block count are `-inf`. If there are no query blocks, the returned tensor's contents are undefined; its shape, dtype, and device are still defined.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | `float16`, `bfloat16`, or `float32` | Required | Index queries. The Triton backend reads by stride and requires an NPU tensor. `num_idx_heads` must be divisible by the number of KV heads (asserted). |
| `k_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | Same as `q` | Required | Paged index-key cache in BNSD layout. `page_size == block_size_k`; page ids in the mapping must be valid (caller convention). |
| `cu_seqlens` | `torch.Tensor` | `[bs + 1]` | `int32` or `int64` | Required | Prefix sums delimiting each request's packed query rows; starts at zero and is non-decreasing (caller convention). |
| `seq_lens` | `torch.Tensor` | `[bs]` | `int32` or `int64` | Required | Total KV length per request; determines each request's block count and the output width. `bs > 0` is required by the implementation. |
| `req_to_token` | `torch.Tensor` | `[num_reqs, max_context]` | `int32` or `int64` | Required | Maps request token positions to token slots; used only when `qblock_mappings` is `None`. Its width must cover the logical block-start columns read by the operator (caller convention). |
| `req_pool_indices` | `torch.Tensor` | `[bs]` | `int32` or `int64` | Required | Row indices into `req_to_token`; used only when `qblock_mappings` is `None`. |
| `block_size_q` | `int` | — | — | Required | Positive query grouping size. Non-powers of two are accepted and rounded up internally for Triton tiling. |
| `block_size_k` | `int` | — | — | Required | Tokens per logical KV block; positive and equal to `page_size`. |
| `sm_scale` | `float` | — | — | Required | Logit scale, commonly `head_dim ** -0.5`. |
| `score_type` | `str` | — | — | `"max"` | Block reduction: `"max"` or `"lse"`; other values raise `AssertionError`. |
| `num_score_chunks` | `int` or `None` | — | — | `None` | Triton KV scheduling hint; selected automatically when `None` and clamped to the available blocks otherwise. Ignored by `torch_native`; does not change scores. |
| `qblock_mappings` | `tuple` or `None` | — | — | `None` | Optional precomputed six-tuple: `qb_to_qstart`, `qb_to_qblock`, `qb_seq_lens`, and `qb_qend` (`int32`, each length `all_seqblock_q`), `block_table` (`int32`, `[all_seqblock_q, max_blocks]`, physical page ids), and Python `int` `all_seqblock_q`. Built internally when omitted. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `score` | `torch.Tensor` | `[num_idx_heads, total_q, max_seqblock_k]` | `float32` | New tensor on `q`'s device. Inputs are not modified. Unwritten positions are `-inf`, except when there are no query blocks, in which case contents are undefined. |

The only explicit assertions are the GQA divisibility condition, `score_type` membership, and (Triton only) `MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP` being `1`, `2`, or `4` (default `1`). The implementations do not validate all shape, dtype, device, or page-id relationships; callers must provide compatible tensors on the same device and valid mappings.

## triton backend

The Triton path supports strided `q` and uses the environment variable `MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP` to control how many consecutive KV blocks are processed together. This changes tiling, not scores. It calls `seq_lens.max().item()`, which synchronizes the device with the host and makes this path unsuitable for graph capture.

## torch_native backend

This independent PyTorch implementation computes the same score-only operation without Triton. It walks one request and one KV block at a time, so it avoids materializing the full query-by-context logits matrix. `num_score_chunks` is ignored; `block_size_q` only affects construction of mappings when they are not supplied. It is intended as a correctness reference, not a performance substitute.

## Example

Requires an installed package containing this operator and the NPU runtime.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score

page_size = 128
num_pages = 3  # ceil(257 / 128)
q = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="npu")
k_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")

# Logical block b maps to physical page b.
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
