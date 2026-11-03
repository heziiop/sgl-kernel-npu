# flash_prefill_bnsd_topk_from_score

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

## Description

During prefill, select `topk` logical KV blocks for each `(head, query)` row of block scores and
append the query's own causal block to the last column, so a downstream sparse-attention kernel can
consume the result.

Let `K = score.shape[-1]`. For query `t`:

- `q_pos_t = max(per_query_seq_lens[t] - 1, 0)` is the last KV position visible to the query;
- `local_t = min(floor(q_pos_t / block_size_k), K - 1)` is the logical block containing it.

Candidates are ordered by `(score[h, t, b] descending, b ascending)`, written `b_0, b_1, ...`. The
first `topk` output slots (`r = 0..topk-1`) take the `r`-th candidate; the block id is written only
when both conditions hold, otherwise `-1` is written:

- `score[h, t, b_r] > -inf` (the input convention is finite scores or `-inf` for unusable blocks);
- `b_r * block_size_k <= q_pos_t` (not past the query's causal position).

The last column (index `topk`) holds `-1` when `local_t` already appears in the first `topk` slots,
and `local_t` otherwise.

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

The first `topk` slots are not compacted: a scored block that is not yet causally visible is written
as `-1`, so `-1` can appear anywhere rather than only at the end. The last slot is always written.
The output shape is `[num_kv_heads, total_q, topk + 1]` with dtype `int32`; when
`topk > max_seqblock_k`, the extra slots are `-1`.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `score` | `torch.Tensor` | `[num_kv_heads, total_q, max_seqblock_k]` | `float32` | Required | Block scores, typically produced by `flash_prefill_bnsd_score`; unusable blocks are `-inf`. The Triton backend reads by stride and accepts non-contiguous input. |
| `per_query_seq_lens` | `torch.Tensor` | `[total_q]` | `int32` | Required | Number of KV positions visible to each query (inclusive causal length), built by SGLang as `prefix + within + 1`; values must not exceed the request's `seq_lens` (caller convention). |
| `topk` | `int` | — | — | Required | Number of blocks selected per query; must be positive (asserted). |
| `block_size_k` | `int` | — | — | Required | Tokens per logical KV block; must be positive and match the block size used to produce `score` (caller convention). |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `topk_idx` | `torch.Tensor` | `[num_kv_heads, total_q, topk + 1]` | `int32` | New contiguous tensor on `score`'s device. The first `topk` slots may hold `-1`; the last slot is the local block or `-1`. Inputs are not modified. |

The implementation asserts that `score` is `float32`, `per_query_seq_lens` is `int32` with length
`total_q`, and `topk > 0`. Callers must also provide `block_size_k > 0`, a non-empty score-block
dimension, tensors on the same device, per-query lengths within each request's KV length, and
finite scores or `-inf` for unusable blocks. These caller conventions are not all checked by the
operator.

## triton backend

The implementation selects a query tile from the score width using a UB-oriented heuristic (up to
16 queries per program). `topk` is a compile-time parameter, so different values may compile
separate kernels. The launch uses `num_warps=4` and `num_stages=1`. Scores are read by stride, so
contiguous input is not required. The wrapper does not call `.item()` or synchronize with the host.

## torch_native backend

An independent PyTorch reference that does not call Triton. It takes a max each round and breaks
ties by the lowest column index, matching the Triton selection order; it does not rely on
`torch.topk`/`argsort` tie ordering. The per-round loop and `score.clone()` make it slower and more
memory-hungry than the Triton path; it is intended for accuracy comparison only.

## Example

Requires an installed package containing this operator and the NPU runtime.

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

Row 0: block 2 has the highest score but `2 * 2 > q_pos = 2`, so the first slot is `-1`; block 0 is
selected and `local = 1` is not, so the last slot is 1. Row 1: blocks 2 and 0 are selected and
`local = 2` is already selected, so the last slot is `-1`.
