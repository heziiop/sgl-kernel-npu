"""Public prefill block-topk operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "indexer.flash_prefill_bnsd_topk_from_score"

_triton_fn = None
_torch_native_fn = None


# Importing this module registers the three names without importing Triton.
@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(score, per_query_seq_lens, topk, block_size_k):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.flash_prefill_bnsd_topk_from_score import (
            flash_prefill_bnsd_topk_from_score,
        )

        _triton_fn = flash_prefill_bnsd_topk_from_score

    return _triton_fn(score, per_query_seq_lens, topk, block_size_k)


@register_impl(OP, "torch_native")
def _torch_native_impl(score, per_query_seq_lens, topk, block_size_k):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.flash_prefill_bnsd_topk_from_score import (
            flash_prefill_bnsd_topk_from_score,
        )

        _torch_native_fn = flash_prefill_bnsd_topk_from_score

    return _torch_native_fn(score, per_query_seq_lens, topk, block_size_k)


@register_operator(OP)
def flash_prefill_bnsd_topk_from_score(
    score,
    per_query_seq_lens,
    topk,
    block_size_k,
    *,
    impl="auto",
):
    """Select the topk KV blocks per query and append the query's local block.

    ``auto`` defaults to ``triton``; returns ``int32 [H, Q, topk + 1]`` with
    ``-1`` holes. ``score`` is fp32 ``[H, Q, K]`` and ``topk`` must be positive.
    """
    return get_impl_by_name(OP, impl)(score, per_query_seq_lens, topk, block_size_k)
