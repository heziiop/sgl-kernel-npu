"""Public prefill score + index-head attention operator with explicit impl selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "indexer.flash_prefill_bnsd_score_attn"

_triton_fn = None
_torch_native_fn = None


# Importing this module registers the three names without importing Triton.
@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(
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
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.flash_prefill_bnsd_score_attn import (
            flash_prefill_bnsd_score_attn,
        )

        _triton_fn = flash_prefill_bnsd_score_attn

    return _triton_fn(
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
        score_type=score_type,
        qblock_mappings=qblock_mappings,
    )


@register_impl(OP, "torch_native")
def _torch_native_impl(
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
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.flash_prefill_bnsd_score_attn import (
            flash_prefill_bnsd_score_attn,
        )

        _torch_native_fn = flash_prefill_bnsd_score_attn

    return _torch_native_fn(
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
        score_type=score_type,
        qblock_mappings=qblock_mappings,
    )


@register_operator(OP)
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
    """Score every (query, KV block) pair and run causal index-head attention.

    The default ``impl="auto"`` resolves to the Triton entry; ``"triton"`` is the
    same function and ``"torch_native"`` is an independent reference.
    ``q`` is ``[total_q, num_idx_heads, head_dim]``, ``k_cache_bnsd`` and
    ``v_cache_bnsd`` are ``[num_pages, page_size, num_kv_heads, head_dim]``, and
    ``num_idx_heads`` must be divisible by ``num_kv_heads``; ``score_type`` is
    ``"max"`` or ``"lse"``. Returns ``(score, idx_o)``: ``score`` is a new fp32
    ``[num_idx_heads, total_q, max_seqblock_k]`` tensor that is ``-inf`` where no
    score was written, and ``idx_o`` is a new ``q.dtype``
    ``[total_q, num_idx_heads, head_dim]`` tensor that is zero where no row was
    computed. Inputs are not modified.
    """

    return get_impl_by_name(OP, impl)(
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
        score_type=score_type,
        qblock_mappings=qblock_mappings,
    )
