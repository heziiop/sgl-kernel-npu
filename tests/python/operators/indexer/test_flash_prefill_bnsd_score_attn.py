"""Accuracy/performance tests for indexer.flash_prefill_bnsd_score_attn.

The Triton entry is compared with the ``torch_native`` reference; tolerances are
provisional until the NPU baseline is recorded.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import make_executor
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

HEAD_DIM = 128  # Index head dim (sparse_index_dim).
BLOCK_K = 128  # Default page size.
# Log2-domain scores differ only by float32 reduction order.
SCORE_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3)}

# Cases that collect performance: production shapes, max score type, bf16.
perf_shape = [
    ((305,), (80817,), 64, 1, 1, 128),
    ((1536,), (251536,), 64, 1, 1, 128),
    ((305,), (80817,), 64, 4, 1, 128),
]


def generate_test_data(
    q_lens,
    seq_lens,
    block_size_q,
    idx_heads=1,
    kv_heads=1,
    block_size_k=BLOCK_K,
    dtype=torch.bfloat16,
    seed=42,
):
    """Build NPU inputs; each request maps logical KV blocks to consecutive pages."""
    blocks = max(
        (int(length) + block_size_k - 1) // block_size_k for length in seq_lens
    )
    requests = len(seq_lens)
    torch.npu.manual_seed(seed)
    k_cache = torch.randn(
        requests * blocks, block_size_k, kv_heads, HEAD_DIM, dtype=dtype
    )
    v_cache = torch.randn(
        requests * blocks, block_size_k, kv_heads, HEAD_DIM, dtype=dtype
    )
    q = torch.randn(sum(q_lens), idx_heads, HEAD_DIM, dtype=dtype)

    req_to_token = torch.zeros(requests, blocks * block_size_k, dtype=torch.int32)
    pages = torch.arange(requests).view(-1, 1) * blocks
    pages = pages + torch.arange(blocks).view(1, -1)
    req_to_token[:, ::block_size_k] = pages * block_size_k

    return dict(
        q=q,
        k_cache_bnsd=k_cache,
        v_cache_bnsd=v_cache,
        cu_seqlens=torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        req_to_token=req_to_token,
        req_pool_indices=torch.arange(requests, dtype=torch.int32),
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        sm_scale=HEAD_DIM**-0.5,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16"),
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        # MiniMax-M3 prefill lengths from the upstream benchmark.
        pytest.param(
            ((305,), (80817,), 64, 1, 1, 128), id="minimax-m3-prefill-1x80k-new305"
        ),
        pytest.param(
            ((773,), (125701,), 64, 1, 1, 128), id="minimax-m3-prefill-1x125k-new773"
        ),
        pytest.param(
            ((1536,), (251536,), 64, 1, 1, 128),
            id="minimax-m3-prefill-1x250k-new1536",
        ),
        # TP=1 has up to four index heads sharing the single index KV head.
        pytest.param(
            ((305,), (80817,), 64, 4, 1, 128),
            id="minimax-m3-prefill-1x80k-new305-gqa4",
        ),
        # Edge cases; all single-head unless stated.
        pytest.param(((1,), (1,), 1, 1, 1, 128), id="edge-single-block"),
        pytest.param(((256,), (256,), 64, 1, 1, 128), id="edge-exact-blocks"),
        pytest.param(((129,), (129,), 16, 1, 1, 128), id="edge-one-over"),
        pytest.param(((7,), (300,), 4, 1, 1, 128), id="edge-query-tail"),
        pytest.param(((9,), (300,), 1, 1, 1, 128), id="edge-bsq1"),
        pytest.param(((40,), (300,), 16, 1, 1, 128), id="edge-bsq16"),
        pytest.param(((70,), (900,), 32, 1, 1, 128), id="edge-bsq32"),
        pytest.param(((37,), (300,), 10, 1, 1, 128), id="edge-bsq-nonpow2"),
        pytest.param(
            ((0, 5, 129, 40), (0, 700, 129, 4096), 16, 1, 1, 128),
            id="edge-empty-request",
        ),
        pytest.param(((0,), (0,), 16, 1, 1, 128), id="edge-all-empty"),
    ],
)
@pytest.mark.parametrize("score_type", ["max", "lse"])
def test_flash_prefill_bnsd_score_attn(case, dtype, score_type):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    q_lens, seq_lens, block_size_q, idx_heads, kv_heads, block_size_k = case
    inputs = generate_test_data(
        q_lens,
        seq_lens,
        block_size_q,
        idx_heads,
        kv_heads,
        block_size_k,
        dtype,
    )
    executor = check_accuracy(
        flash_prefill_bnsd_score_attn,
        kwargs={**inputs, "score_type": score_type},
        standard=SCORE_STANDARD,
    )
    enable_perf = case in perf_shape and score_type == "max" and dtype == torch.bfloat16
    check_performance(executor, enable_perf=enable_perf)


def test_16k_queries_with_200k_context():
    """Largest production shape; the reference is too costly to compare here."""
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    inputs = generate_test_data((16384,), (216384,), 64)
    executor = make_executor(flash_prefill_bnsd_score_attn, kwargs=inputs)
    check_performance(executor, enable_perf=True)


def build_qblock_mappings(inputs):
    """Six-tuple the operator accepts when the caller precomputes it."""
    cu = inputs["cu_seqlens"].to(torch.long)
    seq = inputs["seq_lens"].to(torch.long)
    block_size_q = inputs["block_size_q"]
    page = inputs["block_size_k"]
    max_blocks = (int(seq.max().item()) + page - 1) // page
    per_req = (cu[1:] - cu[:-1] + block_size_q - 1) // block_size_q
    total_qb = int(per_req.sum().item())
    req = inputs["req_pool_indices"].to(torch.long).repeat_interleave(per_req)
    cu_blocks = torch.zeros_like(per_req)
    cu_blocks[1:] = per_req[:-1].cumsum(0)
    local = torch.arange(total_qb, device=cu.device)
    local = local - cu_blocks.repeat_interleave(per_req)
    cols = torch.arange(max_blocks, device=cu.device) * page
    slots = inputs["req_to_token"][req[:, None], cols]
    return (
        cu[:-1].repeat_interleave(per_req).to(torch.int32),
        local.to(torch.int32),
        seq.repeat_interleave(per_req).to(torch.int32),
        cu[1:].repeat_interleave(per_req).to(torch.int32),
        (slots // page).to(torch.int32),
        total_qb,
    )


@pytest.mark.extra_checks
def test_prebuilt_qblock_mappings_match_built_in():
    """SGLang reuses one mapping per forward, so both paths must agree."""
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    inputs = generate_test_data((0, 5, 129, 40), (0, 700, 129, 4096), 16)
    mappings = build_qblock_mappings(inputs)
    built_in = flash_prefill_bnsd_score_attn(**inputs)
    reused = flash_prefill_bnsd_score_attn(**inputs, qblock_mappings=mappings)
    torch.testing.assert_close(reused[0], built_in[0], atol=0, rtol=0)
    torch.testing.assert_close(reused[1], built_in[1], atol=0, rtol=0)


@pytest.mark.extra_checks
def test_score_matches_score_operator():
    """The fused operator's score must equal the score-only operator's output."""
    from sgl_kernel_npu.operators.indexer import (
        flash_prefill_bnsd_score,
        flash_prefill_bnsd_score_attn,
    )

    inputs = generate_test_data((7,), (300,), 16)
    score_kwargs = {k: v for k, v in inputs.items() if k != "v_cache_bnsd"}
    score_only = flash_prefill_bnsd_score(**score_kwargs)
    score_fused = flash_prefill_bnsd_score_attn(**inputs)[0]
    torch.testing.assert_close(score_fused, score_only, atol=1e-6, rtol=1e-6)


@pytest.mark.extra_checks
@pytest.mark.parametrize("score_type", ["max", "lse"])
def test_naive_attention_oracle(score_type):
    """Small-shape full-softmax oracle, independent of the torch_native reference."""
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    inputs = generate_test_data((2, 3), (2, 20), 16, block_size_k=16)
    result = flash_prefill_bnsd_score_attn(
        **inputs, score_type=score_type, impl="triton"
    )
    score_ref, idx_ref = _naive_score_attn(inputs, score_type)
    torch.testing.assert_close(result[0], score_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(result[1].float(), idx_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.extra_checks
def test_multi_kv_head():
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    inputs = generate_test_data((7,), (300,), 16, idx_heads=4, kv_heads=2)
    executor = check_accuracy(flash_prefill_bnsd_score_attn, kwargs=inputs)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
def test_page_size_64():
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    inputs = generate_test_data((7,), (300,), 16, block_size_k=64)
    executor = check_accuracy(flash_prefill_bnsd_score_attn, kwargs=inputs)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
def test_int64_metadata():
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    inputs = generate_test_data((7,), (300,), 16)
    for key in ("cu_seqlens", "seq_lens", "req_pool_indices"):
        inputs[key] = inputs[key].to(torch.int64)
    executor = check_accuracy(flash_prefill_bnsd_score_attn, kwargs=inputs)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["triton", "torch_native"])
def test_invalid_score_type_is_rejected(impl):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    inputs = generate_test_data((5,), (700,), 16)
    with pytest.raises(AssertionError, match="unsupported score_type"):
        flash_prefill_bnsd_score_attn(**inputs, score_type="sum", impl=impl)


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["triton", "torch_native"])
def test_indivisible_heads_are_rejected(impl):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score_attn

    inputs = generate_test_data((5,), (700,), 16, idx_heads=3, kv_heads=2)
    with pytest.raises(AssertionError):
        flash_prefill_bnsd_score_attn(**inputs, impl=impl)


def _naive_score_attn(inputs, score_type):
    """Direct [query, all-keys] definition, including the kernel's global mask."""
    q = inputs["q"].float()
    k_cache = inputs["k_cache_bnsd"]
    v_cache = inputs["v_cache_bnsd"]
    cu = inputs["cu_seqlens"].tolist()
    lens = inputs["seq_lens"].tolist()
    page = inputs["block_size_k"]
    total_q, num_idx_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    gqa = num_idx_heads // num_kv_heads
    max_blocks = max((int(length) + page - 1) // page for length in lens)

    score = torch.full(
        (num_idx_heads, total_q, max_blocks), float("-inf"), dtype=torch.float32
    )
    idx_o = torch.zeros(total_q, num_idx_heads, head_dim)
    scale = inputs["sm_scale"] * 1.4426950409

    for r, (q_start, q_end, seq_len) in enumerate(zip(cu[:-1], cu[1:], lens)):
        q_len = q_end - q_start
        if q_len == 0:
            continue
        num_blocks = (seq_len + page - 1) // page
        pages = [
            int(inputs["req_to_token"][r, b * page].item()) // page
            for b in range(num_blocks)
        ]
        keys = k_cache[torch.tensor(pages, device=k_cache.device)].float()
        values = v_cache[torch.tensor(pages, device=v_cache.device)].float()
        q_pos = torch.arange(q_start, q_end, device=q.device)
        all_pos = torch.arange(num_blocks * page, device=q.device)

        for kv_head in range(num_kv_heads):
            heads = slice(kv_head * gqa, (kv_head + 1) * gqa)
            for b in range(num_blocks):
                key_pos = b * page + torch.arange(page, device=q.device)
                qk = torch.einsum(
                    "qgd,cd->qgc", q[q_start:q_end, heads, :], keys[b, :, kv_head, :]
                )
                qk = qk * scale
                mask = (q_pos[:, None, None] >= key_pos[None, None, :]) & (
                    key_pos[None, None, :] < seq_len
                )
                qk = qk.masked_fill(~mask, float("-inf"))
                best = qk.max(dim=-1).values
                if score_type == "max":
                    val = best
                else:
                    val = best + torch.log2(
                        torch.exp2(qk - best.unsqueeze(-1)).sum(dim=-1)
                    )
                    val = torch.where(torch.isnan(val), float("-inf"), val)
                score[heads, q_start:q_end, b] = val.permute(1, 0)

            logits = torch.einsum(
                "qgd,cd->qgc",
                q[q_start:q_end, heads, :],
                keys.reshape(num_blocks * page, num_kv_heads, head_dim)[:, kv_head, :],
            )
            logits = logits * scale
            mask = (q_pos[:, None, None] >= all_pos[None, None, :]) & (
                all_pos[None, None, :] < seq_len
            )
            logits = logits.masked_fill(~mask, float("-inf"))
            m = logits.max(dim=-1, keepdim=True).values
            p = torch.exp2(logits - m)
            p = p / p.sum(dim=-1, keepdim=True)
            idx_o[q_start:q_end, heads, :] = torch.einsum(
                "qgp,pd->qgd",
                p,
                values.reshape(num_blocks * page, num_kv_heads, head_dim)[
                    :, kv_head, :
                ],
            )

    return score, idx_o
