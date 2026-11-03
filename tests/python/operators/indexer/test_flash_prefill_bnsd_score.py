"""Accuracy/performance tests for indexer.flash_prefill_bnsd_score.

The Triton entry is compared with the ``torch_native`` reference; the tolerance
below is provisional until the NPU baseline is recorded.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import make_executor
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

BLOCK_K = 128  # Index page size, also the logical KV block.
HEAD_DIM = 128  # Index head dim.
# Log2-domain scores differ only by float32 reduction order.
SCORE_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3)}

# Cases that collect performance.
perf_shape = [
    (305, (80817,), 64, 1, 1),
    (1536, (251536,), 64, 1, 1),
]


def generate_test_data(
    q_lens,
    seq_lens,
    block_size_q,
    idx_heads=1,
    kv_heads=1,
    dtype=torch.bfloat16,
    seed=42,
):
    """Build NPU inputs; each request maps logical KV blocks to consecutive pages."""
    blocks = max((int(length) + BLOCK_K - 1) // BLOCK_K for length in seq_lens)
    requests = len(seq_lens)
    torch.npu.manual_seed(seed)
    k_cache = torch.randn(requests * blocks, BLOCK_K, kv_heads, HEAD_DIM, dtype=dtype)
    q = torch.randn(sum(q_lens), idx_heads, HEAD_DIM, dtype=dtype)

    req_to_token = torch.zeros(requests, blocks * BLOCK_K, dtype=torch.int32)
    pages = torch.arange(requests).view(-1, 1) * blocks
    pages = pages + torch.arange(blocks).view(1, -1)
    req_to_token[:, ::BLOCK_K] = pages * BLOCK_K

    return dict(
        q=q,
        k_cache_bnsd=k_cache,
        cu_seqlens=torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        req_to_token=req_to_token,
        req_pool_indices=torch.arange(requests, dtype=torch.int32),
        block_size_q=block_size_q,
        block_size_k=BLOCK_K,
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
            ((305,), (80817,), 64, 1, 1), id="minimax-m3-prefill-1x80k-new305"
        ),
        pytest.param(
            ((773,), (125701,), 64, 1, 1), id="minimax-m3-prefill-1x125k-new773"
        ),
        pytest.param(
            ((1536,), (251536,), 64, 1, 1), id="minimax-m3-prefill-1x250k-new1536"
        ),
        # Edge cases; all single-head (the M3 sparse indexer has no GQA).
        pytest.param(((1,), (1,), 1, 1, 1), id="edge-single-block"),
        pytest.param(((256,), (256,), 64, 1, 1), id="edge-exact-blocks"),
        pytest.param(((129,), (129,), 16, 1, 1), id="edge-one-over"),
        pytest.param(((7,), (300,), 4, 1, 1), id="edge-query-tail"),
        pytest.param(((9,), (300,), 1, 1, 1), id="edge-bsq1"),
        pytest.param(((40,), (300,), 16, 1, 1), id="edge-bsq16"),
        pytest.param(((70,), (900,), 32, 1, 1), id="edge-bsq32"),
        pytest.param(((37,), (300,), 10, 1, 1), id="edge-bsq-nonpow2"),
        pytest.param(
            ((0, 5, 129, 40), (0, 700, 129, 4096), 16, 1, 1), id="edge-empty-request"
        ),
        pytest.param(((0,), (0,), 16, 1, 1), id="edge-all-empty"),
    ],
)
@pytest.mark.parametrize("score_type", ["max", "lse"])
def test_flash_prefill_bnsd_score(case, dtype, score_type):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score

    q_lens, seq_lens, block_size_q, idx_heads, kv_heads = case
    inputs = generate_test_data(
        q_lens, seq_lens, block_size_q, idx_heads, kv_heads, dtype
    )
    executor = check_accuracy(
        flash_prefill_bnsd_score,
        kwargs={**inputs, "score_type": score_type},
        standard=SCORE_STANDARD,
    )
    enable_perf = case in perf_shape and score_type == "max" and dtype == torch.bfloat16
    check_performance(executor, enable_perf=enable_perf)


def test_16k_queries_with_200k_context():
    """Largest production shape; the reference is too costly to compare here."""
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score

    inputs = generate_test_data((16384,), (216384,), 64)
    executor = make_executor(flash_prefill_bnsd_score, kwargs=inputs)
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
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_score

    inputs = generate_test_data((0, 5, 129, 40), (0, 700, 129, 4096), 16)
    mappings = build_qblock_mappings(inputs)
    built_in = flash_prefill_bnsd_score(**inputs)
    reused = flash_prefill_bnsd_score(**inputs, qblock_mappings=mappings)
    torch.testing.assert_close(reused, built_in, atol=0, rtol=0)
