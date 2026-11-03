"""Accuracy/performance tests for indexer.flash_prefill_bnsd_topk_from_score."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import make_executor
from tests.python.utils.performance import check_performance

BLOCK_K = 128  # Indexer page size, also the logical KV block.

# (q_lens, seq_lens, topk) selected for performance collection.
perf_shape = [
    ((305,), (80817,), 16),
    ((1536,), (251536,), 16),
]


def _build_per_query_seq_lens(q_lens, seq_lens):
    """Per-query inclusive causal KV length, as SGLang's _build_prefill_meta."""
    q_lens_t = torch.tensor(q_lens, dtype=torch.long)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long)
    per_query_prefix = (seq_lens_t - q_lens_t).repeat_interleave(q_lens_t)
    cu = torch.tensor([0, *q_lens], dtype=torch.long).cumsum(0)
    within = torch.arange(int(q_lens_t.sum())) - cu[:-1].repeat_interleave(q_lens_t)
    return (per_query_prefix + within + 1).to(torch.int32)


def generate_test_data(q_lens, seq_lens, topk, mode, heads=1, seed=42):
    """Create NPU inputs; each request uses ceil(seq_len / BLOCK_K) blocks.

    random: general scores exercise the ordered selection.
    ties: rounded scores create equal candidates and exercise the tie-break.
    dense: every column is finite, so causally future blocks become -1 holes.
    all_masked: every score is -inf, leaving only the appended local block.
    """
    torch.npu.manual_seed(seed)
    total_q = sum(q_lens)
    per_req_blocks = [(int(s) + BLOCK_K - 1) // BLOCK_K for s in seq_lens]
    max_blocks = max(per_req_blocks)
    per_query_seq_lens = _build_per_query_seq_lens(q_lens, seq_lens)
    score = torch.randn(heads, total_q, max_blocks, dtype=torch.float32)
    if mode == "ties":
        score = torch.round(score)
    elif mode == "all_masked":
        score.fill_(float("-inf"))
    elif mode not in ("random", "dense"):
        raise ValueError(f"unknown mode: {mode}")
    if mode != "dense":
        # Blocks beyond a request's KV length are never written by the score op.
        row = 0
        for q_len, n_blocks in zip(q_lens, per_req_blocks):
            if q_len and n_blocks < max_blocks:
                score[:, row : row + q_len, n_blocks:] = float("-inf")
            row += q_len
    return dict(
        score=score,
        per_query_seq_lens=per_query_seq_lens,
        topk=topk,
        block_size_k=BLOCK_K,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "random",  # General scores.
        "ties",  # Equal candidates: the lowest block index wins.
        "dense",  # Finite future blocks: -1 holes in the selected slots.
        "all_masked",  # All -inf: every topk slot is -1.
    ],
)
@pytest.mark.parametrize(
    "q_lens, seq_lens, topk",
    [
        pytest.param((305,), (80817,), 16, id="minimax-m3-prefill-1x80k-new305"),
        pytest.param((1536,), (251536,), 16, id="minimax-m3-prefill-1x250k-new1536"),
        pytest.param(
            (0, 5, 129, 40),
            (0, 700, 129, 4096),
            16,
            id="minimax-m3-prefill-packed",
        ),
        pytest.param((1,), (1,), 1, id="edge-single-block"),
        pytest.param((7,), (300,), 4, id="edge-topk-gt-k"),
    ],
)
def test_flash_prefill_bnsd_topk_from_score(q_lens, seq_lens, topk, mode):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_topk_from_score

    inputs = generate_test_data(q_lens, seq_lens, topk, mode)
    executor = check_accuracy(flash_prefill_bnsd_topk_from_score, kwargs=inputs)
    enable_perf = (q_lens, seq_lens, topk) in perf_shape and mode == "random"
    check_performance(executor, enable_perf=enable_perf)


def test_16k_queries_with_200k_context():
    """Largest production shape; the reference is too costly to compare here."""
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_topk_from_score

    executor = make_executor(
        flash_prefill_bnsd_topk_from_score,
        kwargs=generate_test_data((16384,), (216384,), 16, "random"),
    )
    check_performance(executor, enable_perf=True)


@pytest.mark.extra_checks
def test_multi_head():
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_topk_from_score

    inputs = generate_test_data(
        (0, 5, 129, 40), (0, 700, 129, 4096), 16, "random", heads=2
    )
    check_accuracy(flash_prefill_bnsd_topk_from_score, kwargs=inputs)


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["triton", "torch_native"])
def test_zero_topk_is_rejected(impl):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_topk_from_score

    inputs = generate_test_data((5,), (700,), 4, "random")
    inputs["topk"] = 0
    with pytest.raises(AssertionError, match="topk must be positive"):
        flash_prefill_bnsd_topk_from_score(**inputs, impl=impl)
