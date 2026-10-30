"""P0 baseline: compare the legacy and migrated flash_decode_bnsd_with_topk_idx.

The migration copies kernels into ``implementation/triton`` and adds an
``operators`` adapter, so the legacy entry is the reference for the new one:
this script runs both on a fixed input matrix and reports any difference.

Usage (needs an Ascend NPU; exits 0 without one)::

    # 1) record the legacy outputs (use_native=False) as the local baseline
    python tests/python/operators/indexer/compare_legacy_vs_new.py --mode record

    # 2) compare the new entry against the recorded baseline later
    python tests/python/operators/indexer/compare_legacy_vs_new.py --mode compare

    # one-shot: record and compare in the same process (same inputs)
    python tests/python/operators/indexer/compare_legacy_vs_new.py --mode both

``topk_idx`` is compared exactly; ``o`` is compared with ``--atol``/``--rtol``
(default 0, bitwise, because both sides run the same Triton kernels). The
baseline lives under ``results/golden/`` (git-ignored, not used by CI) and
records the environment, because inputs are regenerated from per-case seeds.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GOLDEN_DIR = (
    REPO_ROOT / "results" / "golden" / "flash_decode_bnsd_with_topk_idx"
)


def case(cid, **overrides):
    """Describe one baseline case.

    Case ids are aligned with the operator test matrix where possible.
    """
    values = dict(
        id=cid,
        batch=1,
        q_heads=1,
        kv_heads=1,
        head_dim=128,
        block_size=128,
        topk=16,
        seq_lens=[4096],
        max_seqlen=4096,
        disable_index_value=True,
        page="direct",
        score_type="max",
        init_blocks=0,
        local_blocks=0,
        packed=False,
        adaptive=None,
        score_max_chunks=32,
        blocks_per_chunk=16,
        sanitize=False,
        sink=False,
        dtype=torch.bfloat16,
        index_dtype=torch.int32,
        mode="spaced",
        runtime_fill_only=True,
        fused=True,
        seed=42,
        big=False,
    )
    values.update(overrides)
    return values


CASES = [
    case("decode-1x4k"),
    case(
        "decode-4x8k-tp1",
        batch=4,
        q_heads=4,
        seq_lens=[8192] * 4,
        max_seqlen=8192,
    ),
    case(
        "decode-4x32k-tp2",
        batch=4,
        q_heads=2,
        seq_lens=[32768] * 4,
        max_seqlen=32768,
    ),
    case("decode-16x4k", batch=16, seq_lens=[4096] * 16),
    case("decode-graph-cap", seq_lens=[4096], max_seqlen=1048576),
    case(
        "verify-packed-flat",
        batch=12,
        seq_lens=[4096] * 12,
        max_seqlen=1048576,
        packed=True,
    ),
    case(
        "verify-packed-4x3",
        batch=4,
        q_heads=3,
        seq_lens=[4096 + index for index in range(12)],
        max_seqlen=1048576,
        packed=True,
        adaptive=(256, 16),
        blocks_per_chunk=8,
        score_max_chunks=64,
        sanitize=True,
        runtime_fill_only=False,
    ),
    case("short-passthrough", seq_lens=[100]),
    case("empty", seq_lens=[0], max_seqlen=128, topk=4),
    case("topk-zero", topk=0, fused=False),
    case("lse", score_type="lse"),
    case("decode-nonfused", seq_lens=[4096], fused=False),
    case("local-append", seq_lens=[512], topk=2, mode="spaced-desc"),
    case("init-local", seq_lens=[1024], topk=4, init_blocks=1, local_blocks=1),
    case("block-table", batch=2, seq_lens=[4096, 1024], page="table"),
    case("unaligned", batch=2, seq_lens=[129, 257], max_seqlen=257, topk=1),
    case("idx-int64", batch=2, seq_lens=[4096, 1024], index_dtype=torch.int64),
    case(
        "gqa-kvh2",
        batch=2,
        q_heads=4,
        kv_heads=2,
        seq_lens=[4096, 2048],
        page="table",
        topk=8,
    ),
    case("fp16", dtype=torch.float16),
    case(
        "attn-none",
        batch=2,
        q_heads=2,
        seq_lens=[1024, 1024],
        max_seqlen=1024,
        disable_index_value=False,
        page="table",
        mode="random",
        fused=False,
    ),
    case(
        "attn-sink-lse",
        batch=2,
        q_heads=2,
        seq_lens=[1024, 1024],
        max_seqlen=1024,
        disable_index_value=False,
        page="table",
        mode="random",
        score_type="lse",
        sink=True,
        fused=False,
    ),
    case(
        "attn-init-local",
        batch=2,
        q_heads=2,
        seq_lens=[1024, 1024],
        max_seqlen=1024,
        disable_index_value=False,
        page="table",
        mode="random",
        sink=True,
        init_blocks=1,
        local_blocks=1,
        fused=False,
    ),
    case(
        "decode-1x1m",
        seq_lens=[1048576],
        max_seqlen=1048576,
        score_max_chunks=16,
        big=True,
    ),
    case(
        "decode-1x128k",
        seq_lens=[131072],
        max_seqlen=131072,
        score_max_chunks=16,
        big=True,
    ),
]


def _seed_everything(seed):
    torch.manual_seed(seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "manual_seed"):
        npu.manual_seed(seed)


def build_inputs(spec, device):
    """Create the operator keyword arguments for one case (inputs on device)."""
    _seed_everything(spec["seed"])
    batch = spec["batch"]
    q_heads = spec["q_heads"]
    kv_heads = spec["kv_heads"]
    head_dim = spec["head_dim"]
    block_size = spec["block_size"]
    topk = spec["topk"]
    dtype = spec["dtype"]
    gqa = q_heads // kv_heads
    seq_lens = list(spec["seq_lens"])
    expected_rows = batch * gqa if spec["packed"] else batch
    if len(seq_lens) != expected_rows:
        raise ValueError(
            f"{spec['id']}: seq_lens has {len(seq_lens)} rows, "
            f"expected {expected_rows}"
        )
    num_blocks = [(value + block_size - 1) // block_size for value in seq_lens]
    max_blocks = max(num_blocks)
    capacity = max(1, max_blocks)
    if spec["sanitize"]:
        capacity += 2

    q = torch.randn(batch, q_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.zeros(
        capacity, block_size, kv_heads, head_dim, dtype=dtype, device=device
    )
    index = torch.arange(capacity, dtype=torch.float32, device=device)
    if spec["mode"] == "spaced-desc":
        index = index.flip(0)
    elif spec["mode"] not in ("spaced", "random"):
        raise ValueError(f"{spec['id']}: unknown mode {spec['mode']!r}")
    if spec["mode"] == "random":
        k_cache = torch.randn(
            capacity, block_size, kv_heads, head_dim, dtype=dtype, device=device
        )
    else:
        # Two bf16-exact channels encode the block index and a third one the
        # token position; q=[1, 256, 1] reads them back, so the block score is
        # ``index + last_valid_token``: distinct at any context length and
        # sensitive to a wrong causal mask.
        for kv_head in range(kv_heads):
            shifted = (index + kv_head * 7) % capacity
            k_cache[:, :, kv_head, 0] = (shifted % 256).view(-1, 1)
            k_cache[:, :, kv_head, 1] = (shifted // 256).view(-1, 1)
        token_ids = torch.arange(block_size, dtype=torch.float32, device=device)
        k_cache[:, :, :, 2] = token_ids.view(1, -1, 1)
        q[:, :, 0] = 1.0
        q[:, :, 1] = 256.0
        q[:, :, 2] = 1.0
        encoded = k_cache[:, 0, 0, 0].float() + 256.0 * k_cache[:, 0, 0, 1].float()
        assert (
            torch.unique(encoded).numel() == capacity
        ), f"{spec['id']}: block scores must stay distinct after the dtype cast"

    v_cache = None
    if not spec["disable_index_value"]:
        v_cache = torch.randn(
            capacity, block_size, kv_heads, head_dim, dtype=dtype, device=device
        )
    sink = None
    if spec["sink"]:
        sink = torch.randn(q_heads, head_dim, dtype=dtype, device=device)

    block_table = None
    req_to_token = None
    req_pool_indices = None
    if spec["page"] == "table":
        block_table = (
            torch.arange(capacity, dtype=spec["index_dtype"], device=device)
            .view(1, -1)
            .repeat(batch, 1)
            .contiguous()
        )
    else:
        columns = capacity * block_size
        req_to_token = torch.zeros(
            batch, columns, dtype=spec["index_dtype"], device=device
        )
        token_cols = (
            torch.arange(capacity, dtype=torch.long, device=device) * block_size
        )
        token_slots = (
            torch.arange(capacity, dtype=spec["index_dtype"], device=device)
            * block_size
        )
        req_to_token[:, token_cols] = token_slots
        if spec["sanitize"]:
            # The last used logical block points outside the page table and is
            # clamped to num_pages - 1 by sanitize_page_ids.
            req_to_token[0, (max_blocks - 1) * block_size] = (capacity + 4) * block_size
        req_pool_indices = torch.arange(batch, dtype=spec["index_dtype"], device=device)

    kwargs = dict(
        q=q,
        sink=sink,
        k_cache_bnsd=k_cache,
        v_cache_bnsd=v_cache,
        block_table=block_table,
        seq_lens=torch.tensor(seq_lens, dtype=spec["index_dtype"], device=device),
        max_seqlen=spec["max_seqlen"],
        block_size=block_size,
        topk=topk,
        init_blocks=spec["init_blocks"],
        local_blocks=spec["local_blocks"],
        sm_scale=None,
        score_type=spec["score_type"],
        disable_index_value=spec["disable_index_value"],
        num_kv_chunks=None,
        max_num_kv_chunks=8,
        use_triton_topk=None,
        num_topk_chunks=None,
        use_chunked_score=True,
        score_blocks_per_chunk=spec["blocks_per_chunk"],
        score_max_chunks=spec["score_max_chunks"],
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=capacity,
        num_pages=capacity,
        sanitize_page_ids=spec["sanitize"],
        packed_seq_lens=spec["packed"],
        runtime_fill_only=spec["runtime_fill_only"],
        fused_append_local=spec["fused"],
    )
    if spec["adaptive"] is not None:
        short_blocks, short_chunks = spec["adaptive"]
        kwargs["runtime_score_short_max_blocks"] = short_blocks
        kwargs["runtime_score_short_chunks"] = short_chunks
    return kwargs


def _import_ops():
    from sgl_kernel_npu.indexer.flash_block_score_decode import (
        flash_decode_bnsd_with_topk_idx as legacy_op,
    )
    from sgl_kernel_npu.operators.indexer import (
        flash_decode_bnsd_with_topk_idx as new_op,
    )

    return legacy_op, new_op


def _load_ops():
    """Import both entries, falling back to the source tree if needed."""
    try:
        return _import_ops()
    except Exception as installed_error:
        python_dir = REPO_ROOT / "python"
        if not python_dir.is_dir():
            raise
        sys.path.insert(0, str(python_dir))
        try:
            return _import_ops()
        except Exception as source_error:
            raise RuntimeError(
                f"installed import failed ({installed_error}); "
                f"source-tree import failed ({source_error})"
            ) from source_error


def _npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


def _environment():
    environment = {
        "torch": torch.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import torch_npu

        environment["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
    except ImportError:
        environment["torch_npu"] = "unavailable"
    cann = getattr(torch.version, "cann", None)
    if cann:
        environment["cann"] = cann
    try:
        environment["device"] = torch.npu.get_device_name(0)
    except Exception as error:  # pragma: no cover - depends on the driver
        environment["device"] = f"unknown: {error}"
    return environment


def _compare_outputs(o_new, idx_new, ref_o, ref_idx, atol, rtol):
    """Return a list of human-readable differences (empty means equal)."""
    problems = []
    if (o_new is None) != (ref_o is None):
        problems.append(
            "o presence: "
            f"new={'None' if o_new is None else 'tensor'}, "
            f"legacy={'None' if ref_o is None else 'tensor'}"
        )
    elif ref_o is not None:
        o_new = o_new.detach().cpu()
        if o_new.shape != ref_o.shape or o_new.dtype != ref_o.dtype:
            problems.append(
                f"o layout: new={tuple(o_new.shape)}/{o_new.dtype}, "
                f"legacy={tuple(ref_o.shape)}/{ref_o.dtype}"
            )
        else:
            try:
                torch.testing.assert_close(
                    o_new, ref_o, atol=atol, rtol=rtol, equal_nan=False
                )
            except AssertionError as error:
                message = str(error).strip().splitlines()[0]
                problems.append(f"o values: {message}")

    idx_new = idx_new.detach().cpu()
    if idx_new.shape != ref_idx.shape or idx_new.dtype != ref_idx.dtype:
        problems.append(
            f"topk_idx layout: new={tuple(idx_new.shape)}/{idx_new.dtype}, "
            f"legacy={tuple(ref_idx.shape)}/{ref_idx.dtype}"
        )
    elif not torch.equal(idx_new, ref_idx):
        mismatched = int((idx_new != ref_idx).sum().item())
        positions = torch.nonzero(idx_new != ref_idx, as_tuple=False)[:3]
        examples = [tuple(position.tolist()) for position in positions]
        problems.append(f"topk_idx: {mismatched} mismatched slots, first at {examples}")
    return problems


def _select_cases(args):
    if args.case:
        wanted = set(args.case)
        selected = [spec for spec in CASES if spec["id"] in wanted]
        unknown = wanted - {spec["id"] for spec in selected}
        if unknown:
            raise SystemExit(f"unknown case ids: {sorted(unknown)}")
        return selected
    return [spec for spec in CASES if args.big or not spec["big"]]


def _record_path(golden_dir):
    return golden_dir / "golden.pt"


def _meta_path(golden_dir):
    return golden_dir / "meta.json"


def _case_summary(spec):
    summary = {
        key: value
        for key, value in spec.items()
        if key not in ("dtype", "index_dtype", "expected", "big")
    }
    summary["dtype"] = str(spec["dtype"])
    summary["index_dtype"] = str(spec["index_dtype"])
    return summary


def _save_baseline(golden_dir, results, cases, merge, force):
    """Write the baseline; with merge, keep cases recorded earlier."""
    golden_dir.mkdir(parents=True, exist_ok=True)
    path = _record_path(golden_dir)
    meta_path = _meta_path(golden_dir)
    if path.is_file() and not merge and not force:
        raise SystemExit(
            f"baseline already exists at {path}; pass --force to overwrite it, "
            "or record specific --case entries to merge"
        )
    if merge and path.is_file():
        existing = torch.load(path, map_location="cpu")
        existing.update(results)
        results = existing
    meta = _environment()
    cases_meta = {}
    if merge and meta_path.is_file():
        try:
            stored = json.loads(meta_path.read_text()).get("cases", {})
        except ValueError:
            stored = {}
        if isinstance(stored, dict):
            cases_meta = stored
        else:
            print("warning: meta.json uses a legacy case list; ignored")
    cases_meta.update({spec["id"]: _case_summary(spec) for spec in cases})
    meta["cases"] = cases_meta
    meta["seeds"] = {key: value["seed"] for key, value in cases_meta.items()}
    torch.save(results, path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"recorded {len(results)} cases -> {path}")
    print(f"environment -> {meta_path}")


def _load_baseline(golden_dir):
    path = _record_path(golden_dir)
    if not path.is_file():
        raise SystemExit(
            f"no baseline at {path}; run --mode record first "
            "(on the same chip/CANN version)"
        )
    meta = {}
    if _meta_path(golden_dir).is_file():
        try:
            meta = json.loads(_meta_path(golden_dir).read_text())
        except ValueError:
            meta = {}
    return torch.load(path, map_location="cpu"), meta


def _warn_environment(meta):
    """Print a warning when the baseline was recorded in another environment."""
    if not meta:
        print("warning: baseline has no usable meta.json; environment unknown")
        return
    current = _environment()
    for key in ("torch", "torch_npu", "cann", "device"):
        recorded = meta.get(key)
        if recorded and current.get(key) and recorded != current[key]:
            print(
                f"warning: {key} changed since the baseline: "
                f"{recorded!r} -> {current[key]!r}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["record", "compare", "both"], default="both")
    parser.add_argument("--case", action="append", help="case id to run (repeatable)")
    parser.add_argument(
        "--big", action="store_true", help="include the 128K and 1M cases"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing baseline"
    )
    parser.add_argument(
        "--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR, help="baseline directory"
    )
    parser.add_argument("--atol", type=float, default=0.0, help="o absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.0, help="o relative tolerance")
    args = parser.parse_args(argv)

    cases = _select_cases(args)
    if not _npu_available():
        print("skip: an Ascend NPU is required to record or compare the baseline")
        return 0

    try:
        legacy_op, new_op = _load_ops()
    except Exception as error:
        print(f"error: cannot import the operator entries: {error}")
        return 2

    device = torch.device("npu")
    baseline = None
    if args.mode == "compare":
        baseline, baseline_meta = _load_baseline(args.golden_dir)
        _warn_environment(baseline_meta)
    recorded = {}
    failures = 0
    for spec in cases:
        kwargs = build_inputs(spec, device)
        if args.mode in ("record", "both"):
            legacy_o, legacy_idx = legacy_op(**kwargs, use_native=False)
            recorded[spec["id"]] = {
                "o": None if legacy_o is None else legacy_o.detach().cpu(),
                "topk_idx": legacy_idx.detach().cpu(),
            }
            print(f"[record] {spec['id']} topk={tuple(legacy_idx.shape)}")
            if args.mode == "record":
                continue
            reference = recorded[spec["id"]]
        else:
            if spec["id"] not in baseline:
                raise SystemExit(
                    f"case {spec['id']!r} is missing from the baseline; "
                    "re-record it or drop --case"
                )
            reference = baseline[spec["id"]]

        new_o, new_idx = new_op(**kwargs, use_native=False, impl="triton")
        problems = _compare_outputs(
            new_o,
            new_idx,
            reference["o"],
            reference["topk_idx"],
            args.atol,
            args.rtol,
        )
        if problems:
            failures += 1
            print(f"[FAIL] {spec['id']}")
            for problem in problems:
                print(f"       {problem}")
        else:
            print(f"[ok] {spec['id']}")

    if args.mode in ("record", "both"):
        _save_baseline(
            args.golden_dir, recorded, cases, merge=bool(args.case), force=args.force
        )
    if args.mode in ("compare", "both"):
        print(f"compared {len(cases)} cases, failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
