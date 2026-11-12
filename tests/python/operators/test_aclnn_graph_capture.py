"""Capture lifetime regressions for new ACLNN adapters on arch22.

Use existing operator fixtures and compare graph replay with eager AscendC.
The repeated calls expose temporary-storage retention even when one captured
call and its numerical result look correct.
"""

import gc
import importlib

import pytest

torch = pytest.importorskip("torch")


def _make_graph_case(name):
    root = "tests.python.operators."
    if name == "recurrent_gated_delta_rule":
        mod = importlib.import_module(root + "attention.gdn.test_" + name)
        data = mod.generate_test_data("benchmark", (2, 1, 8, 16, 128, 128), "npu")
        api = importlib.import_module("sgl_kernel_npu.operators.attention.gdn")
        return getattr(api, name), (), data, (data["recurrent_state"],)
    if name == "chunk_gated_delta_rule":
        mod = importlib.import_module(root + "attention.gdn.test_" + name)
        data = mod.generate_test_data("npu")
        api = importlib.import_module("sgl_kernel_npu.operators.attention.gdn")
        return getattr(api, name), (), data, ()
    if name == "causal_conv1d":
        mod = importlib.import_module(root + "mamba.test_" + name)
        data = mod.clone_test_data(mod.generate_test_data(0, 1), "npu")
        api = importlib.import_module("sgl_kernel_npu.operators.mamba")
        return getattr(api, name), (), data, (data["conv_states"],)
    if name.startswith("sgem"):
        mod = importlib.import_module(root + "lora.test_" + name)
        case = mod.MAIN_CASES[0].values[0]
        args, y = mod.generate_case(case, torch.bfloat16)
        api = importlib.import_module("sgl_kernel_npu.operators.lora")
        suffix = (y, case["slices"]) if name.endswith("shrink") else (y,)
        return getattr(api, name), (*args, *suffix), {}, (y,)
    if name == "add_example":
        api = importlib.import_module("sgl_kernel_npu.operators.examples")
        args = (torch.ones(1024, device="npu"), torch.ones(1024, device="npu"))
        return getattr(api, name), args, {}, ()
    if name == "apply_token_bitmask":
        mod = importlib.import_module(root + "sampler.test_" + name)
        logits, bits, _ = mod.generate_test_data((16, 129280), torch.float32, "random")
        api = importlib.import_module("sgl_kernel_npu.operators.sampler")
        return getattr(api, name), (logits, bits), {}, (logits,)
    if name == "sparse_attn_sharedkv":
        mod = importlib.import_module(root + "attention.test_" + name)
        data = mod.generate_data(torch.bfloat16)
        api = importlib.import_module("sgl_kernel_npu.operators.attention")
        return getattr(api, name), (), data, ()
    if name == "chunk_kda_fwd":
        mod = importlib.import_module(root + "attention.kda.test_" + name)
        data = mod.generate_data((1, 256, 8, 8, 128, 128, 64), outputs=False, device="npu")
        api = importlib.import_module("sgl_kernel_npu.operators.attention.kda")
        return getattr(api, name), (), data, ()
    if name == "laser_attention":
        mod = importlib.import_module(root + "attention.test_" + name)
        data = mod.generate_test_data(1, 4, 4, 256, 128, torch.float16)
        api = importlib.import_module("sgl_kernel_npu.operators.attention")
        return api.la, (), data, ()
    if name == "ada_block_sparse_attention":
        mod = importlib.import_module(root + "attention.test_" + name)
        data = mod.generate_test_data(*mod.CASES[0])
        api = importlib.import_module("sgl_kernel_npu.operators.attention")
        return getattr(api, name), (), data, ()
    if name == "sparse_block_estimate":
        mod = importlib.import_module(root + "attention.test_" + name)
        q, k, data = mod.generate_test_data(
            (1, 4, 4, 4096, 64), torch.float16, "BNSD", "scaled"
        )
        api = importlib.import_module("sgl_kernel_npu.operators.attention")
        return getattr(api, name), (q.npu(), k.npu()), data, ()
    if name == "mlp_lightning_indexer":
        mod = importlib.import_module(root + "indexer.test_" + name)
        data = mod.generate_data(dict(q_lens=[2, 3], k_lens=[32, 64]), torch.bfloat16)
        data = {k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        api = importlib.import_module("sgl_kernel_npu.operators.indexer")
        return getattr(api, name), (), data, ()
    if name == "mla_preprocess":
        torch.npu.config.allow_internal_format = True
        mod = importlib.import_module(root + "attention.test_" + name)
        data, _, _ = mod.generate_data(1, 32, 7168, torch.bfloat16, "krope_ctkv")
        api = importlib.import_module("sgl_kernel_npu.operators.attention")
        return getattr(api, name), (), data, tuple(data[k] for k in mod._MUTABLE)
    if name == "sparse_attention_score":
        mod = importlib.import_module(
            root + "attention.test_flash_decode_bnsd_with_gqa_share_sparse"
        )
        data = mod.build_inputs(seq_lens=(16384,), page="table")
        data = {k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        api = importlib.import_module("sgl_kernel_npu.operators.attention")
        return api.flash_decode_bnsd_with_gqa_share_sparse, (), data, ()
    raise ValueError(name)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in _tensors(item)]
    return []


def _defined_outputs(operator, result):
    if operator == "laser_attention":
        # Its legacy softmax statistics port is allocated but never written.
        result = result[1]
    return _tensors(result)


def _check_replay(graph, output, mutable, expected, expected_state):
    # Overwrite captured results to prove replay still executes after cleanup.
    for value in [*_tensors(output), *mutable]:
        value.zero_()
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    actual = _tensors(output)
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value.cpu(), reference, atol=0, rtol=0)
    for value, reference in zip(mutable, expected_state):
        torch.testing.assert_close(value.cpu(), reference, atol=0, rtol=0)


# Include zero-workspace operators as controls: record_stream during capture
# alone does not imply that a large, per-invocation buffer is retained.
GRAPH_CASES = [
    "add_example",
    "apply_token_bitmask",
    "recurrent_gated_delta_rule",
    "causal_conv1d",
    "sgemmc_expand",
    "sgemmc_shrink",
    "sgemmv_expand",
    "sgemmv_shrink",
    "sparse_attn_sharedkv",
    "chunk_gated_delta_rule",
    "chunk_kda_fwd",
    "laser_attention",
    "ada_block_sparse_attention",
    "sparse_block_estimate",
    "mlp_lightning_indexer",
    "mla_preprocess",
    "sparse_attention_score",
]


@pytest.mark.extra_checks
@pytest.mark.parametrize("operator", GRAPH_CASES)
@pytest.mark.parametrize("auto_dispatch_capture", [False, True])
def test_aclnn_graph_workspace_reuse(operator, auto_dispatch_capture):
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware
    from tests.python.utils.common import get_test_impls, require_test_impl

    if get_hardware().generation != Generation.ARCH22:
        pytest.skip("These regression fixtures target arch22")
    if get_test_impls()[0] not in ("auto", "ascendc"):
        pytest.skip("Capture lifetime checks target the native ACLNN adapters")
    op, args, kwargs, mutable = _make_graph_case(operator)
    native_impl = "aicpu" if operator == "add_example" else "ascendc"
    require_test_impl(op, native_impl)
    initial = [value.clone() for value in mutable]

    def reset():
        for value, original in zip(mutable, initial):
            value.copy_(original)

    def call():
        return op(*args, **kwargs, impl=native_impl)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        reset()
        eager = call()
    stream.synchronize()
    expected = [value.cpu() for value in _defined_outputs(operator, eager)]
    expected_state = [value.cpu() for value in mutable]
    del eager
    reservations = {}
    for calls in (1, 32):
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        before = torch.npu.memory_reserved()
        graph = torch.npu.NPUGraph()
        output = None
        try:
            with torch.npu.graph(
                graph, stream=stream, auto_dispatch_capture=auto_dispatch_capture
            ):
                for _ in range(calls):
                    reset()
                    output = call()
            stream.synchronize()
            reservations[calls] = torch.npu.memory_reserved() - before
            # Match service startup, which cleans the allocator before replay.
            torch.npu.empty_cache()
            _check_replay(
                graph, _defined_outputs(operator, output), mutable, expected, expected_state
            )
        finally:
            # Do not retain graph outputs in Python after deleting the graph.
            del output, graph
            gc.collect()
            torch.npu.synchronize()
            torch.npu.empty_cache()
    assert reservations[32] <= reservations[1] + 64 * 1024**2, (
        f"{operator} retained per-call graph temporaries: "
        f"one call={reservations[1]}, 32 calls={reservations[32]} bytes"
    )
