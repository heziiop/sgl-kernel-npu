"""Unique KV row writes: full-cache accuracy, queued lifetimes and profiling."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

TOKENS = 16384
CACHE_ROWS = 766080


def generate_data(rows, tokens, width, dtype, pattern, rank=3, index_dtype=torch.int64):
    generator = torch.Generator(device="cpu").manual_seed(29)
    trailing = (1, width) if rank == 3 else (width,)
    cache = torch.full((rows, *trailing), 11, dtype=dtype, device="npu")
    if pattern == "random":
        slots = torch.randperm(rows, generator=generator, device="cpu")[:tokens]
    else:
        slots = torch.arange(rows - tokens, rows, device="cpu")
    indices = slots.to(index_dtype).reshape(-1, 1).npu()
    updates = (
        torch.randint(-64, 64, (tokens, *trailing), generator=generator, device="cpu")
        .to(dtype)
        .npu()
    )
    return cache, indices, updates


@pytest.mark.parametrize("pattern", ["contiguous", "random"])
@pytest.mark.parametrize(
    "width,dtype",
    [(656, torch.int8), (128, torch.int8), (1, torch.float16)],
    ids=["main-kv", "indexer-k", "indexer-scale"],
)
def test_scatter_nd_update_hp(width, dtype, pattern):
    from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

    impl, impl_ref = get_test_impls()
    accuracy_checked = not get_skip_acc_check()
    if accuracy_checked:
        require_distinct_impls(scatter_nd_update_hp, impl, impl_ref)
    else:
        require_test_impl(scatter_nd_update_hp, impl)
    cache, indices, updates = generate_data(CACHE_ROWS, TOKENS, width, dtype, pattern)
    saved_indices, saved_updates = indices.clone(), updates.clone()
    pointer = cache.data_ptr()
    if accuracy_checked:
        reference = cache.clone()
        scatter_nd_update_hp(reference, indices, updates, impl=impl_ref)
    for _ in range(3):
        returned = scatter_nd_update_hp(cache, indices, updates, impl=impl)
        assert returned.data_ptr() == pointer
    if accuracy_checked:
        # Full-cache comparison checks written rows and every untouched row.
        assert torch.equal(cache, reference)
    assert torch.equal(indices, saved_indices)
    assert torch.equal(updates, saved_updates)
    executor = make_executor(
        scatter_nd_update_hp,
        args=(cache, indices, updates),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens", [0, 1, 17])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_scatter_nd_update_hp_boundaries(rank, index_dtype, tokens, dtype):
    from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

    impl, _ = get_test_impls()
    require_test_impl(scatter_nd_update_hp, impl)
    cache, indices, updates = generate_data(
        41, tokens, 33, dtype, "random", rank, index_dtype
    )
    expected = cache.cpu()
    expected.index_copy_(0, indices.cpu().reshape(-1).long(), updates.cpu())
    result = scatter_nd_update_hp(cache, indices, updates, impl=impl)
    assert result.data_ptr() == cache.data_ptr()
    assert torch.equal(cache.cpu(), expected)


def test_scatter_nd_update_hp_temporary_inputs_on_another_stream():
    from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

    impl, _ = get_test_impls()
    require_test_impl(scatter_nd_update_hp, impl)
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(producer):
        cache, indices, updates = generate_data(4096, 1024, 128, torch.int8, "random")
        expected = cache.clone()
        expected.index_copy_(0, indices.reshape(-1), updates)
        temporary_indices, temporary_updates = indices.clone(), updates.clone()
    consumer.wait_stream(producer)
    with torch.npu.stream(consumer):
        for _ in range(20):
            scatter_nd_update_hp(cache, temporary_indices, temporary_updates, impl=impl)
    del temporary_indices, temporary_updates
    # Reuse allocations on the producing stream while the consumer is pending.
    with torch.npu.stream(producer):
        trash = [torch.full_like(updates, -99) for _ in range(32)]
    torch.npu.synchronize()
    assert torch.equal(cache, expected)
    assert len(trash) == 32


def test_scatter_nd_update_hp_storage_offsets():
    from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

    impl, _ = get_test_impls()
    require_test_impl(scatter_nd_update_hp, impl)
    storage = torch.full((44, 1, 33), 11, dtype=torch.int8, device="npu")
    cache = storage[2:-1]
    index_storage = torch.tensor([[0], [1], [7], [39], [0]], device="npu")
    indices = index_storage[1:-1]
    update_storage = torch.full((5, 1, 33), -17, dtype=torch.int8, device="npu")
    updates = update_storage[1:-1]
    expected = storage.cpu()
    expected.index_copy_(0, indices.cpu().flatten() + 2, updates.cpu())
    result = scatter_nd_update_hp(cache, indices, updates, impl=impl)
    assert result.data_ptr() == cache.data_ptr()
    assert torch.equal(storage.cpu(), expected)


@pytest.mark.parametrize("storage_format,rank", [(0, 4), (32, 5)])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16])
def test_scatter_nd_update_hp_base_format_views(storage_format, rank, dtype):
    import torch_npu
    from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

    impl, _ = get_test_impls()
    require_test_impl(scatter_nd_update_hp, impl)
    # Match SGLang's layer/page/slot allocation, then select a layer and flatten.
    shape = (2, 3, 8, 1, 33) if rank == 5 else (2, 3, 8, 33)
    storage = torch_npu.npu_format_cast(
        torch.full(shape, 11, dtype=dtype, device="npu"), storage_format
    )
    cache = storage[1].view(-1, 1, 33)
    assert torch_npu.get_npu_format(cache) == storage_format
    indices = torch.tensor([[1], [7], [23]], device="npu")
    updates = torch.full((3, 1, 33), -17, dtype=dtype, device="npu")
    expected = storage.cpu()
    expected[1].view(-1, 1, 33).index_copy_(0, indices.cpu().flatten(), updates.cpu())
    pointer = cache.data_ptr()
    result = scatter_nd_update_hp(cache, indices, updates, impl=impl)
    assert result.data_ptr() == pointer
    assert torch_npu.get_npu_format(cache) == storage_format
    assert torch.equal(storage.cpu(), expected)


@pytest.mark.parametrize("argument", ["cache", "updates"])
def test_scatter_nd_update_hp_internal_storage(argument):
    import sgl_kernel_npu  # noqa: F401 -- loads the selected native extension
    import torch_npu

    cache, indices, updates = generate_data(64, 16, 32, torch.float16, "random", 2)
    tensors = dict(cache=cache, indices=indices, updates=updates)
    tensors[argument] = torch_npu.npu_format_cast(tensors[argument], 29)
    assert torch_npu.get_npu_format(tensors[argument]) == 29
    pointer = tensors["cache"].data_ptr()
    saved_updates = updates.cpu()
    expected = cache.cpu()
    expected.index_copy_(0, indices.cpu().flatten(), saved_updates)
    result = torch.ops.npu.scatter_nd_update_hp(**tensors)
    assert result.data_ptr() == pointer
    assert torch_npu.get_npu_format(tensors[argument]) == 29
    assert torch.equal(tensors["cache"].cpu(), expected)
    assert torch.equal(tensors["updates"].cpu(), saved_updates)


@pytest.mark.parametrize("cache_axis", [0, 1])
def test_scatter_nd_update_hp_noncontiguous(cache_axis):
    from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

    impl, _ = get_test_impls()
    require_test_impl(scatter_nd_update_hp, impl)
    storage = torch.full((18, 12), 11, dtype=torch.float16, device="npu")
    cache_slice = (slice(1, None, 2), slice(None))
    if cache_axis == 1:
        cache_slice = (slice(None), slice(1, None, 2))
    cache = storage[cache_slice]
    indices = torch.tensor([[1, 0], [4, 0], [8, 0]], device="npu")[:, :1]
    updates = torch.arange(6 * cache.size(1), device="cpu").reshape(3, -1)
    updates = updates.to(torch.float16).npu()[:, ::2]
    assert all(not tensor.is_contiguous() for tensor in (cache, indices, updates))

    # Compare the backing tensor so gaps in the destination view are checked too.
    expected = storage.cpu()
    expected[cache_slice].index_copy_(0, indices.cpu().flatten(), updates.cpu())
    result = scatter_nd_update_hp(cache, indices, updates, impl=impl)
    assert result.data_ptr() == cache.data_ptr()
    assert torch.equal(storage.cpu(), expected)


@pytest.mark.parametrize(
    "invalid,match",
    [
        ("cache_dtype", "INT8, FP16 or BF16"),
        ("index_dtype", "INT32 or INT64"),
        ("indices_rank", "indices must have shape"),
        ("updates_shape", "updates must match"),
        ("cpu_indices", "same NPU"),
    ],
)
def test_scatter_nd_update_hp_rejects_invalid_inputs(invalid, match):
    import sgl_kernel_npu  # noqa: F401 -- loads the selected native extension
    import torch_npu  # noqa: F401 -- registers the NPU dispatch key

    cache, indices, updates = generate_data(41, 17, 33, torch.float16, "random")
    if invalid == "cache_dtype":
        cache, updates = cache.float(), updates.float()
    elif invalid == "index_dtype":
        indices = indices.float()
    elif invalid == "indices_rank":
        indices = indices.flatten()
    elif invalid == "updates_shape":
        updates = updates[..., :-1].contiguous()
    elif invalid == "cpu_indices":
        indices = indices.cpu()
    with pytest.raises(RuntimeError, match=match):
        torch.ops.npu.scatter_nd_update_hp(cache, indices, updates)
