"""In-place KV row writes using the installed CANN high-performance kernel."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "data_move.scatter_nd_update_hp",
    "auto",
    supported_generations={Generation.ARCH22},
)
@register_impl(
    "data_move.scatter_nd_update_hp",
    "cann_ops_nn",
    supported_generations={Generation.ARCH22},
)
def _cann_ops_nn_impl(cache, indices, updates):
    import torch

    return torch.ops.npu.scatter_nd_update_hp(cache, indices, updates)


@register_impl(
    "data_move.scatter_nd_update_hp",
    "torch_native",
    supported_generations={Generation.ARCH22},
)
def _torch_native_impl(cache, indices, updates):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.data_move.scatter_nd_update_hp import (
            scatter_nd_update_hp,
        )

        _torch_native_fn = scatter_nd_update_hp
    return _torch_native_fn(cache, indices, updates)


@register_operator("data_move.scatter_nd_update_hp")
def scatter_nd_update_hp(cache, indices, updates, *, impl="auto"):
    """Write ``updates`` to unique cache rows in place and return ``cache``.

    ``cache`` is contiguous ND INT8 or FP16, shaped [capacity, width] or
    [capacity, 1, width]. ``indices`` is contiguous INT32/INT64 [tokens, 1];
    each index must be unique and in [0, capacity). ``updates`` has the cache's
    dtype and trailing dimensions, with tokens rows. All tensors must be on
    the same NPU. Inputs and updates must not overlap cache storage.

    ``auto`` selects ``cann_ops_nn`` on arch22. This requires an installed
    CANN ScatterNdUpdate high_performance implementation; no device kernel is
    bundled and no fallback backend is selected. ``torch_native`` is an
    explicit reference implementation. Duplicate/out-of-range indices are
    outside the contract and are not checked by synchronizing to the host.
    """
    return get_impl_by_name("data_move.scatter_nd_update_hp", impl)(
        cache, indices, updates
    )
