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
    """Write ``updates`` to selected cache rows in place and return ``cache``.

    ``cache`` is INT8, FP16 or BF16, shaped [capacity, width] or
    [capacity, 1, width]. ``indices`` is INT32/INT64 [tokens, 1]
    with row IDs in [0, capacity). ``updates`` has the cache's dtype and
    trailing dimensions, with tokens rows. All tensors must be on the same
    NPU. The CANN backend prepares contiguous ND tensors when a layout needs
    conversion and copies the result back into ``cache`` if needed.
    ``indices`` must not overlap cache storage.

    ``auto`` selects ``cann_ops_nn`` on arch22. This requires an installed
    CANN ScatterNdUpdate high_performance implementation; no device kernel is
    bundled and no fallback backend is selected. ``torch_native`` is an
    explicit reference implementation.

    The wrapper launches the kernel without checking index values. In the CANN
    high-performance path, updates to the same row may produce a nondeterministic
    result. Out-of-range indices, including -1, may cause writes outside the
    cache, corrupting other data or triggering an NPU error.
    """
    return get_impl_by_name("data_move.scatter_nd_update_hp", impl)(
        cache, indices, updates
    )
