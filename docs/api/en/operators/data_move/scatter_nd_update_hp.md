# scatter_nd_update_hp

Write unique KV cache rows in place with CANN's installed ScatterNdUpdate
high-performance implementation. This operator uses only the new architecture.

## Signature

```python
from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

scatter_nd_update_hp(cache, indices, updates, *, impl="auto")
```

| Argument | Shape | dtype | Contract |
| --- | --- | --- | --- |
| `cache` | `[C,W]` / `[C,1,W]` | INT8 / FP16 | Modified and returned |
| `indices` | `[T,1]` | INT32 / INT64 | Unique IDs in `[0,C)` |
| `updates` | `[T,...]` | Same as cache | No overlap with cache |

`C` is cache capacity, `W` is row width and `T` is the number of updates.
The trailing dimensions of updates must match cache.

All tensors must be contiguous ND tensors on the same NPU. Contiguous views with
a storage offset are supported. Indices must not overlap cache storage either.
Empty updates return cache immediately. Index values are not checked through a
host synchronization; duplicate and out-of-range indices are outside the contract.

`auto` selects `cann_ops_nn` on arch22; `torch_native` explicitly selects the
PyTorch reference. arch35 is unsupported. The installed CANN must provide a
`high_performance` ScatterNdUpdate kernel for the chip and dtypes being used.
Prior A3 measurements used CANN 9.2.1; successful compilation on an older SDK
does not demonstrate availability of its device implementation.

## Implementation

- Registered as a new-architecture `HOST` operator, following LayerNorm.
  Reuses installed CANN binaries without building an OpDef, tiling or device kernel.
- A local two-phase ACLNN bridge requests `IMPL_MODE_HIGH_PERFORMANCE` through
  the L0 launcher. No alternative backend is selected on failure.
- Reuses existing symbol resolution and `OpCommand`. Descriptor creation,
  preparation, allocation and launch run in the same queued handler.
- Host memory pooling is local to this operator. Descriptors are destroyed before
  pooled memory is recycled. Workspace uses the NPU caching allocator directly.
- Tensor captures and stream recording protect queued and asynchronous device use.
- The CANN-only translation unit matches the installed nnopbase C++ ABI and
  connects to Torch through C functions. Other HOST operators are unaffected.

## Build and test

After sourcing the target CANN environment, run from the repository root:

```bash
python setup.py --arch-version 2201 --ops scatter_nd_update_hp \
    --without-third-party -j 8
pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python -m pytest \
    tests/python/operators/data_move/test_scatter_nd_update_hp.py \
    -v --enable-perf=false
```

Use `--ops scatter_nd_update_hp,layer_norm` to build both HOST operators.
Add `--make-clean` when an existing build has a different operator selection.

The native entry is `torch.ops.npu.scatter_nd_update_hp(cache, indices, updates)`.
Tests cover 16384 updates into 766080 cache rows: INT8 widths 656/128 and FP16
width 1, each with contiguous and random row IDs. Full-cache comparisons include
unwritten rows. Boundary, validation and cross-stream temporary-input tests are
also included. Enable profiling with `--enable-perf=true`; use `--prof-dir` and
`--csv-path` to select result locations.

CPU-only selection and reference checks:

```bash
python -m pytest tests/csrc/test_scatter_nd_update_hp_build.py -q
```
