# scatter_nd_update_hp

使用 CANN 内置 ScatterNdUpdate 的高性能实现，原地写入 KV cache 的指定行。
仅接入新架构，原有构建和绑定不变。

## 签名与约束

```python
from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

scatter_nd_update_hp(cache, indices, updates, *, impl="auto")
```

| 参数 | 形状 | dtype | 约束 |
| --- | --- | --- | --- |
| `cache` | `[C,W]` / `[C,1,W]` | INT8 / FP16 | 原地修改并返回同一存储 |
| `indices` | `[T,1]` | INT32 / INT64 | 行号唯一，且在 `[0,C)` 内 |
| `updates` | `[T,...]` | 与 cache 相同 | 不得与 cache 存储重叠 |

`C` 为 cache 容量，`W` 为行宽，`T` 为 updates 数量。
updates 的后续维度必须与 cache 一致。

三个张量必须在同一 NPU，连续且为 ND 存储格式；支持带 storage offset 的连续 view。
`indices` 也不得与 cache 存储重叠。空 updates 直接返回 cache。
不通过 Host 同步检查索引值；重复和越界索引不在接口契约内。

`auto` 在 arch22 上选择 `cann_ops_nn`，`torch_native` 是显式选择的 PyTorch 参考实现。
arch35 不注册此算子。需要目标 CANN 为对应芯片、dtype 和索引类型提供
`ScatterNdUpdate` 的 `high_performance` 实现；此前 A3 测试使用 CANN 9.2.1，
不能仅凭本封装能在旧 SDK 上编译，就推断旧 SDK 也有对应设备实现。

## 实现

- 按 LayerNorm 的新架构注册为 `EXECUTION_UNIT HOST`。复用 CANN 已安装的 kernel，
  不编译、打包 OpDef、tiling 或设备 kernel。
- 本地两阶段 ACLNN 入口通过 L0 launcher 指定 `IMPL_MODE_HIGH_PERFORMANCE`。
  不主动回退到其他实现；实际执行的 kernel 可从 profiler 确认。
- 复用 `aclnn_runtime.h` 的符号解析和 torch_npu `OpCommand` 队列。
  描述符创建、GetWorkspaceSize、workspace 申请及执行均在同一个 handler 内。
- 对象池仅在本算子内部接入，描述符先销毁，再回收 Host 临时内存。
  workspace 使用 `NPUCachingAllocator::raw_alloc_with_stream` / `raw_delete`。
- handler 持有三个输入 Tensor，并记录使用 stream，保护排队及跨 stream 的设备访问。
- CANN bridge 按本机 `libnnopbase.so` 的 C++ ABI 编译，经 C 接口连接 Torch adapter；
  不改变其他 HOST 算子的编译选项，也不增加统一运行时框架。

## 构建与使用

在目标 CANN 环境下，从仓库根目录构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python setup.py --arch-version 2201 --ops scatter_nd_update_hp \
    --without-third-party -j 8
pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
```

也可与 LayerNorm 等其他新架构算子一起构建：
`--ops scatter_nd_update_hp,layer_norm`。
已有不同选择的 build 目录时，在构建命令中加 `--make-clean`。

```python
import torch
import torch_npu
import sgl_kernel_npu

from sgl_kernel_npu.operators.data_move import scatter_nd_update_hp

cache = torch.zeros((766080, 1, 656), dtype=torch.int8, device="npu")
indices = torch.arange(16384, dtype=torch.int64, device="npu").reshape(-1, 1)
updates = torch.ones((16384, 1, 656), dtype=torch.int8, device="npu")
scatter_nd_update_hp(cache, indices, updates)

# 也可以直接通过 Torch 注册入口调用。
torch.ops.npu.scatter_nd_update_hp(cache, indices, updates)
```

## 测试

```bash
# 16K 主 KV / indexer K / scale，连续和随机写入；同时覆盖边界及输入校验。
python -m pytest \
    tests/python/operators/data_move/test_scatter_nd_update_hp.py \
    -v --enable-perf=false

# 使用仓库的公共 profiling 和结果汇总机制。
python -m pytest \
    tests/python/operators/data_move/test_scatter_nd_update_hp.py -v \
    -k 'main-kv or indexer-k or indexer-scale' \
    --enable-perf=true --prof-dir ./profilings/scatter_hp --csv-path ./scatter_hp.csv

# 不需要 NPU 的构建选择与参考实现检查。
python -m pytest tests/csrc/test_scatter_nd_update_hp_build.py -q
```

生产形状测试使用 cache 766080 行、16384 updates，INT8 宽度 656 / 128、FP16 宽度 1。
精度检查比较完整 cache，包含未写入区域，并验证索引、updates 不被修改。
另有 INT32 索引、2D、空输入、非对齐宽度，以及跨 stream 临时输入生命周期检查。
