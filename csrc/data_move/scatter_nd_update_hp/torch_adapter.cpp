// SPDX-License-Identifier: Apache-2.0
#include "aclnn_runtime.h"
#include "scatter_nd_update_hp.h"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

namespace {

class HostMemoryPool
{
    using InitFn = int (*)(void *, bool);
    using VoidFn = void (*)(void *, bool);

public:
    HostMemoryPool()
    {
        // Resolve cleanup callbacks before activating the thread-local pool.
        (void)Release();
        (void)UnInit();
        TORCH_CHECK(Init()(nullptr, false) == 0, "CANN host memory pool initialization failed");
    }
    ~HostMemoryPool()
    {
        Release()(nullptr, false);
        UnInit()(nullptr, false);
    }
    HostMemoryPool(const HostMemoryPool &) = delete;
    HostMemoryPool &operator=(const HostMemoryPool &) = delete;

private:
    static InitFn Init()
    {
        static auto fn = reinterpret_cast<InitFn>(sgl_cann::op_api_symbol_with_fallback("InitHugeMemThreadLocal"));
        return fn;
    }
    static VoidFn Release()
    {
        static auto fn = reinterpret_cast<VoidFn>(sgl_cann::op_api_symbol_with_fallback("ReleaseHugeMem"));
        return fn;
    }
    static VoidFn UnInit()
    {
        static auto fn = reinterpret_cast<VoidFn>(sgl_cann::op_api_symbol_with_fallback("UnInitHugeMemThreadLocal"));
        return fn;
    }
};

// Unlike the general descriptor helper, these handles need no shared ownership:
// creation, launch and destruction all happen inside the same handler.
auto Describe(const at::Tensor &tensor)
{
    aclDataType dtype;
    switch (tensor.scalar_type()) {
        case at::kChar:
            dtype = ACL_INT8;
            break;
        case at::kHalf:
            dtype = ACL_FLOAT16;
            break;
        case at::kInt:
            dtype = ACL_INT32;
            break;
        case at::kLong:
            dtype = ACL_INT64;
            break;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", tensor.scalar_type());
    }
    auto *descriptor = aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, tensor.strides().data(), 0,
                                       ACL_FORMAT_ND, tensor.sizes().data(), tensor.dim(), tensor.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>(descriptor, &aclDestroyTensor);
}

at::Tensor &Scatter(at::Tensor &cache, const at::Tensor &indices, const at::Tensor &updates)
{
    TORCH_CHECK(cache.device().type() == c10::DeviceType::PrivateUse1, "cache must be on NPU");
    TORCH_CHECK(indices.device() == cache.device() && updates.device() == cache.device(),
                "All tensors must be on the same NPU");
    TORCH_CHECK(cache.is_contiguous() && indices.is_contiguous() && updates.is_contiguous(),
                "Only contiguous tensors are supported");
    for (const auto &tensor : {cache, indices, updates}) {
        auto format = at_npu::native::get_npu_format(tensor);
        // Views of 4D/5D KV allocations can retain NCHW/NCDHW storage tags.
        // These base layouts are linear; contiguous views can be described as ND.
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW || format == ACL_FORMAT_NCDHW,
                    "Only ND/NCHW/NCDHW base storage formats are supported; got format ", format);
    }
    TORCH_CHECK(cache.scalar_type() == at::kChar || cache.scalar_type() == at::kHalf, "cache must be INT8 or FP16");
    TORCH_CHECK(cache.scalar_type() == updates.scalar_type(), "cache and updates must have the same dtype");
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "indices must be INT32 or INT64");
    TORCH_CHECK(cache.dim() == 2 || cache.dim() == 3,
                "cache must have shape [capacity, width] or [capacity, 1, width]");
    TORCH_CHECK(cache.dim() != 3 || cache.size(1) == 1, "3D cache must have a singleton middle dimension");
    TORCH_CHECK(indices.dim() == 2 && indices.size(1) == 1, "indices must have shape [tokens, 1]");
    TORCH_CHECK(updates.dim() == cache.dim() && updates.size(0) == indices.size(0) &&
                    updates.sizes().slice(1) == cache.sizes().slice(1),
                "updates must match cache trailing dimensions and token count");
    if (updates.numel() == 0) {
        return cache;
    }
    const c10::DeviceGuard guard(cache.device());
    auto stream = c10_npu::getCurrentNPUStream();
    auto acl_stream = stream.stream(false);
    // Capture protects storage through host queueing; recordStream also protects
    // asynchronous device use when inputs were allocated on another stream.
    for (const auto &tensor : {cache, indices, updates}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    auto launch = [cache, indices, updates, acl_stream]() -> int {
        // Descriptors must be destroyed before their pooled host memory is recycled.
        HostMemoryPool pool;
        auto self = Describe(cache);
        auto index = Describe(indices);
        auto update = Describe(updates);
        uint64_t workspace_size = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnSglScatterNdUpdateHpGetWorkspaceSize(self.get(), index.get(), update.get(), &workspace_size,
                                                                &executor);
        TORCH_CHECK(status == ACL_SUCCESS, "Scatter HP prepare failed: ", aclGetRecentErrMsg());
        std::unique_ptr<void, decltype(&c10_npu::NPUCachingAllocator::raw_delete)> workspace(
            nullptr, &c10_npu::NPUCachingAllocator::raw_delete);
        if (workspace_size != 0) {
            workspace.reset(c10_npu::NPUCachingAllocator::raw_alloc_with_stream(workspace_size, acl_stream));
        }
        status = aclnnSglScatterNdUpdateHp(workspace.get(), workspace_size, executor, acl_stream);
        TORCH_CHECK(status == ACL_SUCCESS, "Scatter HP launch failed: ", aclGetRecentErrMsg());
        return status;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglScatterNdUpdateHp", launch);
    return cache;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("scatter_nd_update_hp(Tensor(a!) cache, Tensor indices, Tensor updates) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("scatter_nd_update_hp", &Scatter);
}
