#pragma once

#include <ATen/ATen.h>
#include <c10/util/SmallVector.h>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <utility>

namespace sgl_cann {

// Keep host submission and Tensor lifetime handling consistent for ACLNN bridges.
// The launch callable must retain every Tensor and descriptor used by the device.
class OpApiContext {
public:
    // Only list tensors newly allocated by this adapter on stream. The adapter
    // must access them only on stream, and their owners must outlive this context.
    // Caller-owned inputs, outputs and state must keep stream recording: capture
    // alone does not make their allocation stream safe to reuse their storage.
    explicit OpApiContext(c10_npu::NPUStream stream, std::initializer_list<at::Tensor> local_tensors = {})
        : stream_(stream), capturing_(c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::Active)
    {
        if (capturing_) {
            for (const auto &tensor : local_tensors) {
                if (tensor.defined()) capture_local_storage_.push_back(tensor.storage().unsafeGetStorageImpl());
            }
        }
    }

    void record_stream(const at::Tensor &tensor) const
    {
        // The allocator reuses blocks on their allocation stream. These local
        // tensors can be reused in stream order; recording them would instead
        // retain each invocation's storage until capture ends. Everything else
        // may have been allocated on another stream and needs lifetime tracking.
        if (capturing_ && tensor.defined() &&
            std::find(capture_local_storage_.begin(), capture_local_storage_.end(),
                      tensor.storage().unsafeGetStorageImpl()) != capture_local_storage_.end()) {
            return;
        }
        tensor.record_stream(stream_);
    }

    template <typename Launch>
    void run(const char *name, Launch &&launch) const
    {
        if (capturing_) {
            // stream() submits earlier host-queued work before this direct launch;
            // it does not wait for device completion. Destroy the callable's
            // Tensor captures on the capture thread so its graph pool can reuse
            // temporary storage between successive operators.
            TORCH_CHECK(stream_.stream() != nullptr, name, ": failed to get capture stream");
            launch();
        } else {
            at_npu::native::OpCommand::RunOpApi(name, std::forward<Launch>(launch));
        }
    }

private:
    c10_npu::NPUStream stream_;
    bool capturing_;
    c10::SmallVector<const c10::StorageImpl *, 4> capture_local_storage_;
};

// Resolve from this extension, never from a process-wide libcust_opapi name.
inline void *op_api_symbol(const char *name)
{
    static void *library = [] {
        Dl_info info{};
        TORCH_CHECK(dladdr(reinterpret_cast<void *>(&op_api_symbol), &info) && info.dli_fname,
                    "Cannot locate the CANN Torch extension");
        auto path = std::filesystem::canonical(info.dli_fname).parent_path().parent_path() /
                    "_cann/vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so";
        auto handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(handle, "Cannot load ", path.string(), ": ", dlerror());
        return handle;  // Keep the library loaded while queued calls can use it.
    }();
    auto symbol = dlsym(library, name);
    TORCH_CHECK(symbol, "Missing CANN symbol: ", name);
    return symbol;
}

inline std::shared_ptr<aclTensor> tensor_descriptor(const at::Tensor &tensor, aclDataType dtype)
{
    // The caller prepares contiguous work tensors. Describe their logical span,
    // including views with an offset, exactly as in the standalone ACLNN test.
    auto descriptor = aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, tensor.strides().data(), 0,
                                      ACL_FORMAT_ND, tensor.sizes().data(), tensor.dim(), tensor.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}

}  // namespace sgl_cann
