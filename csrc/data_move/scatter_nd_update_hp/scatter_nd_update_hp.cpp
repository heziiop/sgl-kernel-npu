// SPDX-License-Identifier: Apache-2.0
#include "scatter_nd_update_hp.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace {
OP_TYPE_REGISTER(ScatterNdUpdate);

aclnnStatus AddScatterHp(aclTensor *self, const aclTensor *indices, const aclTensor *updates, aclOpExecutor *executor)
{
    L0_DFX(AddScatterHp, self, indices, updates);
    return ADD_TO_LAUNCHER_LIST_AICORE(ScatterNdUpdate, OP_INPUT(self, indices, updates), OP_OUTPUT(self),
                                       OP_ATTR(false), OP_OPTION(op::OpImplMode::IMPL_MODE_HIGH_PERFORMANCE));
}
}  // namespace

extern "C" aclnnStatus aclnnSglScatterNdUpdateHpGetWorkspaceSize(aclTensor *self, const aclTensor *indices,
                                                                 const aclTensor *updates, uint64_t *workspaceSize,
                                                                 aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSglScatterNdUpdateHp, DFX_IN(self, indices, updates), DFX_OUT(self));
    auto owner = CREATE_EXECUTOR();
    if (owner.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto status = AddScatterHp(self, indices, updates, owner.get());
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSglScatterNdUpdateHp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSglScatterNdUpdateHp);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
