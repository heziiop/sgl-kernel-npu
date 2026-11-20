// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <aclnn/acl_meta.h>

// Only C types cross between the CANN and Torch C++ ABIs.
extern "C" aclnnStatus aclnnSglScatterNdUpdateHpGetWorkspaceSize(aclTensor *self, const aclTensor *indices,
                                                                 const aclTensor *updates, uint64_t *workspaceSize,
                                                                 aclOpExecutor **executor);
extern "C" aclnnStatus aclnnSglScatterNdUpdateHp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 aclrtStream stream);
