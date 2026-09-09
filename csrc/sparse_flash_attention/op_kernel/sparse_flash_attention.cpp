/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention.cpp
 * \brief
 */

// The top-level AscendC build does not define the event identifiers that the
// original OPP generator injected for the cube implementation.
#ifndef EVENT_ID0
#define EVENT_ID0 0
#endif
#ifndef EVENT_ID1
#define EVENT_ID1 1
#endif
#ifndef EVENT_ID2
#define EVENT_ID2 2
#endif
#ifndef EVENT_ID3
#define EVENT_ID3 3
#endif
#ifndef EVENT_ID4
#define EVENT_ID4 4
#endif
#ifndef EVENT_ID5
#define EVENT_ID5 5
#endif
#ifndef EVENT_ID6
#define EVENT_ID6 6
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 7
#endif

#include "kernel_operator.h"
#include "sparse_flash_attention_tiling_data.h"
#include "sparse_flash_attention_template_tiling_key.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/sparse_flash_attention_kernel_mla.h"
#else
#include "arch22/sparse_flash_attention_kernel_mla.h"
#endif

using namespace AscendC;

#if (__CCE_AICORE__ == 310)
#if defined(__DAV_C310_CUBE__)
#define SFA_OP_IMPL(templateClass, tilingdataClass, ...)                                                            \
    do {                                                                                                            \
        using CubeBlockType =                                                                                       \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::SFAMatmulService<__VA_ARGS__>,           \
                                      BaseApi::SFAMatmulServiceDummy<__VA_ARGS__>>::type;                           \
        using VecBlockType =                                                                                        \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::SFAVectorServiceDummy<__VA_ARGS__>,      \
                                      BaseApi::SFAVectorService<__VA_ARGS__>>::type;                                \
        templateClass<CubeBlockType, VecBlockType> op;                                                              \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV, blocktable, queryRope, \
                keyRope, attentionOut, softmaxMax, softmaxSum, user, nullptr, tiling, &tPipe);                      \
        op.Process();                                                                                               \
    } while (0)
#else
#define SFA_OP_IMPL(templateClass, tilingdataClass, ...)                                                            \
    do {                                                                                                            \
        using CubeBlockType =                                                                                       \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::SFAMatmulService<__VA_ARGS__>,           \
                                      BaseApi::SFAMatmulServiceDummy<__VA_ARGS__>>::type;                           \
        using VecBlockType =                                                                                        \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::SFAVectorServiceDummy<__VA_ARGS__>,      \
                                      BaseApi::SFAVectorService<__VA_ARGS__>>::type;                                \
        templateClass<CubeBlockType, VecBlockType> op;                                                              \
        tilingdataClass tiling_data_in;                                                                             \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                                       \
        const tilingdataClass *__restrict tilingData = &tiling_data_in;                                             \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV, blocktable, queryRope, \
                keyRope, attentionOut, softmaxMax, softmaxSum, user, tilingData, tiling, &tPipe);                   \
        op.Process();                                                                                               \
    } while (0)
#endif
#else
#define SFA_OP_IMPL(templateClass, tilingdataClass, ...)                                                            \
    do {                                                                                                            \
        templateClass<SFAType<__VA_ARGS__>> op;                                                                     \
        const __gm__ tilingdataClass *__restrict tiling_data =                                                      \
            reinterpret_cast<const __gm__ tilingdataClass *>(tiling);                                               \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV, blocktable, queryRope, \
                keyRope, attentionOut, softmaxMax, softmaxSum, user, tiling_data, tiling, &tPipe);                  \
        op.Process();                                                                                               \
    } while (0)
#endif

template <int flash_decode, int page_attention, int layout_t, int kv_layout_t, int template_mode, int split_g,
          bool is_bf16>
__aicore__ inline void RunSparseFlashAttention(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices,
                                               GM_ADDR blocktable, GM_ADDR actualSeqLengthsQuery,
                                               GM_ADDR actualSeqLengthsKV, GM_ADDR queryRope, GM_ADDR keyRope,
                                               GM_ADDR attentionOut, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

#if (__CCE_AICORE__ == 310)
    if constexpr (is_bf16) {
        SFA_OP_IMPL(BaseApi::SparseFlashAttentionKernelMla, SparseFlashAttentionTilingDataMla, bfloat16_t, bfloat16_t,
                    float, bfloat16_t, flash_decode, page_attention, static_cast<SFA_LAYOUT>(layout_t),
                    static_cast<SFA_LAYOUT>(kv_layout_t), static_cast<SFATemplateMode>(template_mode), split_g);
    } else {
        SFA_OP_IMPL(BaseApi::SparseFlashAttentionKernelMla, SparseFlashAttentionTilingDataMla, half, half, float, half,
                    flash_decode, page_attention, static_cast<SFA_LAYOUT>(layout_t),
                    static_cast<SFA_LAYOUT>(kv_layout_t), static_cast<SFATemplateMode>(template_mode), split_g);
    }
#else
    if constexpr (!is_bf16) {
        SFA_OP_IMPL(SparseFlashAttentionMla, SparseFlashAttentionTilingDataMla, half, half, half, flash_decode,
                    static_cast<SFA_LAYOUT>(layout_t), static_cast<SFA_LAYOUT>(kv_layout_t), template_mode);
    } else {  // bf16
        SFA_OP_IMPL(SparseFlashAttentionMla, SparseFlashAttentionTilingDataMla, bfloat16_t, bfloat16_t, bfloat16_t,
                    flash_decode, static_cast<SFA_LAYOUT>(layout_t), static_cast<SFA_LAYOUT>(kv_layout_t),
                    template_mode);
    }
#endif
}

#define DEFINE_SFA_KERNEL(kernel_name, flash_decode, page_attention, layout_t, kv_layout_t, template_mode, split_g,    \
                          is_bf16)                                                                                     \
    extern "C" __global__ __aicore__ void kernel_name(                                                                 \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices, GM_ADDR blocktable,                          \
        GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKV, GM_ADDR queryRope, GM_ADDR keyRope,                 \
        GM_ADDR attentionOut, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace, GM_ADDR tiling)               \
    {                                                                                                                  \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                                             \
        RunSparseFlashAttention<flash_decode, page_attention, layout_t, kv_layout_t, template_mode, split_g, is_bf16>( \
            query, key, value, sparseIndices, blocktable, actualSeqLengthsQuery, actualSeqLengthsKV, queryRope,        \
            keyRope, attentionOut, softmaxMax, softmaxSum, workspace, tiling);                                         \
    }

#define DEFINE_SFA_VARIANTS(prefix, page_attention, layout_t, kv_layout_t)                          \
    DEFINE_SFA_KERNEL(prefix##_c, 0, page_attention, layout_t, kv_layout_t, 0, 0, false)            \
    DEFINE_SFA_KERNEL(prefix##_c_splitg, 0, page_attention, layout_t, kv_layout_t, 0, 1, false)     \
    DEFINE_SFA_KERNEL(prefix##_v, 0, page_attention, layout_t, kv_layout_t, 1, 0, false)            \
    DEFINE_SFA_KERNEL(prefix##_v_splitg, 0, page_attention, layout_t, kv_layout_t, 1, 1, false)     \
    DEFINE_SFA_KERNEL(prefix##_bf16_c, 0, page_attention, layout_t, kv_layout_t, 0, 0, true)        \
    DEFINE_SFA_KERNEL(prefix##_bf16_c_splitg, 0, page_attention, layout_t, kv_layout_t, 0, 1, true) \
    DEFINE_SFA_KERNEL(prefix##_bf16_v, 0, page_attention, layout_t, kv_layout_t, 1, 0, true)        \
    DEFINE_SFA_KERNEL(prefix##_bf16_v_splitg, 0, page_attention, layout_t, kv_layout_t, 1, 1, true)

DEFINE_SFA_VARIANTS(sgl_sfa_bsnd, 0, SFA_LAYOUT_BSND, SFA_LAYOUT_BSND)
DEFINE_SFA_VARIANTS(sgl_sfa_pa_bsnd, 1, SFA_LAYOUT_BSND, SFA_LAYOUT_PA_BSND)
DEFINE_SFA_VARIANTS(sgl_sfa_tnd, 0, SFA_LAYOUT_TND, SFA_LAYOUT_TND)
DEFINE_SFA_VARIANTS(sgl_sfa_pa_tnd, 1, SFA_LAYOUT_TND, SFA_LAYOUT_PA_BSND)

#undef DEFINE_SFA_VARIANTS
#undef DEFINE_SFA_KERNEL
