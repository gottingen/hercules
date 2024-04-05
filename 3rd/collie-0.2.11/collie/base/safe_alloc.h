// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//


#ifndef COLLIE_BASE_SAFE_ALLOC_H_
#define COLLIE_BASE_SAFE_ALLOC_H_

#include <collie/base/macros.h>
#include <collie/base/debug_assert.h>
#include <memory>

#ifndef COLLIE_SAFE_ENABLE_ASSERTIONS
#define COLLIE_SAFE_ALLOC_ASSERTIONS 0
#endif

namespace collie {

    struct alloc_assert_handler : collie::debug_assert::set_level<COLLIE_SAFE_ALLOC_ASSERTIONS>,
                            collie::debug_assert::default_handler {
    };

    COLLIE_ATTRIBUTE_RETURNS_NONNULL inline void *safe_malloc(size_t Sz) {
        void *Result = std::malloc(Sz);
        if (Result == nullptr) {
            // It is implementation-defined whether allocation occurs if the space
            // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
            // non-zero, if the space requested was zero.
            if (Sz == 0) {
                return safe_malloc(1);
            }
            DEBUG_ASSERT(false,  alloc_assert_handler{}, "Allocation failed");
        }
        return Result;
    }

    COLLIE_ATTRIBUTE_RETURNS_NONNULL inline void *safe_calloc(size_t Count,
                                                            size_t Sz) {
        void *Result = std::calloc(Count, Sz);
        if (Result == nullptr) {
            // It is implementation-defined whether allocation occurs if the space
            // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
            // non-zero, if the space requested was zero.
            if (Count == 0 || Sz == 0)
                return safe_malloc(1);
            DEBUG_ASSERT(false,  alloc_assert_handler{}, "Allocation failed");
        }
        return Result;
    }

    COLLIE_ATTRIBUTE_RETURNS_NONNULL inline void *safe_realloc(void *Ptr, size_t Sz) {
        void *Result = std::realloc(Ptr, Sz);
        if (Result == nullptr) {
            // It is implementation-defined whether allocation occurs if the space
            // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
            // non-zero, if the space requested was zero.
            if (Sz == 0)
                return safe_malloc(1);
            DEBUG_ASSERT(false,  alloc_assert_handler{}, "Allocation failed");
        }
        return Result;
    }

}  // namespace collie

#endif  // COLLIE_BASE_SAFE_ALLOC_H_
