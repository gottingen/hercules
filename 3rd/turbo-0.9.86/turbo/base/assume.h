// Copyright 2023 The Turbo Authors.
//
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

#ifndef TURBO_BASE_ASSUME_H_
#define TURBO_BASE_ASSUME_H_

#include <cstdlib>
#include "turbo/platform/port.h"

namespace turbo {

    /**
     * @ingroup turbo_base
     * @brief This function is used to make assumptions about the condition.
     *        It uses different compiler intrinsics based on the compiler being used.
     * @details The `assume_unreachable` function is used to indicate to the compiler
     *          that a certain code path is unreachable. This function uses different
     *          compiler intrinsics based on the compiler being used. It is used when
     *          the condition is always false. This can be useful for optimization
     *          purposes, as it allows the compiler to make assumptions about the code
     *          that can potentially lead to more efficient code generation.
     * @param cond The condition that is assumed to be true.
     */
    TURBO_FORCE_INLINE void assume(bool cond) {
#if defined(__clang__) // Must go first because Clang also defines __GNUC__.
        __builtin_assume(cond);
#elif defined(__GNUC__)
        if (!cond) {
            __builtin_unreachable();
        }
#elif defined(_MSC_VER)
        __assume(cond);
#else
        // Do nothing.
#endif
    }

    /**
     * @ingroup turbo_base
     * @brief This function is used to make assumptions about the condition.
     *        It uses different compiler intrinsics based on the compiler being used.
     *        @see assume(bool cond), this function is used when the condition is always false.
     */
    TURBO_FORCE_INLINE void assume_unreachable() {
        assume(false);
        // Do a bit more to get the compiler to understand
        // that this function really will never return.
#if defined(__GNUC__)
        __builtin_unreachable();
#elif defined(_MSC_VER)
        __assume(0);
#else
        // Well, it's better than nothing.
        std::abort();
#endif
    }

} // namespace turbo

#endif // TURBO_BASE_ASSUME_H_
