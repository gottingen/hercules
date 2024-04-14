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

#ifndef TURBO_BASE_PROCESSOR_H_
#define TURBO_BASE_PROCESSOR_H_

#include "turbo/platform/port.h"

namespace turbo {

    void cpu_relax();

    void barrier();

    // Pause instruction to prevent excess processor bus usage, only works in GCC

    inline void cpu_relax() {
#if defined(TURBO_PROCESSOR_ARM)
        asm volatile("yield\n": : :"memory");
#elif defined(TURBO_PROCESSOR_LOONGARCH64)
        asm volatile("nop\n": : :"memory");
#else
        asm volatile("pause\n": : :"memory");
#endif
    }

    // Compile read-write barrier
    inline void barrier() {
        asm volatile("": : :"memory");
    }

# define TURBO_LOOP_WHEN(expr, num_spins)                               \
    do {                                                                \
        /*sched_yield may change errno*/                                \
        const int saved_errno = errno;                                  \
        for (int cnt = 0, saved_nspin = (num_spins); (expr); ++cnt) {   \
            if (cnt < saved_nspin) {                                    \
                cpu_relax();                                            \
            } else {                                                    \
                sched_yield();                                          \
            }                                                           \
        }                                                               \
        errno = saved_errno;                                            \
    } while (0)

}  // namespace turbo

#endif // TURBO_BASE_PROCESSOR_H_
