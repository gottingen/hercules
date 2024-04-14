// Copyright 2022 The Turbo Authors.
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

#ifndef TURBO_BASE_INTERNAL_PREFETCH_H_
#define TURBO_BASE_INTERNAL_PREFETCH_H_

#include "turbo/platform/port.h"

#if TURBO_WITH_SSE3 || TURBO_WITH_SSE2

#include <xmmintrin.h>

#endif

#if defined(_MSC_VER) && (TURBO_WITH_SSE3 || TURBO_WITH_SSE2)
#include <intrin.h>
#pragma intrinsic(_mm_prefetch)
#endif

// Compatibility wrappers around __builtin_prefetch, to prefetch data
// for read if supported by the toolchain.

// Move data into the cache before it is read, or "prefetch" it.
//
// The value of `addr` is the address of the memory to prefetch. If
// the target and compiler support it, data prefetch instructions are
// generated. If the prefetch is done some time before the memory is
// read, it may be in the cache by the time the read occurs.
//
// The function names specify the temporal locality heuristic applied,
// using the names of Intel prefetch instructions:
//
//   T0 - high degree of temporal locality; data should be left in as
//        many levels of the cache possible
//   T1 - moderate degree of temporal locality
//   T2 - low degree of temporal locality
//   Nta - no temporal locality, data need not be left in the cache
//         after the read
//
// Incorrect or gratuitous use of these functions can degrade
// performance, so use them only when representative benchmarks show
// an improvement.
//
// Example usage:
//
//   turbo::prefetch_t0(addr);
//
// Currently, the different prefetch calls behave on some Intel
// architectures as follows:
//
//                 SNB..SKL   SKX
// prefetch_t0()   L1/L2/L3  L1/L2
// prefetch_t1()      L2/L3     L2
// prefetch_t2()      L2/L3     L2
// prefetch_nta()  L1/--/L3  L1*
//
// * On SKX prefetch_nta() will bring the line into L1 but will evict
//   from L3 cache. This might result in surprising behavior.
//
// SNB = Sandy Bridge, SKL = Skylake, SKX = Skylake Xeon.
//
namespace turbo {

    void prefetch_t0(const void *addr);

    void prefetch_t1(const void *addr);

    void prefetch_t2(const void *addr);

    void prefetch_nta(const void *addr);

// Implementation details follow.

#if TURBO_HAVE_BUILTIN(__builtin_prefetch) || defined(__GNUC__)

#define TURBO_INTERNAL_HAVE_PREFETCH 1

    // See __builtin_prefetch:
    // https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html.
    //
    // These functions speculatively load for read only. This is
    // safe for all currently supported platforms. However, prefetch for
    // store may have problems depending on the target platform.
    //
    inline void prefetch_t0(const void *addr) {
        // Note: this uses prefetcht0 on Intel.
        __builtin_prefetch(addr, 0, 3);
    }

    inline void prefetch_t1(const void *addr) {
        // Note: this uses prefetcht1 on Intel.
        __builtin_prefetch(addr, 0, 2);
    }

    inline void prefetch_t2(const void *addr) {
        // Note: this uses prefetcht2 on Intel.
        __builtin_prefetch(addr, 0, 1);
    }

    inline void prefetch_nta(const void *addr) {
        // Note: this uses prefetchtnta on Intel.
        __builtin_prefetch(addr, 0, 0);
    }

#elif (TURBO_WITH_SSE3 || TURBO_WITH_SSE2)

#define TURBO_INTERNAL_HAVE_PREFETCH 1

    inline void prefetch_t0(const void* addr) {
      _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
    }
    inline void prefetch_t1(const void* addr) {
      _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T1);
    }
    inline void prefetch_t2(const void* addr) {
      _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T2);
    }
    inline void prefetch_nta(const void* addr) {
      _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA);
    }

#else
    inline void prefetch_t0(const void*) {}
    inline void prefetch_t1(const void*) {}
    inline void prefetch_t2(const void*) {}
    inline void prefetch_nta(const void*) {}
#endif

}  // namespace turbo

#endif  // TURBO_BASE_INTERNAL_PREFETCH_H_
