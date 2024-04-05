// Copyright 2024 The Elastic-AI Authors.
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

#ifndef COLLIE_SIMD_TYPES_SSSE3_REGISTER_H_
#define COLLIE_SIMD_TYPES_SSSE3_REGISTER_H_

#include <collie/simd/types/sse3_register.h>

#if COLLIE_SIMD_WITH_SSSE3

#include <tmmintrin.h>

#endif

namespace collie::simd {
    /**
     * @ingroup architectures
     *
     * SSSE3 instructions
     */
    struct ssse3 : sse3 {
        static constexpr bool supported() noexcept { return COLLIE_SIMD_WITH_SSSE3; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return generic::version(1, 3, 1); }

        static constexpr char const *name() noexcept { return "ssse3"; }
    };

#if COLLIE_SIMD_WITH_SSSE3
    namespace types {
        COLLIE_SIMD_DECLARE_SIMD_REGISTER_ALIAS(ssse3, sse3);
    }
#endif
}

#endif  // COLLIE_SIMD_TYPES_SSSE3_REGISTER_H_
