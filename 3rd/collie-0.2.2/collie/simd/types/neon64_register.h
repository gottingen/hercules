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


#ifndef COLLIE_SIMD_TYPES_NEON64_REGISTER_H_
#define COLLIE_SIMD_TYPES_NEON64_REGISTER_H_

#include <collie/simd/types/neon_register.h>

namespace collie::simd {
    /**
     * @ingroup architectures
     *
     * NEON instructions for arm64
     */
    struct neon64 : neon {
        static constexpr bool supported() noexcept { return COLLIE_SIMD_WITH_NEON64; }

        static constexpr bool available() noexcept { return true; }

        static constexpr bool requires_alignment() noexcept { return true; }

        static constexpr std::size_t alignment() noexcept { return 16; }

        static constexpr unsigned version() noexcept { return generic::version(8, 1, 0); }

        static constexpr char const *name() noexcept { return "arm64+neon"; }
    };

#if COLLIE_SIMD_WITH_NEON64

    namespace types
    {
        COLLIE_SIMD_DECLARE_SIMD_REGISTER_ALIAS(neon64, neon);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(double, neon64, float64x2_t);

        template <class T>
        struct get_bool_simd_register<T, neon64>
            : detail::neon_bool_simd_register<T, neon64>
        {
        };
    }

#endif

}

#endif  // COLLIE_SIMD_TYPES_NEON64_REGISTER_H_
