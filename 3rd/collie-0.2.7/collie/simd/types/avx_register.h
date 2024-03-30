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


#ifndef COLLIE_SIMD_TYPES_AVX_REGISTER_H_
#define COLLIE_SIMD_TYPES_AVX_REGISTER_H_

#include <collie/simd/types/simd_generic_arch.h>

namespace collie::simd {

    /**
     * @ingroup architectures
     *
     * AVX instructions
     */
    struct avx : generic {
        static constexpr bool supported() noexcept { return COLLIE_SIMD_WITH_AVX; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return generic::version(2, 1, 0); }

        static constexpr std::size_t alignment() noexcept { return 32; }

        static constexpr bool requires_alignment() noexcept { return true; }

        static constexpr char const *name() noexcept { return "avx"; }
    };
}

#if COLLIE_SIMD_WITH_AVX

#include <immintrin.h>

namespace collie::simd {
    namespace types {

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(signed char, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned char, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(char, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned short, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(short, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned int, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(int, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned long int, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(long int, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned long long int, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(long long int, avx, __m256i);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(float, avx, __m256);

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(double, avx, __m256d);
    }
}
#endif
#endif  // COLLIE_SIMD_TYPES_AVX_REGISTER_H_
