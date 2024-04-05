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

#ifndef COLLIE_SIMD_TYPES_AVX512F_REGISTER_H_
#define COLLIE_SIMD_TYPES_AVX512F_REGISTER_H_

#include <collie/simd/types/simd_generic_arch.h>

namespace collie::simd {

    /**
     * @ingroup architectures
     *
     * AVX512F instructions
     */
    struct avx512f : generic {
        static constexpr bool supported() noexcept { return COLLIE_SIMD_AVX512F; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return generic::version(3, 1, 0); }

        static constexpr std::size_t alignment() noexcept { return 64; }

        static constexpr bool requires_alignment() noexcept { return true; }

        static constexpr char const *name() noexcept { return "avx512f"; }
    };

#if COLLIE_SIMD_AVX512F

    namespace types
    {
        template <class T>
        struct simd_avx512_bool_register
        {
            using register_type = typename std::conditional<
                (sizeof(T) < 4), std::conditional<(sizeof(T) == 1), __mmask64, __mmask32>,
                std::conditional<(sizeof(T) == 4), __mmask16, __mmask8>>::type::type;
            register_type data;
            simd_avx512_bool_register() = default;
            simd_avx512_bool_register(register_type r) { data = r; }
            operator register_type() const noexcept { return data; }
        };
        template <class T>
        struct get_bool_simd_register<T, avx512f>
        {
            using type = simd_avx512_bool_register<T>;
        };

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(signed char, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned char, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(char, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned short, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(short, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned int, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(int, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned long int, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(long int, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned long long int, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(long long int, avx512f, __m512i);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(float, avx512f, __m512);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(double, avx512f, __m512d);

    }
#endif
}

#endif  // COLLIE_SIMD_TYPES_AVX512F_REGISTER_H_
