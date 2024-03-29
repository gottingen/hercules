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

#ifndef COLLIE_SIMD_TYPES_NEON_REGISTER_H_
#define COLLIE_SIMD_TYPES_NEON_REGISTER_H_

#include <collie/simd/types/simd_generic_arch.h>
#include <collie/simd/types/register.h>

#if COLLIE_SIMD_WITH_NEON
#include <arm_neon.h>
#endif

namespace collie::simd {
    /**
     * @ingroup architectures
     *
     * NEON instructions for arm32
     */
    struct neon : generic {
        static constexpr bool supported() noexcept { return COLLIE_SIMD_WITH_NEON; }

        static constexpr bool available() noexcept { return true; }

        static constexpr bool requires_alignment() noexcept { return true; }

        static constexpr std::size_t alignment() noexcept { return 16; }

        static constexpr unsigned version() noexcept { return generic::version(7, 0, 0); }

        static constexpr char const *name() noexcept { return "arm32+neon"; }
    };

#if COLLIE_SIMD_WITH_NEON
    namespace types
    {
        namespace detail
        {
            template <size_t S>
            struct neon_vector_type_impl;

            template <>
            struct neon_vector_type_impl<8>
            {
                using signed_type = int8x16_t;
                using unsigned_type = uint8x16_t;
            };

            template <>
            struct neon_vector_type_impl<16>
            {
                using signed_type = int16x8_t;
                using unsigned_type = uint16x8_t;
            };

            template <>
            struct neon_vector_type_impl<32>
            {
                using signed_type = int32x4_t;
                using unsigned_type = uint32x4_t;
            };

            template <>
            struct neon_vector_type_impl<64>
            {
                using signed_type = int64x2_t;
                using unsigned_type = uint64x2_t;
            };

            template <class T>
            using signed_neon_vector_type = typename neon_vector_type_impl<8 * sizeof(T)>::signed_type;

            template <class T>
            using unsigned_neon_vector_type = typename neon_vector_type_impl<8 * sizeof(T)>::unsigned_type;

            template <class T>
            using neon_vector_type = typename std::conditional<std::is_signed<T>::value,
                                                               signed_neon_vector_type<T>,
                                                               unsigned_neon_vector_type<T>>::type;

            using char_neon_vector_type = typename std::conditional<std::is_signed<char>::value,
                                                                    signed_neon_vector_type<char>,
                                                                    unsigned_neon_vector_type<char>>::type;
        }

        COLLIE_SIMD_DECLARE_SIMD_REGISTER(signed char, neon, detail::neon_vector_type<signed char>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned char, neon, detail::neon_vector_type<unsigned char>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(char, neon, detail::char_neon_vector_type);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(short, neon, detail::neon_vector_type<short>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned short, neon, detail::neon_vector_type<unsigned short>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(int, neon, detail::neon_vector_type<int>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned int, neon, detail::neon_vector_type<unsigned int>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(long int, neon, detail::neon_vector_type<long int>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned long int, neon, detail::neon_vector_type<unsigned long int>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(long long int, neon, detail::neon_vector_type<long long int>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned long long int, neon, detail::neon_vector_type<unsigned long long int>);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(float, neon, float32x4_t);
        COLLIE_SIMD_DECLARE_INVALID_SIMD_REGISTER(double, neon);

        namespace detail
        {
            template <size_t S>
            struct get_unsigned_type;

            template <>
            struct get_unsigned_type<1>
            {
                using type = uint8_t;
            };

            template <>
            struct get_unsigned_type<2>
            {
                using type = uint16_t;
            };

            template <>
            struct get_unsigned_type<4>
            {
                using type = uint32_t;
            };

            template <>
            struct get_unsigned_type<8>
            {
                using type = uint64_t;
            };

            template <size_t S>
            using get_unsigned_type_t = typename get_unsigned_type<S>::type;

            template <class T, class A>
            struct neon_bool_simd_register
            {
                using type = simd_register<get_unsigned_type_t<sizeof(T)>, A>;
            };
        }

        template <class T>
        struct get_bool_simd_register<T, neon>
            : detail::neon_bool_simd_register<T, neon>
        {
        };

    }
#endif

}  // namespace collie::simd

#endif  // COLLIE_SIMD_TYPES_NEON_REGISTER_H_
