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

#ifndef COLLIE_SIMD_TYPES_SIMD_TRAITS_H_
#define COLLIE_SIMD_TYPES_SIMD_TRAITS_H_

#include <type_traits>

#include <collie/simd/types/batch.h>

/**
 * high level type traits
 *
 * @defgroup batch_traits Type traits
 *
 **/

namespace collie::simd {

    /**************************************
     * simd_traits and revert_simd_traits *
     **************************************/

    template<class T, class A = default_arch>
    struct has_simd_register : types::has_simd_register<T, A> {
    };

    namespace detail {
        template<class T, bool>
        struct simd_traits_impl;

        template<class T>
        struct simd_traits_impl<T, false> {
            using type = T;
            using bool_type = bool;
            static constexpr size_t size = 1;
        };

        template<class T>
        constexpr size_t simd_traits_impl<T, false>::size;

        template<class T>
        struct simd_traits_impl<T, true> {
            using type = batch<T>;
            using bool_type = typename type::batch_bool_type;
            static constexpr size_t size = type::size;
        };

        template<class T>
        constexpr size_t simd_traits_impl<T, true>::size;

        template<class T, class A>
        struct static_check_supported_config_emitter {

            static_assert(A::supported(),
                          "usage of batch type with unsupported architecture");
            static_assert(!A::supported() || collie::simd::has_simd_register<T, A>::value,
                          "usage of batch type with unsupported type");
        };

        template<class T, class A>
        struct static_check_supported_config_emitter<std::complex<T>, A> : static_check_supported_config_emitter<T, A> {
        };

        // consistency checker
        template<class T, class A>
        inline void static_check_supported_config() {
            (void) static_check_supported_config_emitter<T, A>();
        }
    }

    template<class T>
    struct simd_traits : detail::simd_traits_impl<T, collie::simd::has_simd_register<T>::value> {
    };

    template<class T>
    struct simd_traits<std::complex<T>>
            : detail::simd_traits_impl<std::complex<T>, collie::simd::has_simd_register<T>::value> {
    };


    template<class T>
    struct revert_simd_traits {
        using type = T;
        static constexpr size_t size = simd_traits<type>::size;
    };

    template<class T>
    constexpr size_t revert_simd_traits<T>::size;

    template<class T>
    struct revert_simd_traits<batch<T>> {
        using type = T;
        static constexpr size_t size = batch<T>::size;
    };

    template<class T>
    constexpr size_t revert_simd_traits<batch<T>>::size;

    template<class T>
    using simd_type = typename simd_traits<T>::type;

    template<class T>
    using simd_bool_type = typename simd_traits<T>::bool_type;

    template<class T>
    using revert_simd_type = typename revert_simd_traits<T>::type;

    /********************
     * simd_return_type *
     ********************/

    namespace detail {
        template<class T1, class T2>
        struct simd_condition {
            static constexpr bool value = (std::is_same<T1, T2>::value && !std::is_same<T1, bool>::value) ||
                                          (std::is_same<T1, bool>::value && !std::is_same<T2, bool>::value) ||
                                          std::is_same<T1, float>::value || std::is_same<T1, double>::value ||
                                          std::is_same<T1, int8_t>::value || std::is_same<T1, uint8_t>::value ||
                                          std::is_same<T1, int16_t>::value || std::is_same<T1, uint16_t>::value ||
                                          std::is_same<T1, int32_t>::value || std::is_same<T1, uint32_t>::value ||
                                          std::is_same<T1, int64_t>::value || std::is_same<T1, uint64_t>::value ||
                                          std::is_same<T1, char>::value || detail::is_complex<T1>::value;
        };

        template<class T1, class T2, class A>
        struct simd_return_type_impl
                : std::enable_if<simd_condition<T1, T2>::value, batch<T2, A>> {
        };

        template<class T2, class A>
        struct simd_return_type_impl<bool, T2, A>
                : std::enable_if<simd_condition<bool, T2>::value, batch_bool<T2, A>> {
        };

        template<class T2, class A>
        struct simd_return_type_impl<bool, std::complex<T2>, A>
                : std::enable_if<simd_condition<bool, T2>::value, batch_bool<T2, A>> {
        };

        template<class T1, class T2, class A>
        struct simd_return_type_impl<std::complex<T1>, T2, A>
                : std::enable_if<simd_condition<T1, T2>::value, batch<std::complex<T2>, A>> {
        };

        template<class T1, class T2, class A>
        struct simd_return_type_impl<std::complex<T1>, std::complex<T2>, A>
                : std::enable_if<simd_condition<T1, T2>::value, batch<std::complex<T2>, A>> {
        };

    }

    template<class T1, class T2, class A = default_arch>
    using simd_return_type = typename detail::simd_return_type_impl<T1, T2, A>::type;

    /**
     * @ingroup batch_traits
     *
     * type traits that inherits from @c std::true_type for @c batch<...> types and from
     * @c std::false_type otherwise.
     *
     * @tparam T type to analyze.
     */
    template<class T>
    struct is_batch;

    template<class T>
    struct is_batch : std::false_type {
    };

    template<class T, class A>
    struct is_batch<batch<T, A>> : std::true_type {
    };

    /**
     * @ingroup batch_traits
     *
     * type traits that inherits from @c std::true_type for @c batch_bool<...> types and from
     * @c std::false_type otherwise.
     *
     * @tparam T type to analyze.
     */

    template<class T>
    struct is_batch_bool : std::false_type {
    };

    template<class T, class A>
    struct is_batch_bool<batch_bool<T, A>> : std::true_type {
    };

    /**
     * @ingroup batch_traits
     *
     * type traits that inherits from @c std::true_type for @c batch<std::complex<...>>
     * types and from @c std::false_type otherwise.
     *
     * @tparam T type to analyze.
     */

    template<class T>
    struct is_batch_complex : std::false_type {
    };

    template<class T, class A>
    struct is_batch_complex<batch<std::complex<T>, A>> : std::true_type {
    };

    /**
     * @ingroup batch_traits
     *
     * type traits whose @c type field is set to @c T::value_type if @c
     * is_batch<T>::value and to @c T otherwise.
     *
     * @tparam T type to analyze.
     */
    template<class T>
    struct scalar_type {
        using type = T;
    };
    template<class T, class A>
    struct scalar_type<batch<T, A>> {
        using type = T;
    };

    template<class T>
    using scalar_type_t = typename scalar_type<T>::type;

    /**
     * @ingroup batch_traits
     *
     * type traits whose @c type field is set to @c T::value_type if @c
     * is_batch_bool<T>::value and to @c bool otherwise.
     *
     * @tparam T type to analyze.
     */
    template<class T>
    struct mask_type {
        using type = bool;
    };
    template<class T, class A>
    struct mask_type<batch<T, A>> {
        using type = typename batch<T, A>::batch_bool_type;
    };

    template<class T>
    using mask_type_t = typename mask_type<T>::type;
}

#endif  // COLLIE_SIMD_TYPES_SIMD_TRAITS_H_
