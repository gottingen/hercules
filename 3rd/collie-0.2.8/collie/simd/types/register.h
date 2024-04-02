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

#ifndef COLLIE_SIMD_TYPES_REGISTER_H_
#define COLLIE_SIMD_TYPES_REGISTER_H_

#include <type_traits>

namespace collie::simd {
    namespace types {
        template<class T, class A>
        struct has_simd_register : std::false_type {
        };

        template<class T, class Arch>
        struct simd_register {
            struct register_type {
            };
        };

#define COLLIE_SIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA, VECTOR_TYPE) \
    template <>                                                    \
    struct simd_register<SCALAR_TYPE, ISA>                         \
    {                                                              \
        using register_type = VECTOR_TYPE;                         \
        register_type data;                                        \
        inline operator register_type() const noexcept             \
        {                                                          \
            return data;                                           \
        }                                                          \
    };                                                             \
    template <>                                                    \
    struct has_simd_register<SCALAR_TYPE, ISA> : std::true_type    \
    {                                                              \
    }

#define COLLIE_SIMD_DECLARE_INVALID_SIMD_REGISTER(SCALAR_TYPE, ISA)    \
    template <>                                                  \
    struct has_simd_register<SCALAR_TYPE, ISA> : std::false_type \
    {                                                            \
    }

#define COLLIE_SIMD_DECLARE_SIMD_REGISTER_ALIAS(ISA, ISA_BASE)                          \
    template <class T>                                                            \
    struct simd_register<T, ISA> : simd_register<T, ISA_BASE>                     \
    {                                                                             \
        using register_type = typename simd_register<T, ISA_BASE>::register_type; \
        simd_register(register_type reg) noexcept                                 \
            : simd_register<T, ISA_BASE> { reg }                                  \
        {                                                                         \
        }                                                                         \
        simd_register() = default;                                                \
    };                                                                            \
    template <class T>                                                            \
    struct has_simd_register<T, ISA> : has_simd_register<T, ISA_BASE>             \
    {                                                                             \
    }

        template<class T, class Arch>
        struct get_bool_simd_register {
            using type = simd_register<T, Arch>;
        };

        template<class T, class Arch>
        using get_bool_simd_register_t = typename get_bool_simd_register<T, Arch>::type;
    }

    namespace kernel {
        template<class A>
            // makes requires_arch equal to A const&, using type_traits functions
        using requires_arch = typename std::add_lvalue_reference<typename std::add_const<A>::type>::type;
        template<class T>
        struct convert {
        };
    }
}

#endif  // COLLIE_SIMD_TYPES_REGISTER_H_
