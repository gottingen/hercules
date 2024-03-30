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


#ifndef COLLIE_SIMD_TYPES_AVX512VNNI_AVX512VBMI_REGISTER_H_
#define COLLIE_SIMD_TYPES_AVX512VNNI_AVX512VBMI_REGISTER_H_

#include <collie/simd/types/avx512vbmi_register.h>

namespace collie::simd {
    template<typename arch>
    struct avx512vnni;

    /**
     * @ingroup architectures
     *
     * AVX512VNNI instructions
     */
    template<>
    struct avx512vnni<avx512vbmi> : avx512vbmi {
        static constexpr bool supported() noexcept { return COLLIE_SIMD_WITH_AVX512VNNI_AVX512VBMI; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return generic::version(3, 6, 1); }

        static constexpr char const *name() noexcept { return "avx512vnni+avx512vbmi"; }
    };

#if COLLIE_SIMD_WITH_AVX512VNNI_AVX512VBMI

    namespace types
    {
        template <class T>
        struct get_bool_simd_register<T, avx512vnni<avx512vbmi>>
        {
            using type = simd_avx512_bool_register<T>;
        };

        COLLIE_SIMD_DECLARE_SIMD_REGISTER_ALIAS(avx512vnni<avx512vbmi>, avx512vbmi);

    }
#endif
}
#endif  // COLLIE_SIMD_TYPES_AVX512VNNI_AVX512VBMI_REGISTER_H_
