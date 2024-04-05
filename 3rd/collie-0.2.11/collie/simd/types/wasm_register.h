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

#ifndef COLLIE_SIMD_TYPES_WASM_REGISTER_H_
#define COLLIE_SIMD_TYPES_WASM_REGISTER_H_

#include <collie/simd/types/simd_generic_arch.h>
#include <collie/simd/types/register.h>

#if COLLIE_SIMD_WASM
#include <wasm_simd128.h>
#endif

namespace collie::simd {
    /**
     * @ingroup architectures
     *
     * WASM instructions
     */
    struct wasm : generic {
        static constexpr bool supported() noexcept { return COLLIE_SIMD_WASM; }

        static constexpr bool available() noexcept { return true; }

        static constexpr bool requires_alignment() noexcept { return true; }

        static constexpr unsigned version() noexcept { return generic::version(10, 0, 0); }

        static constexpr std::size_t alignment() noexcept { return 16; }

        static constexpr char const *name() noexcept { return "wasm"; }
    };

#if COLLIE_SIMD_WASM
    namespace types {
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(signed char, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned char, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(char, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned short, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(short, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned int, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(int, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned long int, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(long int, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(unsigned long long int, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(long long int, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(float, wasm, v128_t);
        COLLIE_SIMD_DECLARE_SIMD_REGISTER(double, wasm, v128_t);
    }
#endif
}

#endif  // COLLIE_SIMD_TYPES_WASM_REGISTER_H_
