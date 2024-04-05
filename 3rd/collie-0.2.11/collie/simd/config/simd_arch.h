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

#ifndef COLLIE_SIMD_CONFIG_ARCH_H_
#define COLLIE_SIMD_CONFIG_ARCH_H_

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <collie/base/engine_registry.h>
#include <collie/simd/types/all_registers.h>
#include <collie/port/simd_config.h>
#include <collie/simd/config/simd_cpuid.h>

namespace collie::simd {

    // just for testing
    struct unsupported {};

    using all_x86_architectures = engine_list<
            avx512vnni<avx512vbmi>, avx512vbmi, avx512ifma, avx512pf, avx512vnni<avx512bw>, avx512bw, avx512er, avx512dq, avx512cd, avx512f,
            avxvnni, fma3<avx2>, avx2, fma3<avx>, avx, fma4, fma3<sse4_2>,
            sse4_2, sse4_1, /*sse4a,*/ ssse3, sse3, sse2>;

    using all_sve_architectures = engine_list<detail::sve<512>, detail::sve<256>, detail::sve<128>>;
    using all_rvv_architectures = engine_list<detail::rvv<512>, detail::rvv<256>, detail::rvv<128>>;
    using all_arm_architectures = typename engine_join<all_sve_architectures, engine_list<neon64, neon>>::type;
    using all_riscv_architectures = all_rvv_architectures;
    using all_wasm_architectures = engine_list<wasm>;
    using all_architectures = typename engine_join<all_riscv_architectures, all_wasm_architectures, all_arm_architectures, all_x86_architectures>::type;

    using supported_architectures = typename engine_supported<all_architectures>::type;

    using x86_arch = typename engine_supported<all_x86_architectures>::type::best;
    using arm_arch = typename engine_supported<all_arm_architectures>::type::best;
    using riscv_arch = typename engine_supported<all_riscv_architectures>::type::best;
    using best_arch = typename supported_architectures::best;

#ifdef COLLIE_SIMD_DEFAULT_ARCH
    using default_arch = COLLIE_SIMD_DEFAULT_ARCH;
#else
    using default_arch = best_arch;
#endif

    // Generic function dispatch, à la ifunc
    template<class ArchList = supported_architectures, class F>
    inline engine_dispatcher<F, ArchList> simd_dispatch(F &&f) noexcept {
        return dispatch<ArchList, F>(available_architectures().best, std::forward<F>(f));
    }

} // namespace collie::simd

#endif  // COLLIE_SIMD_CONFIG_ARCH_H_
