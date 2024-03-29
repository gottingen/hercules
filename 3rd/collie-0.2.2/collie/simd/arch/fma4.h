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

#ifndef COLLIE_SIMD_ARCH_FMA4_H_
#define COLLIE_SIMD_ARCH_FMA4_H_

#include <collie/simd/types/fma4_register.h>

namespace collie::simd {

    namespace kernel {
        using namespace types;

        // fnma
        template<class A>
        inline batch<float, A>
        fnma(simd_register<float, A> const &x, simd_register<float, A> const &y, simd_register<float, A> const &z,
             requires_arch<fma4>) noexcept {
            return _mm_nmacc_ps(x, y, z);
        }

        template<class A>
        inline batch<double, A>
        fnma(simd_register<double, A> const &x, simd_register<double, A> const &y, simd_register<double, A> const &z,
             requires_arch<fma4>) noexcept {
            return _mm_nmacc_pd(x, y, z);
        }

        // fnms
        template<class A>
        inline batch<float, A>
        fnms(simd_register<float, A> const &x, simd_register<float, A> const &y, simd_register<float, A> const &z,
             requires_arch<fma4>) noexcept {
            return _mm_nmsub_ps(x, y, z);
        }

        template<class A>
        inline batch<double, A>
        fnms(simd_register<double, A> const &x, simd_register<double, A> const &y, simd_register<double, A> const &z,
             requires_arch<fma4>) noexcept {
            return _mm_nmsub_pd(x, y, z);
        }

        // fma
        template<class A>
        inline batch<float, A>
        fma(simd_register<float, A> const &x, simd_register<float, A> const &y, simd_register<float, A> const &z,
            requires_arch<fma4>) noexcept {
            return _mm_macc_ps(x, y, z);
        }

        template<class A>
        inline batch<double, A>
        fma(simd_register<double, A> const &x, simd_register<double, A> const &y, simd_register<double, A> const &z,
            requires_arch<fma4>) noexcept {
            return _mm_macc_pd(x, y, z);
        }

        // fms
        template<class A>
        inline batch<float, A>
        fms(simd_register<float, A> const &x, simd_register<float, A> const &y, simd_register<float, A> const &z,
            requires_arch<fma4>) noexcept {
            return _mm_msub_ps(x, y, z);
        }

        template<class A>
        inline batch<double, A>
        fms(simd_register<double, A> const &x, simd_register<double, A> const &y, simd_register<double, A> const &z,
            requires_arch<fma4>) noexcept {
            return _mm_msub_pd(x, y, z);
        }
    }

}  // namespace collie::simd

#endif  // COLLIE_SIMD_ARCH_FMA4_H_
