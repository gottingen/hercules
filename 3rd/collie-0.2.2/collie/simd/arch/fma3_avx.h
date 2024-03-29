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

#ifndef COLLIE_SIMD_ARCH_FMA3_AVX_H_
#define COLLIE_SIMD_ARCH_FMA3_AVX_H_

#include <collie/simd/types/fma3_avx_register.h>

namespace collie::simd {

    namespace kernel {
        using namespace types;

        // fnma
        template<class A>
        inline batch<float, A> fnma(batch<float, A> const &x, batch<float, A> const &y, batch<float, A> const &z,
                                    requires_arch<fma3<avx>>) noexcept {
            return _mm256_fnmadd_ps(x, y, z);
        }

        template<class A>
        inline batch<double, A> fnma(batch<double, A> const &x, batch<double, A> const &y, batch<double, A> const &z,
                                     requires_arch<fma3<avx>>) noexcept {
            return _mm256_fnmadd_pd(x, y, z);
        }

        // fnms
        template<class A>
        inline batch<float, A> fnms(batch<float, A> const &x, batch<float, A> const &y, batch<float, A> const &z,
                                    requires_arch<fma3<avx>>) noexcept {
            return _mm256_fnmsub_ps(x, y, z);
        }

        template<class A>
        inline batch<double, A> fnms(batch<double, A> const &x, batch<double, A> const &y, batch<double, A> const &z,
                                     requires_arch<fma3<avx>>) noexcept {
            return _mm256_fnmsub_pd(x, y, z);
        }

        // fma
        template<class A>
        inline batch<float, A> fma(batch<float, A> const &x, batch<float, A> const &y, batch<float, A> const &z,
                                   requires_arch<fma3<avx>>) noexcept {
            return _mm256_fmadd_ps(x, y, z);
        }

        template<class A>
        inline batch<double, A> fma(batch<double, A> const &x, batch<double, A> const &y, batch<double, A> const &z,
                                    requires_arch<fma3<avx>>) noexcept {
            return _mm256_fmadd_pd(x, y, z);
        }

        // fms
        template<class A>
        inline batch<float, A> fms(batch<float, A> const &x, batch<float, A> const &y, batch<float, A> const &z,
                                   requires_arch<fma3<avx>>) noexcept {
            return _mm256_fmsub_ps(x, y, z);
        }

        template<class A>
        inline batch<double, A> fms(batch<double, A> const &x, batch<double, A> const &y, batch<double, A> const &z,
                                    requires_arch<fma3<avx>>) noexcept {
            return _mm256_fmsub_pd(x, y, z);
        }

    }

}  // namespace collie::simd

#endif  // COLLIE_SIMD_ARCH_FMA3_AVX_H_
