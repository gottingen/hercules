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

#ifndef COLLIE_SIMD_ARCH_GENERIC_ARITHMETIC_H_
#define COLLIE_SIMD_ARCH_GENERIC_ARITHMETIC_H_

#include <complex>
#include <limits>
#include <type_traits>

#include <collie/simd/arch/generic/details.h>

namespace collie::simd {

    namespace kernel {

        using namespace types;

        // bitwise_lshift
        template<class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline batch<T, A>
        bitwise_lshift(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept {
            return detail::apply([](T x, T y) noexcept { return x << y; },
                                 self, other);
        }

        // bitwise_rshift
        template<class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline batch<T, A>
        bitwise_rshift(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept {
            return detail::apply([](T x, T y) noexcept { return x >> y; },
                                 self, other);
        }

        // decr
        template<class A, class T>
        inline batch<T, A> decr(batch<T, A> const &self, requires_arch<generic>) noexcept {
            return self - T(1);
        }

        // decr_if
        template<class A, class T, class Mask>
        inline batch<T, A> decr_if(batch<T, A> const &self, Mask const &mask, requires_arch<generic>) noexcept {
            return select(mask, decr(self), self);
        }

        // div
        template<class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> div(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept {
            return detail::apply([](T x, T y) noexcept -> T { return x / y; },
                                 self, other);
        }

        // fma
        template<class A, class T>
        inline batch<T, A>
        fma(batch<T, A> const &x, batch<T, A> const &y, batch<T, A> const &z, requires_arch<generic>) noexcept {
            return x * y + z;
        }

        template<class A, class T>
        inline batch<std::complex<T>, A>
        fma(batch<std::complex<T>, A> const &x, batch<std::complex<T>, A> const &y, batch<std::complex<T>, A> const &z,
            requires_arch<generic>) noexcept {
            auto res_r = fms(x.real(), y.real(), fms(x.imag(), y.imag(), z.real()));
            auto res_i = fma(x.real(), y.imag(), fma(x.imag(), y.real(), z.imag()));
            return {res_r, res_i};
        }

        // fms
        template<class A, class T>
        inline batch<T, A>
        fms(batch<T, A> const &x, batch<T, A> const &y, batch<T, A> const &z, requires_arch<generic>) noexcept {
            return x * y - z;
        }

        template<class A, class T>
        inline batch<std::complex<T>, A>
        fms(batch<std::complex<T>, A> const &x, batch<std::complex<T>, A> const &y, batch<std::complex<T>, A> const &z,
            requires_arch<generic>) noexcept {
            auto res_r = fms(x.real(), y.real(), fma(x.imag(), y.imag(), z.real()));
            auto res_i = fma(x.real(), y.imag(), fms(x.imag(), y.real(), z.imag()));
            return {res_r, res_i};
        }

        // fnma
        template<class A, class T>
        inline batch<T, A>
        fnma(batch<T, A> const &x, batch<T, A> const &y, batch<T, A> const &z, requires_arch<generic>) noexcept {
            return -x * y + z;
        }

        template<class A, class T>
        inline batch<std::complex<T>, A>
        fnma(batch<std::complex<T>, A> const &x, batch<std::complex<T>, A> const &y, batch<std::complex<T>, A> const &z,
             requires_arch<generic>) noexcept {
            auto res_r = -fms(x.real(), y.real(), fma(x.imag(), y.imag(), z.real()));
            auto res_i = -fma(x.real(), y.imag(), fms(x.imag(), y.real(), z.imag()));
            return {res_r, res_i};
        }

        // fnms
        template<class A, class T>
        inline batch<T, A>
        fnms(batch<T, A> const &x, batch<T, A> const &y, batch<T, A> const &z, requires_arch<generic>) noexcept {
            return -x * y - z;
        }

        template<class A, class T>
        inline batch<std::complex<T>, A>
        fnms(batch<std::complex<T>, A> const &x, batch<std::complex<T>, A> const &y, batch<std::complex<T>, A> const &z,
             requires_arch<generic>) noexcept {
            auto res_r = -fms(x.real(), y.real(), fms(x.imag(), y.imag(), z.real()));
            auto res_i = -fma(x.real(), y.imag(), fma(x.imag(), y.real(), z.imag()));
            return {res_r, res_i};
        }

        // hadd
        template<class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline T hadd(batch<T, A> const &self, requires_arch<generic>) noexcept {
            alignas(A::alignment()) T buffer[batch<T, A>::size];
            self.store_aligned(buffer);
            T res = 0;
            for (T val: buffer) {
                res += val;
            }
            return res;
        }

        // incr
        template<class A, class T>
        inline batch<T, A> incr(batch<T, A> const &self, requires_arch<generic>) noexcept {
            return self + T(1);
        }

        // incr_if
        template<class A, class T, class Mask>
        inline batch<T, A> incr_if(batch<T, A> const &self, Mask const &mask, requires_arch<generic>) noexcept {
            return select(mask, incr(self), self);
        }

        // mul
        template<class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline batch<T, A> mul(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept {
            return detail::apply([](T x, T y) noexcept -> T { return x * y; },
                                 self, other);
        }

        // rotl
        template<class A, class T, class STy>
        inline batch<T, A> rotl(batch<T, A> const &self, STy other, requires_arch<generic>) noexcept {
            constexpr auto N = std::numeric_limits<T>::digits;
            return (self << other) | (self >> (N - other));
        }

        // rotr
        template<class A, class T, class STy>
        inline batch<T, A> rotr(batch<T, A> const &self, STy other, requires_arch<generic>) noexcept {
            constexpr auto N = std::numeric_limits<T>::digits;
            return (self >> other) | (self << (N - other));
        }

        // sadd
        template<class A>
        inline batch<float, A>
        sadd(batch<float, A> const &self, batch<float, A> const &other, requires_arch<generic>) noexcept {
            return add(self, other); // no saturated arithmetic on floating point numbers
        }

        template<class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline batch<T, A> sadd(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept {
            if (std::is_signed<T>::value) {
                auto mask = (other >> (8 * sizeof(T) - 1));
                auto self_pos_branch = min(std::numeric_limits<T>::max() - other, self);
                auto self_neg_branch = max(std::numeric_limits<T>::min() - other, self);
                return other + select(batch_bool<T, A>(mask.data), self_neg_branch, self_pos_branch);
            } else {
                const auto diffmax = std::numeric_limits<T>::max() - self;
                const auto mindiff = min(diffmax, other);
                return self + mindiff;
            }
        }

        template<class A>
        inline batch<double, A>
        sadd(batch<double, A> const &self, batch<double, A> const &other, requires_arch<generic>) noexcept {
            return add(self, other); // no saturated arithmetic on floating point numbers
        }

        // ssub
        template<class A>
        inline batch<float, A>
        ssub(batch<float, A> const &self, batch<float, A> const &other, requires_arch<generic>) noexcept {
            return sub(self, other); // no saturated arithmetic on floating point numbers
        }

        template<class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline batch<T, A> ssub(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept {
            if (std::is_signed<T>::value) {
                return sadd(self, -other);
            } else {
                const auto diff = min(self, other);
                return self - diff;
            }
        }

        template<class A>
        inline batch<double, A>
        ssub(batch<double, A> const &self, batch<double, A> const &other, requires_arch<generic>) noexcept {
            return sub(self, other); // no saturated arithmetic on floating point numbers
        }

    }

}

#endif  // COLLIE_SIMD_ARCH_GENERIC_ARITHMETIC_H_
