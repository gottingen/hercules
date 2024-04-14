// Copyright 2020 The Turbo Authors.
//
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

#include "int128.h"

#include <stddef.h>

#include <cassert>
#include <iomanip>
#include <ostream>  // NOLINT(readability/streams)
#include <sstream>
#include <string>
#include <type_traits>

#include "turbo/base/bits.h"
#include "turbo/platform/port.h"

namespace turbo {

    TURBO_DLL const uint128 kuint128max = make_uint128(
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max());

    namespace {

        // Returns the 0-based position of the last set bit (i.e., most significant bit)
        // in the given uint128. The argument is not 0.
        //
        // For example:
        //   Given: 5 (decimal) == 101 (binary)
        //   Returns: 2
        TURBO_FORCE_INLINE int Fls128(uint128 n) {
            if (uint64_t hi = uint128_high64(n)) {
                TURBO_ASSUME(hi != 0);
                return 127 - countl_zero(hi);
            }
            const uint64_t low = uint128_low64(n);
            TURBO_ASSUME(low != 0);
            return 63 - countl_zero(low);
        }

        // Long division/modulo for uint128 implemented using the shift-subtract
        // division algorithm adapted from:
        // https://stackoverflow.com/questions/5386377/division-without-using
        inline void DivModImpl(uint128 dividend, uint128 divisor, uint128 *quotient_ret,
                               uint128 *remainder_ret) {
            assert(divisor != 0);

            if (divisor > dividend) {
                *quotient_ret = 0;
                *remainder_ret = dividend;
                return;
            }

            if (divisor == dividend) {
                *quotient_ret = 1;
                *remainder_ret = 0;
                return;
            }

            uint128 denominator = divisor;
            uint128 quotient = 0;

            // Left aligns the MSB of the denominator and the dividend.
            const int shift = Fls128(dividend) - Fls128(denominator);
            denominator <<= shift;

            // Uses shift-subtract algorithm to divide dividend by denominator. The
            // remainder will be left in dividend.
            for (int i = 0; i <= shift; ++i) {
                quotient <<= 1;
                if (dividend >= denominator) {
                    dividend -= denominator;
                    quotient |= 1;
                }
                denominator >>= 1;
            }

            *quotient_ret = quotient;
            *remainder_ret = dividend;
        }

        template<typename T>
        uint128 MakeUint128FromFloat(T v) {
            static_assert(std::is_floating_point<T>::value, "");

            // Rounding behavior is towards zero, same as for built-in types.

            // Undefined behavior if v is NaN or cannot fit into uint128.
            assert(std::isfinite(v) && v > -1 &&
                   (std::numeric_limits<T>::max_exponent <= 128 ||
                    v < std::ldexp(static_cast<T>(1), 128)));

            if (v >= std::ldexp(static_cast<T>(1), 64)) {
                uint64_t hi = static_cast<uint64_t>(std::ldexp(v, -64));
                uint64_t lo = static_cast<uint64_t>(v - std::ldexp(static_cast<T>(hi), 64));
                return make_uint128(hi, lo);
            }

            return make_uint128(0, static_cast<uint64_t>(v));
        }

#if defined(__clang__) && !TURBO_WITH_SSE3
        // Workaround for clang bug: https://bugs.llvm.org/show_bug.cgi?id=38289
        // Casting from long double to uint64_t is miscompiled and drops bits.
        // It is more work, so only use when we need the workaround.
        uint128 MakeUint128FromFloat(long double v) {
          // Go 50 bits at a time, that fits in a double
          static_assert(std::numeric_limits<double>::digits >= 50, "");
          static_assert(std::numeric_limits<long double>::digits <= 150, "");
          // Undefined behavior if v is not finite or cannot fit into uint128.
          assert(std::isfinite(v) && v > -1 && v < std::ldexp(1.0L, 128));

          v = std::ldexp(v, -100);
          uint64_t w0 = static_cast<uint64_t>(static_cast<double>(std::trunc(v)));
          v = std::ldexp(v - static_cast<double>(w0), 50);
          uint64_t w1 = static_cast<uint64_t>(static_cast<double>(std::trunc(v)));
          v = std::ldexp(v - static_cast<double>(w1), 50);
          uint64_t w2 = static_cast<uint64_t>(static_cast<double>(std::trunc(v)));
          return (static_cast<uint128>(w0) << 100) | (static_cast<uint128>(w1) << 50) |
                 static_cast<uint128>(w2);
        }
#endif  // __clang__ && !TURBO_WITH_SSE3
    }  // namespace

    uint128::uint128(float v) : uint128(MakeUint128FromFloat(v)) {}

    uint128::uint128(double v) : uint128(MakeUint128FromFloat(v)) {}

    uint128::uint128(long double v) : uint128(MakeUint128FromFloat(v)) {}

    namespace {

        std::string Uint128ToFormattedString(uint128 v, std::ios_base::fmtflags flags) {
            // Select a divisor which is the largest power of the base < 2^64.
            uint128 div;
            int div_base_log;
            switch (flags & std::ios::basefield) {
                case std::ios::hex:
                    div = 0x1000000000000000;  // 16^15
                    div_base_log = 15;
                    break;
                case std::ios::oct:
                    div = 01000000000000000000000;  // 8^21
                    div_base_log = 21;
                    break;
                default:  // std::ios::dec
                    div = 10000000000000000000u;  // 10^19
                    div_base_log = 19;
                    break;
            }

            // Now piece together the uint128 representation from three chunks of the
            // original value, each less than "div" and therefore representable as a
            // uint64_t.
            std::ostringstream os;
            std::ios_base::fmtflags copy_mask =
                    std::ios::basefield | std::ios::showbase | std::ios::uppercase;
            os.setf(flags & copy_mask, copy_mask);
            uint128 high = v;
            uint128 low;
            DivModImpl(high, div, &high, &low);
            uint128 mid;
            DivModImpl(high, div, &high, &mid);
            if (uint128_low64(high) != 0) {
                os << uint128_low64(high);
                os << std::noshowbase << std::setfill('0') << std::setw(div_base_log);
                os << uint128_low64(mid);
                os << std::setw(div_base_log);
            } else if (uint128_low64(mid) != 0) {
                os << uint128_low64(mid);
                os << std::noshowbase << std::setfill('0') << std::setw(div_base_log);
            }
            os << uint128_low64(low);
            return os.str();
        }

    }  // namespace

    std::ostream &operator<<(std::ostream &os, uint128 v) {
        std::ios_base::fmtflags flags = os.flags();
        std::string rep = Uint128ToFormattedString(v, flags);

        // Add the requisite padding.
        std::streamsize width = os.width(0);
        if (static_cast<size_t>(width) > rep.size()) {
            const size_t count = static_cast<size_t>(width) - rep.size();
            std::ios::fmtflags adjustfield = flags & std::ios::adjustfield;
            if (adjustfield == std::ios::left) {
                rep.append(count, os.fill());
            } else if (adjustfield == std::ios::internal &&
                       (flags & std::ios::showbase) &&
                       (flags & std::ios::basefield) == std::ios::hex && v != 0) {
                rep.insert(2, count, os.fill());
            } else {
                rep.insert(0, count, os.fill());
            }
        }

        return os << rep;
    }

    namespace {

        uint128 UnsignedAbsoluteValue(int128 v) {
            // Cast to uint128 before possibly negating because -int128_min() is undefined.
            return int128_high64(v) < 0 ? -uint128(v) : uint128(v);
        }

    }  // namespace

    std::ostream &operator<<(std::ostream &os, int128 v) {
        std::ios_base::fmtflags flags = os.flags();
        std::string rep;

        // Add the sign if needed.
        bool print_as_decimal =
                (flags & std::ios::basefield) == std::ios::dec ||
                (flags & std::ios::basefield) == std::ios_base::fmtflags();
        if (print_as_decimal) {
            if (int128_high64(v) < 0) {
                rep = "-";
            } else if (flags & std::ios::showpos) {
                rep = "+";
            }
        }

        rep.append(Uint128ToFormattedString(
                print_as_decimal ? UnsignedAbsoluteValue(v) : uint128(v), os.flags()));

        // Add the requisite padding.
        std::streamsize width = os.width(0);
        if (static_cast<size_t>(width) > rep.size()) {
            const size_t count = static_cast<size_t>(width) - rep.size();
            switch (flags & std::ios::adjustfield) {
                case std::ios::left:
                    rep.append(count, os.fill());
                    break;
                case std::ios::internal:
                    if (print_as_decimal && (rep[0] == '+' || rep[0] == '-')) {
                        rep.insert(1, count, os.fill());
                    } else if ((flags & std::ios::basefield) == std::ios::hex &&
                               (flags & std::ios::showbase) && v != 0) {
                        rep.insert(2, count, os.fill());
                    } else {
                        rep.insert(0, count, os.fill());
                    }
                    break;
                default:  // std::ios::right
                    rep.insert(0, count, os.fill());
                    break;
            }
        }

        return os << rep;
    }

}  // namespace turbo