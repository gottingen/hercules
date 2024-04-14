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
#include "turbo/platform/port.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

#ifdef TURBO_HAVE_INTRINSIC_INT128

#include "turbo/base/int128.h"

#include <algorithm>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "turbo/platform/internal/cycleclock.h"
#include "turbo/meta/type_traits.h"
#include "turbo/random/random.h"

#if defined(_MSC_VER) && _MSC_VER == 1900
// Disable "unary minus operator applied to unsigned type" warnings in Microsoft
// Visual C++ 14 (2015).
#pragma warning(disable:4146)
#endif

namespace {

    using IntTypes = std::tuple<bool, char, signed char, unsigned char, char16_t,
            char32_t, wchar_t,
            short,           // NOLINT(runtime/int)
            unsigned short,  // NOLINT(runtime/int)
            int, unsigned int,
            long,                // NOLINT(runtime/int)
            unsigned long,       // NOLINT(runtime/int)
            long long,           // NOLINT(runtime/int)
            unsigned long long>;

    using FloatingPointTypes = std::tuple<float, double, long double>;

    TEST_CASE_TEMPLATE_DEFINE("Uint128TraitsTest", T, IntTypesConstructAssignTest) {
        static_assert(std::is_constructible<turbo::uint128, T>::value,
                      "turbo::uint128 must be constructible from T");
        static_assert(std::is_assignable<turbo::uint128 &, T>::value,
                      "turbo::uint128 must be assignable from T");
        static_assert(!std::is_assignable<T &, turbo::uint128>::value,
                      "T must not be assignable from turbo::uint128");
    }

    TEST_CASE_TEMPLATE_APPLY(IntTypesConstructAssignTest, IntTypes);


    TEST_CASE_TEMPLATE_DEFINE("Uint128TraitsTest", T, FloatingPointTypesConstructAssignTest) {
        static_assert(std::is_constructible<turbo::uint128, T>::value,
                      "turbo::uint128 must be constructible from T");
        static_assert(!std::is_assignable<turbo::uint128 &, T>::value,
                      "turbo::uint128 must not be assignable from T");
        static_assert(!std::is_assignable<T &, turbo::uint128>::value,
                      "T must not be assignable from turbo::uint128");
    }

    TEST_CASE_TEMPLATE_APPLY(FloatingPointTypesConstructAssignTest, FloatingPointTypes);
    // These type traits done separately as TYPED_TEST requires typeinfo, and not
    // all platforms have this for __int128 even though they define the type.
    TEST_CASE("UINT128") {
        SUBCASE("IntrinsicTypeTraitsTest") {
            static_assert(std::is_constructible<turbo::uint128, __int128>::value,
                          "turbo::uint128 must be constructible from __int128");
            static_assert(std::is_assignable<turbo::uint128 &, __int128>::value,
                          "turbo::uint128 must be assignable from __int128");
            static_assert(!std::is_assignable<__int128 &, turbo::uint128>::value,
                          "__int128 must not be assignable from turbo::uint128");

            static_assert(std::is_constructible<turbo::uint128, unsigned __int128>::value,
                          "turbo::uint128 must be constructible from unsigned __int128");
            static_assert(std::is_assignable<turbo::uint128 &, unsigned __int128>::value,
                          "turbo::uint128 must be assignable from unsigned __int128");
            static_assert(!std::is_assignable<unsigned __int128 &, turbo::uint128>::value,
                          "unsigned __int128 must not be assignable from turbo::uint128");
        }

        SUBCASE("TrivialTraitsTest") {
            static_assert(std::is_trivially_default_constructible<turbo::uint128>::value,
                          "");
            static_assert(std::is_trivially_copy_constructible<turbo::uint128>::value,
                          "");
            static_assert(std::is_trivially_copy_assignable<turbo::uint128>::value, "");
            static_assert(std::is_trivially_destructible<turbo::uint128>::value, "");
        }

        SUBCASE("AllTests") {
            turbo::uint128 zero = 0;
            turbo::uint128 one = 1;
            turbo::uint128 one_2arg = turbo::make_uint128(0, 1);
            turbo::uint128 two = 2;
            turbo::uint128 three = 3;
            turbo::uint128 big = turbo::make_uint128(2000, 2);
            turbo::uint128 big_minus_one = turbo::make_uint128(2000, 1);
            turbo::uint128 bigger = turbo::make_uint128(2001, 1);
            turbo::uint128 biggest = turbo::uint128_max();
            turbo::uint128 high_low = turbo::make_uint128(1, 0);
            turbo::uint128 low_high =
                    turbo::make_uint128(0, std::numeric_limits<uint64_t>::max());
            REQUIRE_LT(one, two);
            REQUIRE_GT(two, one);
            REQUIRE_LT(one, big);
            REQUIRE_LT(one, big);
            REQUIRE_EQ(one, one_2arg);
            REQUIRE_NE(one, two);
            REQUIRE_GT(big, one);
            REQUIRE_GE(big, two);
            REQUIRE_GE(big, big_minus_one);
            REQUIRE_GT(big, big_minus_one);
            REQUIRE_LT(big_minus_one, big);
            REQUIRE_LE(big_minus_one, big);
            REQUIRE_NE(big_minus_one, big);
            REQUIRE_LT(big, biggest);
            REQUIRE_LE(big, biggest);
            REQUIRE_GT(biggest, big);
            REQUIRE_GE(biggest, big);
            REQUIRE_EQ(big, ~~big);
            REQUIRE_EQ(one, one | one);
            REQUIRE_EQ(big, big | big);
            REQUIRE_EQ(one, one | zero);
            REQUIRE_EQ(one, one & one);
            REQUIRE_EQ(big, big & big);
            REQUIRE_EQ(zero, one & zero);
            REQUIRE_EQ(zero, big & ~big);
            REQUIRE_EQ(zero, one ^ one);
            REQUIRE_EQ(zero, big ^ big);
            REQUIRE_EQ(one, one ^ zero);

            // Shift operators.
            REQUIRE_EQ(big, big << 0);
            REQUIRE_EQ(big, big >> 0);
            REQUIRE_GT(big << 1, big);
            REQUIRE_LT(big >> 1, big);
            REQUIRE_EQ(big, (big << 10) >> 10);
            REQUIRE_EQ(big, (big >> 1) << 1);
            REQUIRE_EQ(one, (one << 80) >> 80);
            REQUIRE_EQ(zero, (one >> 80) << 80);

            // Shift assignments.
            turbo::uint128 big_copy = big;
            REQUIRE_EQ(big << 0, big_copy <<= 0);
            big_copy = big;
            REQUIRE_EQ(big >> 0, big_copy >>= 0);
            big_copy = big;
            REQUIRE_EQ(big << 1, big_copy <<= 1);
            big_copy = big;
            REQUIRE_EQ(big >> 1, big_copy >>= 1);
            big_copy = big;
            REQUIRE_EQ(big << 10, big_copy <<= 10);
            big_copy = big;
            REQUIRE_EQ(big >> 10, big_copy >>= 10);
            big_copy = big;
            REQUIRE_EQ(big << 64, big_copy <<= 64);
            big_copy = big;
            REQUIRE_EQ(big >> 64, big_copy >>= 64);
            big_copy = big;
            REQUIRE_EQ(big << 73, big_copy <<= 73);
            big_copy = big;
            REQUIRE_EQ(big >> 73, big_copy >>= 73);

            REQUIRE_EQ(turbo::uint128_high64(biggest), std::numeric_limits<uint64_t>::max());
            REQUIRE_EQ(turbo::uint128_low64(biggest), std::numeric_limits<uint64_t>::max());
            REQUIRE_EQ(zero + one, one);
            REQUIRE_EQ(one + one, two);
            REQUIRE_EQ(big_minus_one + one, big);
            REQUIRE_EQ(one - one, zero);
            REQUIRE_EQ(one - zero, one);
            REQUIRE_EQ(zero - one, biggest);
            REQUIRE_EQ(big - big, zero);
            REQUIRE_EQ(big - one, big_minus_one);
            REQUIRE_EQ(big + std::numeric_limits<uint64_t>::max(), bigger);
            REQUIRE_EQ(biggest + 1, zero);
            REQUIRE_EQ(zero - 1, biggest);
            REQUIRE_EQ(high_low - one, low_high);
            REQUIRE_EQ(low_high + one, high_low);
            REQUIRE_EQ(turbo::uint128_high64((turbo::uint128(1) << 64) - 1), 0);
            REQUIRE_EQ(turbo::uint128_low64((turbo::uint128(1) << 64) - 1),
                       std::numeric_limits<uint64_t>::max());
            REQUIRE(!!one);
            REQUIRE(!!high_low);
            REQUIRE_FALSE(!!zero);
            REQUIRE_FALSE(!one);
            REQUIRE_FALSE(!high_low);
            REQUIRE(!zero);
            REQUIRE(zero == 0);       // NOLINT(readability/check)
            REQUIRE_FALSE(zero != 0);      // NOLINT(readability/check)
            REQUIRE_FALSE(one == 0);       // NOLINT(readability/check)
            REQUIRE(one != 0);        // NOLINT(readability/check)
            REQUIRE_FALSE(high_low == 0);  // NOLINT(readability/check)
            REQUIRE(high_low != 0);   // NOLINT(readability/check)

            turbo::uint128 test = zero;
            REQUIRE_EQ(++test, one);
            REQUIRE_EQ(test, one);
            REQUIRE_EQ(test++, one);
            REQUIRE_EQ(test, two);
            REQUIRE_EQ(test -= 2, zero);
            REQUIRE_EQ(test, zero);
            REQUIRE_EQ(test += 2, two);
            REQUIRE_EQ(test, two);
            REQUIRE_EQ(--test, one);
            REQUIRE_EQ(test, one);
            REQUIRE_EQ(test--, one);
            REQUIRE_EQ(test, zero);
            REQUIRE_EQ(test |= three, three);
            REQUIRE_EQ(test &= one, one);
            REQUIRE_EQ(test ^= three, two);
            REQUIRE_EQ(test >>= 1, one);
            REQUIRE_EQ(test <<= 1, two);

            REQUIRE_EQ(big, +big);
            REQUIRE_EQ(two, +two);
            REQUIRE_EQ(turbo::uint128_max(), +turbo::uint128_max());
            REQUIRE_EQ(zero, +zero);

            REQUIRE_EQ(big, -(-big));
            REQUIRE_EQ(two, -((-one) - 1));
            REQUIRE_EQ(turbo::uint128_max(), -one);
            REQUIRE_EQ(zero, -zero);

            REQUIRE_EQ(turbo::uint128_max(), turbo::kuint128max);
        }
    }

    TEST_CASE("INT128") {
        SUBCASE("RightShiftOfNegativeNumbers") {
            turbo::int128 minus_six = -6;
            turbo::int128 minus_three = -3;
            turbo::int128 minus_two = -2;
            turbo::int128 minus_one = -1;
            if ((-6 >> 1) == -3) {
                // Right shift is arithmetic (sign propagates)
                REQUIRE_EQ(minus_six >> 1, minus_three);
                REQUIRE_EQ(minus_six >> 2, minus_two);
                REQUIRE_EQ(minus_six >> 65, minus_one);
            } else {
                // Right shift is logical (zeros shifted in at MSB)
                REQUIRE_EQ(minus_six >> 1, turbo::int128(turbo::uint128(minus_six) >> 1));
                REQUIRE_EQ(minus_six >> 2, turbo::int128(turbo::uint128(minus_six) >> 2));
                REQUIRE_EQ(minus_six >> 65, turbo::int128(turbo::uint128(minus_six) >> 65));
            }
        }

        SUBCASE("ConversionTests") {
            REQUIRE(turbo::make_uint128(1, 0));

            unsigned __int128 intrinsic =
                    (static_cast<unsigned __int128>(0x3a5b76c209de76f6) << 64) +
                    0x1f25e1d63a2b46c5;
            turbo::uint128 custom =
                    turbo::make_uint128(0x3a5b76c209de76f6, 0x1f25e1d63a2b46c5);

            REQUIRE_EQ(custom, turbo::uint128(intrinsic));
            REQUIRE_EQ(custom, turbo::uint128(static_cast<__int128>(intrinsic)));
            REQUIRE_EQ(intrinsic, static_cast<unsigned __int128>(custom));
            REQUIRE_EQ(intrinsic, static_cast<__int128>(custom));

            // verify that an integer greater than 2**64 that can be stored precisely
            // inside a double is converted to a turbo::uint128 without loss of
            // information.
            double precise_double = 0x530e * std::pow(2.0, 64.0) + 0xda74000000000000;
            turbo::uint128 from_precise_double(precise_double);
            turbo::uint128 from_precise_ints =
                    turbo::make_uint128(0x530e, 0xda74000000000000);
            REQUIRE_EQ(from_precise_double, from_precise_ints);
            REQUIRE_LT(std::abs(static_cast<double>(from_precise_ints) - precise_double), 0.001);

            double approx_double = 0xffffeeeeddddcccc * std::pow(2.0, 64.0) +
                                   0xbbbbaaaa99998888;
            turbo::uint128 from_approx_double(approx_double);
            REQUIRE_LT(std::abs(static_cast<double>(from_approx_double) - approx_double), 0.01);

            double round_to_zero = 0.7;
            double round_to_five = 5.8;
            double round_to_nine = 9.3;
            REQUIRE_EQ(static_cast<turbo::uint128>(round_to_zero), 0);
            REQUIRE_EQ(static_cast<turbo::uint128>(round_to_five), 5);
            REQUIRE_EQ(static_cast<turbo::uint128>(round_to_nine), 9);

            turbo::uint128 highest_precision_in_long_double =
                    ~turbo::uint128{} >> (128 - std::numeric_limits<long double>::digits);
            REQUIRE_EQ(highest_precision_in_long_double,
                       static_cast<turbo::uint128>(
                               static_cast<long double>(highest_precision_in_long_double)));
            // Apply a mask just to make sure all the bits are the right place.
            const turbo::uint128 arbitrary_mask =
                    turbo::make_uint128(0xa29f622677ded751, 0xf8ca66add076f468);
            REQUIRE_EQ(highest_precision_in_long_double & arbitrary_mask,
                       static_cast<turbo::uint128>(static_cast<long double>(
                               highest_precision_in_long_double & arbitrary_mask)));

            REQUIRE_EQ(static_cast<turbo::uint128>(-0.1L), 0);
        }

        SUBCASE("OperatorAssignReturnRef") {
            turbo::uint128 v(1);
            (v += 4) -= 3;
            REQUIRE_EQ(2, v);
        }

        SUBCASE("Multiply") {
            turbo::uint128 a, b, c;

            // Zero test.
            a = 0;
            b = 0;
            c = a * b;
            REQUIRE_EQ(0, c);

            // Max carries.
            a = turbo::uint128(0) - 1;
            b = turbo::uint128(0) - 1;
            c = a * b;
            REQUIRE_EQ(1, c);

            // Self-operation with max carries.
            c = turbo::uint128(0) - 1;
            c *= c;
            REQUIRE_EQ(1, c);

            // 1-bit x 1-bit.
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 64; ++j) {
                    a = turbo::uint128(1) << i;
                    b = turbo::uint128(1) << j;
                    c = a * b;
                    REQUIRE_EQ(turbo::uint128(1) << (i + j), c);
                }
            }

            // Verified with dc.
            a = turbo::make_uint128(0xffffeeeeddddcccc, 0xbbbbaaaa99998888);
            b = turbo::make_uint128(0x7777666655554444, 0x3333222211110000);
            c = a * b;
            REQUIRE_EQ(turbo::make_uint128(0x530EDA741C71D4C3, 0xBF25975319080000), c);
            REQUIRE_EQ(0, c - b * a);
            REQUIRE_EQ(a * a - b * b, (a + b) * (a - b));

            // Verified with dc.
            a = turbo::make_uint128(0x0123456789abcdef, 0xfedcba9876543210);
            b = turbo::make_uint128(0x02468ace13579bdf, 0xfdb97531eca86420);
            c = a * b;
            REQUIRE_EQ(turbo::make_uint128(0x97a87f4f261ba3f2, 0x342d0bbf48948200), c);
            REQUIRE_EQ(0, c - b * a);
            REQUIRE_EQ(a * a - b * b, (a + b) * (a - b));
        }

        SUBCASE("AliasTests") {
            turbo::uint128 x1 = turbo::make_uint128(1, 2);
            turbo::uint128 x2 = turbo::make_uint128(2, 4);
            x1 += x1;
            REQUIRE_EQ(x2, x1);

            turbo::uint128 x3 = turbo::make_uint128(1, static_cast<uint64_t>(1) << 63);
            turbo::uint128 x4 = turbo::make_uint128(3, 0);
            x3 += x3;
            REQUIRE_EQ(x4, x3);
        }

        SUBCASE("DivideAndMod") {
            using std::swap;

            // a := q * b + r
            turbo::uint128 a, b, q, r;

            // Zero test.
            a = 0;
            b = 123;
            q = a / b;
            r = a % b;
            REQUIRE_EQ(0, q);
            REQUIRE_EQ(0, r);

            a = turbo::make_uint128(0x530eda741c71d4c3, 0xbf25975319080000);
            q = turbo::make_uint128(0x4de2cab081, 0x14c34ab4676e4bab);
            b = turbo::uint128(0x1110001);
            r = turbo::uint128(0x3eb455);
            REQUIRE_EQ(a, q * b + r);  // Sanity-check.

            turbo::uint128 result_q, result_r;
            result_q = a / b;
            result_r = a % b;
            REQUIRE_EQ(q, result_q);
            REQUIRE_EQ(r, result_r);

            // Try the other way around.
            swap(q, b);
            result_q = a / b;
            result_r = a % b;
            REQUIRE_EQ(q, result_q);
            REQUIRE_EQ(r, result_r);
            // Restore.
            swap(b, q);

            // Dividend < divisor; result should be q:0 r:<dividend>.
            swap(a, b);
            result_q = a / b;
            result_r = a % b;
            REQUIRE_EQ(0, result_q);
            REQUIRE_EQ(a, result_r);
            // Try the other way around.
            swap(a, q);
            result_q = a / b;
            result_r = a % b;
            REQUIRE_EQ(0, result_q);
            REQUIRE_EQ(a, result_r);
            // Restore.
            swap(q, a);
            swap(b, a);

            // Try a large remainder.
            b = a / 2 + 1;
            turbo::uint128 expected_r =
                    turbo::make_uint128(0x29876d3a0e38ea61, 0xdf92cba98c83ffff);
            // Sanity checks.
            REQUIRE_EQ(a / 2 - 1, expected_r);
            REQUIRE_EQ(a, b + expected_r);
            result_q = a / b;
            result_r = a % b;
            REQUIRE_EQ(1, result_q);
            REQUIRE_EQ(expected_r, result_r);
        }

        SUBCASE("DivideAndModRandomInputs") {
            const int kNumIters = 1 << 18;
            std::uniform_int_distribution<uint64_t> uniform_uint64;
            for (int i = 0; i < kNumIters; ++i) {
                const turbo::uint128 a =
                        turbo::make_uint128(turbo::fast_uniform<uint64_t>(), turbo::fast_uniform<uint64_t>());
                const turbo::uint128 b =
                        turbo::make_uint128(turbo::fast_uniform<uint64_t>(), turbo::fast_uniform<uint64_t>());
                if (b == 0) {
                    continue;  // Avoid a div-by-zero.
                }
                const turbo::uint128 q = a / b;
                const turbo::uint128 r = a % b;
                REQUIRE_EQ(a, b * q + r);
            }
        }

        SUBCASE("ConstexprTest") {
            constexpr turbo::uint128 zero = turbo::uint128();
            constexpr turbo::uint128 one = 1;
            constexpr turbo::uint128 minus_two = -2;
            REQUIRE_EQ(zero, turbo::uint128(0));
            REQUIRE_EQ(one, turbo::uint128(1));
            REQUIRE_EQ(minus_two, turbo::make_uint128(-1, -2));
        }

        SUBCASE("NumericLimitsTest") {
            static_assert(std::numeric_limits<turbo::uint128>::is_specialized, "");
            static_assert(!std::numeric_limits<turbo::uint128>::is_signed, "");
            static_assert(std::numeric_limits<turbo::uint128>::is_integer, "");
            REQUIRE_EQ(static_cast<int>(128 * std::log10(2)),
                       std::numeric_limits<turbo::uint128>::digits10);
            REQUIRE_EQ(0, std::numeric_limits<turbo::uint128>::min());
            REQUIRE_EQ(0, std::numeric_limits<turbo::uint128>::lowest());
            REQUIRE_EQ(turbo::uint128_max(), std::numeric_limits<turbo::uint128>::max());
        }


        SUBCASE("ConversionTest") {
            turbo::int128 nonnegative_signed_values[] = {
                    0,
                    1,
                    0xffeeddccbbaa9988,
                    turbo::make_int128(0x7766554433221100, 0),
                    turbo::make_int128(0x1234567890abcdef, 0xfedcba0987654321),
                    turbo::int128_max()};
            for (turbo::int128 value: nonnegative_signed_values) {
                REQUIRE_EQ(value, turbo::int128(turbo::uint128(value)));

                turbo::uint128 assigned_value;
                assigned_value = value;
                REQUIRE_EQ(value, turbo::int128(assigned_value));
            }

            turbo::int128 negative_values[] = {
                    -1, -0x1234567890abcdef,
                    turbo::make_int128(-0x5544332211ffeedd, 0),
                    -turbo::make_int128(0x76543210fedcba98, 0xabcdef0123456789)};
            for (turbo::int128 value: negative_values) {
                REQUIRE_EQ(turbo::uint128(-value), -turbo::uint128(value));

                turbo::uint128 assigned_value;
                assigned_value = value;
                REQUIRE_EQ(turbo::uint128(-value), -assigned_value);
            }
        }

// These type traits done separately as TYPED_TEST requires typeinfo, and not
// all platforms have this for __int128 even though they define the type.
        SUBCASE("IntrinsicTypeTraitsTest") {
            static_assert(std::is_constructible<turbo::int128, __int128>::value,
                          "turbo::int128 must be constructible from __int128");
            static_assert(std::is_assignable<turbo::int128 &, __int128>::value,
                          "turbo::int128 must be assignable from __int128");
            static_assert(!std::is_assignable<__int128 &, turbo::int128>::value,
                          "__int128 must not be assignable from turbo::int128");

            static_assert(std::is_constructible<turbo::int128, unsigned __int128>::value,
                          "turbo::int128 must be constructible from unsigned __int128");
            static_assert(!std::is_assignable<turbo::int128 &, unsigned __int128>::value,
                          "turbo::int128 must be assignable from unsigned __int128");
            static_assert(!std::is_assignable<unsigned __int128 &, turbo::int128>::value,
                          "unsigned __int128 must not be assignable from turbo::int128");
        }

        SUBCASE("TrivialTraitsTest") {
            static_assert(std::is_trivially_default_constructible<turbo::int128>::value,
                          "");
            static_assert(std::is_trivially_copy_constructible<turbo::int128>::value, "");
            static_assert(std::is_trivially_copy_assignable<turbo::int128>::value, "");
            static_assert(std::is_trivially_destructible<turbo::int128>::value, "");
        }

        SUBCASE("BoolConversionTest") {
            REQUIRE_FALSE(turbo::int128(0));
            for (int i = 0; i < 64; ++i) {
                REQUIRE(turbo::make_int128(0, uint64_t{1} << i));
            }
            for (int i = 0; i < 63; ++i) {
                REQUIRE(turbo::make_int128(int64_t{1} << i, 0));
            }
            REQUIRE(turbo::int128_min());

            REQUIRE_EQ(turbo::int128(1), turbo::int128(true));
            REQUIRE_EQ(turbo::int128(0), turbo::int128(false));
        }


        SUBCASE("FactoryTest") {
            REQUIRE_EQ(turbo::int128(-1), turbo::make_int128(-1, -1));
            REQUIRE_EQ(turbo::int128(-31), turbo::make_int128(-1, -31));
            REQUIRE_EQ(turbo::int128(std::numeric_limits<int64_t>::min()),
                       turbo::make_int128(-1, std::numeric_limits<int64_t>::min()));
            REQUIRE_EQ(turbo::int128(0), turbo::make_int128(0, 0));
            REQUIRE_EQ(turbo::int128(1), turbo::make_int128(0, 1));
            REQUIRE_EQ(turbo::int128(std::numeric_limits<int64_t>::max()),
                       turbo::make_int128(0, std::numeric_limits<int64_t>::max()));
        }

        SUBCASE("HighLowTest") {
            struct HighLowPair {
                int64_t high;
                uint64_t low;
            };
            HighLowPair values[]{{0,    0},
                                 {0,    1},
                                 {1,    0},
                                 {123,  456},
                                 {-654, 321}};
            for (const HighLowPair &pair: values) {
                turbo::int128 value = turbo::make_int128(pair.high, pair.low);
                REQUIRE_EQ(pair.low, turbo::int128_low64(value));
                REQUIRE_EQ(pair.high, turbo::int128_high64(value));
            }
        }

        SUBCASE("LimitsTest") {
            REQUIRE_EQ(turbo::make_int128(0x7fffffffffffffff, 0xffffffffffffffff),
                       turbo::int128_max());
            REQUIRE_EQ(turbo::int128_max(), ~turbo::int128_min());
        }

        SUBCASE("IntrinsicConversionTest") {
            __int128 intrinsic =
                    (static_cast<__int128>(0x3a5b76c209de76f6) << 64) + 0x1f25e1d63a2b46c5;
            turbo::int128 custom =
                    turbo::make_int128(0x3a5b76c209de76f6, 0x1f25e1d63a2b46c5);

            REQUIRE_EQ(custom, turbo::int128(intrinsic));
            REQUIRE_EQ(intrinsic, static_cast<__int128>(custom));
        }

        SUBCASE("ConstexprTest") {
            constexpr turbo::int128 zero = turbo::int128();
            constexpr turbo::int128 one = 1;
            constexpr turbo::int128 minus_two = -2;
            constexpr turbo::int128 min = turbo::int128_min();
            constexpr turbo::int128 max = turbo::int128_max();
            REQUIRE_EQ(zero, turbo::int128(0));
            REQUIRE_EQ(one, turbo::int128(1));
            REQUIRE_EQ(minus_two, turbo::make_int128(-1, -2));
            REQUIRE_GT(max, one);
            REQUIRE_LT(min, minus_two);
        }

        SUBCASE("ComparisonTest") {
            struct TestCase {
                turbo::int128 smaller;
                turbo::int128 larger;
            };
            TestCase cases[] = {
                    {turbo::int128(0),               turbo::int128(123)},
                    {turbo::make_int128(-12, 34),     turbo::make_int128(12, 34)},
                    {turbo::make_int128(1, 1000),     turbo::make_int128(1000, 1)},
                    {turbo::make_int128(-1000, 1000), turbo::make_int128(-1, 1)},
            };
            for (const TestCase &pair: cases) {
                //SCOPED_TRACE(::testing::Message() << "pair.smaller = " << pair.smaller
                //                                 << "; pair.larger = " << pair.larger);

                REQUIRE(pair.smaller == pair.smaller);  // NOLINT(readability/check)
                REQUIRE(pair.larger == pair.larger);    // NOLINT(readability/check)
                REQUIRE_FALSE(pair.smaller == pair.larger);  // NOLINT(readability/check)

                REQUIRE(pair.smaller != pair.larger);    // NOLINT(readability/check)
                REQUIRE_FALSE(pair.smaller != pair.smaller);  // NOLINT(readability/check)
                REQUIRE_FALSE(pair.larger != pair.larger);    // NOLINT(readability/check)

                REQUIRE(pair.smaller < pair.larger);   // NOLINT(readability/check)
                REQUIRE_FALSE(pair.larger < pair.smaller);  // NOLINT(readability/check)

                REQUIRE(pair.larger > pair.smaller);   // NOLINT(readability/check)
                REQUIRE_FALSE(pair.smaller > pair.larger);  // NOLINT(readability/check)

                REQUIRE(pair.smaller <= pair.larger);   // NOLINT(readability/check)
                REQUIRE_FALSE(pair.larger <= pair.smaller);  // NOLINT(readability/check)
                REQUIRE(pair.smaller <= pair.smaller);  // NOLINT(readability/check)
                REQUIRE(pair.larger <= pair.larger);    // NOLINT(readability/check)

                REQUIRE(pair.larger >= pair.smaller);   // NOLINT(readability/check)
                REQUIRE_FALSE(pair.smaller >= pair.larger);  // NOLINT(readability/check)
                REQUIRE(pair.smaller >= pair.smaller);  // NOLINT(readability/check)
                REQUIRE(pair.larger >= pair.larger);    // NOLINT(readability/check)
            }
        }

        SUBCASE("UnaryPlusTest") {
            int64_t values64[] = {0, 1, 12345, 0x4000000000000000,
                                  std::numeric_limits<int64_t>::max()};
            for (int64_t value: values64) {
                //SCOPED_TRACE(::testing::Message() << "value = " << value);

                REQUIRE_EQ(turbo::int128(value), +turbo::int128(value));
                REQUIRE_EQ(turbo::int128(-value), +turbo::int128(-value));
                REQUIRE_EQ(turbo::make_int128(value, 0), +turbo::make_int128(value, 0));
                REQUIRE_EQ(turbo::make_int128(-value, 0), +turbo::make_int128(-value, 0));
            }
        }

        SUBCASE("UnaryNegationTest") {
            int64_t values64[] = {0, 1, 12345, 0x4000000000000000,
                                  std::numeric_limits<int64_t>::max()};
            for (int64_t value: values64) {
                //SCOPED_TRACE(::testing::Message() << "value = " << value);

                REQUIRE_EQ(turbo::int128(-value), -turbo::int128(value));
                REQUIRE_EQ(turbo::int128(value), -turbo::int128(-value));
                REQUIRE_EQ(turbo::make_int128(-value, 0), -turbo::make_int128(value, 0));
                REQUIRE_EQ(turbo::make_int128(value, 0), -turbo::make_int128(-value, 0));
            }
        }

        SUBCASE("LogicalNotTest") {
            REQUIRE(!turbo::int128(0));
            for (int i = 0; i < 64; ++i) {
                REQUIRE_FALSE(!turbo::make_int128(0, uint64_t{1} << i));
            }
            for (int i = 0; i < 63; ++i) {
                REQUIRE_FALSE(!turbo::make_int128(int64_t{1} << i, 0));
            }
        }

        SUBCASE("AdditionSubtractionTest") {
            // 64 bit pairs that will not cause overflow / underflow. These test negative
            // carry; positive carry must be checked separately.
            std::pair<int64_t, int64_t> cases[]{
                    {0,               0},                              // 0, 0
                    {0,               2945781290834},                  // 0, +
                    {1908357619234,   0},                  // +, 0
                    {0,               -1204895918245},                 // 0, -
                    {-2957928523560,  0},                 // -, 0
                    {89023982312461,  98346012567134},    // +, +
                    {-63454234568239, -23456235230773},  // -, -
                    {98263457263502,  -21428561935925},   // +, -
                    {-88235237438467, 15923659234573},   // -, +
            };
            for (const auto &pair: cases) {
                // SCOPED_TRACE(::testing::Message()
                //            << "pair = {" << pair.first << ", " << pair.second << '}');

                REQUIRE_EQ(turbo::int128(pair.first + pair.second),
                           turbo::int128(pair.first) + turbo::int128(pair.second));
                REQUIRE_EQ(turbo::int128(pair.second + pair.first),
                           turbo::int128(pair.second) += turbo::int128(pair.first));

                REQUIRE_EQ(turbo::int128(pair.first - pair.second),
                           turbo::int128(pair.first) - turbo::int128(pair.second));
                REQUIRE_EQ(turbo::int128(pair.second - pair.first),
                           turbo::int128(pair.second) -= turbo::int128(pair.first));

                REQUIRE_EQ(
                        turbo::make_int128(pair.second + pair.first, 0),
                        turbo::make_int128(pair.second, 0) + turbo::make_int128(pair.first, 0));
                REQUIRE_EQ(
                        turbo::make_int128(pair.first + pair.second, 0),
                        turbo::make_int128(pair.first, 0) += turbo::make_int128(pair.second, 0));

                REQUIRE_EQ(
                        turbo::make_int128(pair.second - pair.first, 0),
                        turbo::make_int128(pair.second, 0) - turbo::make_int128(pair.first, 0));
                REQUIRE_EQ(
                        turbo::make_int128(pair.first - pair.second, 0),
                        turbo::make_int128(pair.first, 0) -= turbo::make_int128(pair.second, 0));
            }

            // check positive carry
            REQUIRE_EQ(turbo::make_int128(31, 0),
                       turbo::make_int128(20, 1) +
                       turbo::make_int128(10, std::numeric_limits<uint64_t>::max()));
        }

        SUBCASE("IncrementDecrementTest") {
            turbo::int128 value = 0;
            REQUIRE_EQ(0, value++);
            REQUIRE_EQ(1, value);
            REQUIRE_EQ(1, value--);
            REQUIRE_EQ(0, value);
            REQUIRE_EQ(-1, --value);
            REQUIRE_EQ(-1, value);
            REQUIRE_EQ(0, ++value);
            REQUIRE_EQ(0, value);
        }

        SUBCASE("MultiplicationTest") {
            // 1 bit x 1 bit, and negative combinations
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 127 - i; ++j) {
                    // SCOPED_TRACE(::testing::Message() << "i = " << i << "; j = " << j);
                    turbo::int128 a = turbo::int128(1) << i;
                    turbo::int128 b = turbo::int128(1) << j;
                    turbo::int128 c = turbo::int128(1) << (i + j);

                    REQUIRE_EQ(c, a * b);
                    REQUIRE_EQ(-c, -a * b);
                    REQUIRE_EQ(-c, a * -b);
                    REQUIRE_EQ(c, -a * -b);

                    REQUIRE_EQ(c, turbo::int128(a) *= b);
                    REQUIRE_EQ(-c, turbo::int128(-a) *= b);
                    REQUIRE_EQ(-c, turbo::int128(a) *= -b);
                    REQUIRE_EQ(c, turbo::int128(-a) *= -b);
                }
            }

            // Pairs of random values that will not overflow signed 64-bit multiplication
            std::pair<int64_t, int64_t> small_values[] = {
                    {0x5e61,        0xf29f79ca14b4},    // +, +
                    {0x3e033b,      -0x612c0ee549},   // +, -
                    {-0x052ce7e8,   0x7c728f0f},   // -, +
                    {-0x3af7054626, -0xfb1e1d},  // -, -
            };
            for (const std::pair<int64_t, int64_t> &pair: small_values) {
                //SCOPED_TRACE(::testing::Message()
                //            << "pair = {" << pair.first << ", " << pair.second << '}');

                REQUIRE_EQ(turbo::int128(pair.first * pair.second),
                           turbo::int128(pair.first) * turbo::int128(pair.second));
                REQUIRE_EQ(turbo::int128(pair.first * pair.second),
                           turbo::int128(pair.first) *= turbo::int128(pair.second));

                REQUIRE_EQ(turbo::make_int128(pair.first * pair.second, 0),
                           turbo::make_int128(pair.first, 0) * turbo::int128(pair.second));
                REQUIRE_EQ(turbo::make_int128(pair.first * pair.second, 0),
                           turbo::make_int128(pair.first, 0) *= turbo::int128(pair.second));
            }

            // Pairs of positive random values that will not overflow 64-bit
            // multiplication and can be left shifted by 32 without overflow
            std::pair<int64_t, int64_t> small_values2[] = {
                    {0x1bb0a110, 0x31487671},
                    {0x4792784e, 0x28add7d7},
                    {0x7b66553a, 0x11dff8ef},
            };
            for (const std::pair<int64_t, int64_t> &pair: small_values2) {
                //SCOPED_TRACE(::testing::Message()
                //            << "pair = {" << pair.first << ", " << pair.second << '}');

                turbo::int128 a = turbo::int128(pair.first << 32);
                turbo::int128 b = turbo::int128(pair.second << 32);
                turbo::int128 c = turbo::make_int128(pair.first * pair.second, 0);

                REQUIRE_EQ(c, a * b);
                REQUIRE_EQ(-c, -a * b);
                REQUIRE_EQ(-c, a * -b);
                REQUIRE_EQ(c, -a * -b);

                REQUIRE_EQ(c, turbo::int128(a) *= b);
                REQUIRE_EQ(-c, turbo::int128(-a) *= b);
                REQUIRE_EQ(-c, turbo::int128(a) *= -b);
                REQUIRE_EQ(c, turbo::int128(-a) *= -b);
            }

            // check 0, 1, and -1 behavior with large values
            turbo::int128 large_values[] = {
                    {turbo::make_int128(0xd66f061af02d0408, 0x727d2846cb475b53)},
                    {turbo::make_int128(0x27b8d5ed6104452d, 0x03f8a33b0ee1df4f)},
                    {-turbo::make_int128(0x621b6626b9e8d042, 0x27311ac99df00938)},
                    {-turbo::make_int128(0x34e0656f1e95fb60, 0x4281cfd731257a47)},
            };
            for (turbo::int128 value: large_values) {
                REQUIRE_EQ(0, 0 * value);
                REQUIRE_EQ(0, value * 0);
                REQUIRE_EQ(0, turbo::int128(0) *= value);
                REQUIRE_EQ(0, value *= 0);

                REQUIRE_EQ(value, 1 * value);
                REQUIRE_EQ(value, value * 1);
                REQUIRE_EQ(value, turbo::int128(1) *= value);
                REQUIRE_EQ(value, value *= 1);

                REQUIRE_EQ(-value, -1 * value);
                REQUIRE_EQ(-value, value * -1);
                REQUIRE_EQ(-value, turbo::int128(-1) *= value);
                REQUIRE_EQ(-value, value *= -1);
            }

            // Manually calculated random large value cases
            REQUIRE_EQ(turbo::make_int128(0xcd0efd3442219bb, 0xde47c05bcd9df6e1),
                       turbo::make_int128(0x7c6448, 0x3bc4285c47a9d253) * 0x1a6037537b);
            REQUIRE_EQ(-turbo::make_int128(0x1f8f149850b1e5e6, 0x1e50d6b52d272c3e),
                       -turbo::make_int128(0x23, 0x2e68a513ca1b8859) * 0xe5a434cd14866e);
            REQUIRE_EQ(-turbo::make_int128(0x55cae732029d1fce, 0xca6474b6423263e4),
                       0xa9b98a8ddf66bc * -turbo::make_int128(0x81, 0x672e58231e2469d7));
            REQUIRE_EQ(turbo::make_int128(0x19c8b7620b507dc4, 0xfec042b71a5f29a4),
                       -0x3e39341147 * -turbo::make_int128(0x6a14b2, 0x5ed34cca42327b3c));

            REQUIRE_EQ(turbo::make_int128(0xcd0efd3442219bb, 0xde47c05bcd9df6e1),
                       turbo::make_int128(0x7c6448, 0x3bc4285c47a9d253) *= 0x1a6037537b);
            REQUIRE_EQ(-turbo::make_int128(0x1f8f149850b1e5e6, 0x1e50d6b52d272c3e),
                       -turbo::make_int128(0x23, 0x2e68a513ca1b8859) *= 0xe5a434cd14866e);
            REQUIRE_EQ(-turbo::make_int128(0x55cae732029d1fce, 0xca6474b6423263e4),
                       turbo::int128(0xa9b98a8ddf66bc) *=
                               -turbo::make_int128(0x81, 0x672e58231e2469d7));
            REQUIRE_EQ(turbo::make_int128(0x19c8b7620b507dc4, 0xfec042b71a5f29a4),
                       turbo::int128(-0x3e39341147) *=
                               -turbo::make_int128(0x6a14b2, 0x5ed34cca42327b3c));
        }

        SUBCASE("DivisionAndModuloTest") {
            // Check against 64 bit division and modulo operators with a sample of
            // randomly generated pairs.
            std::pair<int64_t, int64_t> small_pairs[] = {
                    {0x15f2a64138,        0x67da05},
                    {0x5e56d194af43045f,  0xcf1543fb99},
                    {0x15e61ed052036a,    -0xc8e6},
                    {0x88125a341e85,      -0xd23fb77683},
                    {-0xc06e20,           0x5a},
                    {-0x4f100219aea3e85d, 0xdcc56cb4efe993},
                    {-0x168d629105,       -0xa7},
                    {-0x7b44e92f03ab2375, -0x6516},
            };
            for (const std::pair<int64_t, int64_t> &pair: small_pairs) {
                //SCOPED_TRACE(::testing::Message()
                //            << "pair = {" << pair.first << ", " << pair.second << '}');

                turbo::int128 dividend = pair.first;
                turbo::int128 divisor = pair.second;
                int64_t quotient = pair.first / pair.second;
                int64_t remainder = pair.first % pair.second;

                REQUIRE_EQ(quotient, dividend / divisor);
                REQUIRE_EQ(quotient, turbo::int128(dividend) /= divisor);
                REQUIRE_EQ(remainder, dividend % divisor);
                REQUIRE_EQ(remainder, turbo::int128(dividend) %= divisor);
            }

            // Test behavior with 0, 1, and -1 with a sample of randomly generated large
            // values.
            turbo::int128 values[] = {
                    turbo::make_int128(0x63d26ee688a962b2, 0x9e1411abda5c1d70),
                    turbo::make_int128(0x152f385159d6f986, 0xbf8d48ef63da395d),
                    -turbo::make_int128(0x3098d7567030038c, 0x14e7a8a098dc2164),
                    -turbo::make_int128(0x49a037aca35c809f, 0xa6a87525480ef330),
            };
            for (turbo::int128 value: values) {
                //SCOPED_TRACE(::testing::Message() << "value = " << value);

                REQUIRE_EQ(0, 0 / value);
                REQUIRE_EQ(0, turbo::int128(0) /= value);
                REQUIRE_EQ(0, 0 % value);
                REQUIRE_EQ(0, turbo::int128(0) %= value);

                REQUIRE_EQ(value, value / 1);
                REQUIRE_EQ(value, turbo::int128(value) /= 1);
                REQUIRE_EQ(0, value % 1);
                REQUIRE_EQ(0, turbo::int128(value) %= 1);

                REQUIRE_EQ(-value, value / -1);
                REQUIRE_EQ(-value, turbo::int128(value) /= -1);
                REQUIRE_EQ(0, value % -1);
                REQUIRE_EQ(0, turbo::int128(value) %= -1);
            }

            // Min and max values
            REQUIRE_EQ(0, turbo::int128_max() / turbo::int128_min());
            REQUIRE_EQ(turbo::int128_max(), turbo::int128_max() % turbo::int128_min());
            REQUIRE_EQ(-1, turbo::int128_min() / turbo::int128_max());
            REQUIRE_EQ(-1, turbo::int128_min() % turbo::int128_max());

            // Power of two division and modulo of random large dividends
            turbo::int128 positive_values[] = {
                    turbo::make_int128(0x21e1a1cc69574620, 0xe7ac447fab2fc869),
                    turbo::make_int128(0x32c2ff3ab89e66e8, 0x03379a613fd1ce74),
                    turbo::make_int128(0x6f32ca786184dcaf, 0x046f9c9ecb3a9ce1),
                    turbo::make_int128(0x1aeb469dd990e0ee, 0xda2740f243cd37eb),
            };
            for (turbo::int128 value: positive_values) {
                for (int i = 0; i < 127; ++i) {
                    //SCOPED_TRACE(::testing::Message()
                    //             << "value = " << value << "; i = " << i);
                    turbo::int128 power_of_two = turbo::int128(1) << i;

                    REQUIRE_EQ(value >> i, value / power_of_two);
                    REQUIRE_EQ(value >> i, turbo::int128(value) /= power_of_two);
                    REQUIRE_EQ(value & (power_of_two - 1), value % power_of_two);
                    REQUIRE_EQ(value & (power_of_two - 1),
                               turbo::int128(value) %= power_of_two);
                }
            }

            // Manually calculated cases with random large dividends
            struct DivisionModCase {
                turbo::int128 dividend;
                turbo::int128 divisor;
                turbo::int128 quotient;
                turbo::int128 remainder;
            };
            DivisionModCase manual_cases[] = {
                    {turbo::make_int128(0x6ada48d489007966, 0x3c9c5c98150d5d69),
                                                                                 turbo::make_int128(0x8bc308fb,
                                                                                                   0x8cb9cc9a3b803344),  0xc3b87e08,
                                                                                                                                                      turbo::make_int128(
                                                                                                                                                              0x1b7db5e1,
                                                                                                                                                              0xd9eca34b7af04b49)},
                    {turbo::make_int128(0xd6946511b5b, 0x4886c5c96546bf5f),
                                                                                 -turbo::make_int128(0x263b,
                                                                                                    0xfd516279efcfe2dc), -0x59cbabf0,
                                                                                                                                                      turbo::make_int128(
                                                                                                                                                              0x622,
                                                                                                                                                              0xf462909155651d1f)},
                    {-turbo::make_int128(0x33db734f9e8d1399, 0x8447ac92482bca4d), 0x37495078240,
                                                                                                                         -turbo::make_int128(
                                                                                                                                 0xf01f1,
                                                                                                                                 0xbc0368bf9a77eae8), -0x21a508f404d},
                    {-turbo::make_int128(0x13f837b409a07e7d, 0x7fc8e248a7d73560), -0x1b9f,
                                                                                                                         turbo::make_int128(
                                                                                                                                 0xb9157556d724,
                                                                                                                                 0xb14f635714d7563e), -0x1ade},
            };
            for (const DivisionModCase test_case: manual_cases) {
                REQUIRE_EQ(test_case.quotient, test_case.dividend / test_case.divisor);
                REQUIRE_EQ(test_case.quotient,
                           turbo::int128(test_case.dividend) /= test_case.divisor);
                REQUIRE_EQ(test_case.remainder, test_case.dividend % test_case.divisor);
                REQUIRE_EQ(test_case.remainder,
                           turbo::int128(test_case.dividend) %= test_case.divisor);
            }
        }

        SUBCASE("BitwiseLogicTest") {
            REQUIRE_EQ(turbo::int128(-1), ~turbo::int128(0));

            turbo::int128 values[]{
                    0, -1, 0xde400bee05c3ff6b, turbo::make_int128(0x7f32178dd81d634a, 0),
                    turbo::make_int128(0xaf539057055613a9, 0x7d104d7d946c2e4d)};
            for (turbo::int128 value: values) {
                REQUIRE_EQ(value, ~~value);

                REQUIRE_EQ(value, value | value);
                REQUIRE_EQ(value, value & value);
                REQUIRE_EQ(0, value ^ value);

                REQUIRE_EQ(value, turbo::int128(value) |= value);
                REQUIRE_EQ(value, turbo::int128(value) &= value);
                REQUIRE_EQ(0, turbo::int128(value) ^= value);

                REQUIRE_EQ(value, value | 0);
                REQUIRE_EQ(0, value & 0);
                REQUIRE_EQ(value, value ^ 0);

                REQUIRE_EQ(turbo::int128(-1), value | turbo::int128(-1));
                REQUIRE_EQ(value, value & turbo::int128(-1));
                REQUIRE_EQ(~value, value ^ turbo::int128(-1));
            }

            // small sample of randomly generated int64_t's
            std::pair<int64_t, int64_t> pairs64[]{
                    {0x7f86797f5e991af4, 0x1ee30494fb007c97},
                    {0x0b278282bacf01af, 0x58780e0a57a49e86},
                    {0x059f266ccb93a666, 0x3d5b731bae9286f5},
                    {0x63c0c4820f12108c, 0x58166713c12e1c3a},
                    {0x381488bb2ed2a66e, 0x2220a3eb76a3698c},
                    {0x2a0a0dfb81e06f21, 0x4b60585927f5523c},
                    {0x555b1c3a03698537, 0x25478cd19d8e53cb},
                    {0x4750f6f27d779225, 0x16397553c6ff05fc},
            };
            for (const std::pair<int64_t, int64_t> &pair: pairs64) {
                //SCOPED_TRACE(::testing::Message()
                //            << "pair = {" << pair.first << ", " << pair.second << '}');

                REQUIRE_EQ(turbo::make_int128(~pair.first, ~pair.second),
                           ~turbo::make_int128(pair.first, pair.second));

                REQUIRE_EQ(turbo::int128(pair.first & pair.second),
                           turbo::int128(pair.first) & turbo::int128(pair.second));
                REQUIRE_EQ(turbo::int128(pair.first | pair.second),
                           turbo::int128(pair.first) | turbo::int128(pair.second));
                REQUIRE_EQ(turbo::int128(pair.first ^ pair.second),
                           turbo::int128(pair.first) ^ turbo::int128(pair.second));

                REQUIRE_EQ(turbo::int128(pair.first & pair.second),
                           turbo::int128(pair.first) &= turbo::int128(pair.second));
                REQUIRE_EQ(turbo::int128(pair.first | pair.second),
                           turbo::int128(pair.first) |= turbo::int128(pair.second));
                REQUIRE_EQ(turbo::int128(pair.first ^ pair.second),
                           turbo::int128(pair.first) ^= turbo::int128(pair.second));

                REQUIRE_EQ(
                        turbo::make_int128(pair.first & pair.second, 0),
                        turbo::make_int128(pair.first, 0) & turbo::make_int128(pair.second, 0));
                REQUIRE_EQ(
                        turbo::make_int128(pair.first | pair.second, 0),
                        turbo::make_int128(pair.first, 0) | turbo::make_int128(pair.second, 0));
                REQUIRE_EQ(
                        turbo::make_int128(pair.first ^ pair.second, 0),
                        turbo::make_int128(pair.first, 0) ^ turbo::make_int128(pair.second, 0));

                REQUIRE_EQ(
                        turbo::make_int128(pair.first & pair.second, 0),
                        turbo::make_int128(pair.first, 0) &= turbo::make_int128(pair.second, 0));
                REQUIRE_EQ(
                        turbo::make_int128(pair.first | pair.second, 0),
                        turbo::make_int128(pair.first, 0) |= turbo::make_int128(pair.second, 0));
                REQUIRE_EQ(
                        turbo::make_int128(pair.first ^ pair.second, 0),
                        turbo::make_int128(pair.first, 0) ^= turbo::make_int128(pair.second, 0));
            }
        }

        SUBCASE("BitwiseShiftTest") {
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j <= i; ++j) {
                    // Left shift from j-th bit to i-th bit.
                    //SCOPED_TRACE(::testing::Message() << "i = " << i << "; j = " << j);
                    REQUIRE_EQ(uint64_t{1} << i, turbo::int128(uint64_t{1} << j) << (i - j));
                    REQUIRE_EQ(uint64_t{1} << i, turbo::int128(uint64_t{1} << j) <<= (i - j));
                }
            }
            for (int i = 0; i < 63; ++i) {
                for (int j = 0; j < 64; ++j) {
                    // Left shift from j-th bit to (i + 64)-th bit.
                    //SCOPED_TRACE(::testing::Message() << "i = " << i << "; j = " << j);
                    REQUIRE_EQ(turbo::make_int128(uint64_t{1} << i, 0),
                               turbo::int128(uint64_t{1} << j) << (i + 64 - j));
                    REQUIRE_EQ(turbo::make_int128(uint64_t{1} << i, 0),
                               turbo::int128(uint64_t{1} << j) <<= (i + 64 - j));
                }
                for (int j = 0; j <= i; ++j) {
                    // Left shift from (j + 64)-th bit to (i + 64)-th bit.
                    // SCOPED_TRACE(::testing::Message() << "i = " << i << "; j = " << j);
                    REQUIRE_EQ(turbo::make_int128(uint64_t{1} << i, 0),
                               turbo::make_int128(uint64_t{1} << j, 0) << (i - j));
                    REQUIRE_EQ(turbo::make_int128(uint64_t{1} << i, 0),
                               turbo::make_int128(uint64_t{1} << j, 0) <<= (i - j));
                }
            }

            for (int i = 0; i < 64; ++i) {
                for (int j = i; j < 64; ++j) {
                    // Right shift from j-th bit to i-th bit.
                    //SCOPED_TRACE(::testing::Message() << "i = " << i << "; j = " << j);
                    REQUIRE_EQ(uint64_t{1} << i, turbo::int128(uint64_t{1} << j) >> (j - i));
                    REQUIRE_EQ(uint64_t{1} << i, turbo::int128(uint64_t{1} << j) >>= (j - i));
                }
                for (int j = 0; j < 63; ++j) {
                    // Right shift from (j + 64)-th bit to i-th bit.
                    //SCOPED_TRACE(::testing::Message() << "i = " << i << "; j = " << j);
                    REQUIRE_EQ(uint64_t{1} << i,
                               turbo::make_int128(uint64_t{1} << j, 0) >> (j + 64 - i));
                    REQUIRE_EQ(uint64_t{1} << i,
                               turbo::make_int128(uint64_t{1} << j, 0) >>= (j + 64 - i));
                }
            }
            for (int i = 0; i < 63; ++i) {
                for (int j = i; j < 63; ++j) {
                    // Right shift from (j + 64)-th bit to (i + 64)-th bit.
                    // SCOPED_TRACE(::testing::Message() << "i = " << i << "; j = " << j);
                    REQUIRE_EQ(turbo::make_int128(uint64_t{1} << i, 0),
                               turbo::make_int128(uint64_t{1} << j, 0) >> (j - i));
                    REQUIRE_EQ(turbo::make_int128(uint64_t{1} << i, 0),
                               turbo::make_int128(uint64_t{1} << j, 0) >>= (j - i));
                }
            }
        }

        SUBCASE("NumericLimitsTest") {
            static_assert(std::numeric_limits<turbo::int128>::is_specialized, "");
            static_assert(std::numeric_limits<turbo::int128>::is_signed, "");
            static_assert(std::numeric_limits<turbo::int128>::is_integer, "");
            REQUIRE_EQ(static_cast<int>(127 * std::log10(2)),
                       std::numeric_limits<turbo::int128>::digits10);
            REQUIRE_EQ(turbo::int128_min(), std::numeric_limits<turbo::int128>::min());
            REQUIRE_EQ(turbo::int128_min(), std::numeric_limits<turbo::int128>::lowest());
            REQUIRE_EQ(turbo::int128_max(), std::numeric_limits<turbo::int128>::max());
        }
    }

    TEST_CASE_TEMPLATE_DEFINE("Int128IntegerTraitsTest", T, int128intConstructAssignTest) {
        static_assert(std::is_constructible<turbo::int128, T>::value,
                      "turbo::int128 must be constructible from T");
        static_assert(std::is_assignable<turbo::int128 &, T>::value,
                      "turbo::int128 must be assignable from T");
        static_assert(!std::is_assignable<T &, turbo::int128>::value,
                      "T must not be assignable from turbo::int128");
    }

    TEST_CASE_TEMPLATE_APPLY(int128intConstructAssignTest, IntTypes);

    TEST_CASE_TEMPLATE_DEFINE("Int128FloatTraitsTest", T, int128floatConstructAssignTest) {
        static_assert(std::is_constructible<turbo::int128, T>::value,
                      "turbo::int128 must be constructible from T");
        static_assert(!std::is_assignable<turbo::int128 &, T>::value,
                      "turbo::int128 must not be assignable from T");
        static_assert(!std::is_assignable<T &, turbo::int128>::value,
                      "T must not be assignable from turbo::int128");
    }

    TEST_CASE_TEMPLATE_APPLY(int128floatConstructAssignTest, FloatingPointTypes);


    TEST_CASE_TEMPLATE_DEFINE("Int128FloatConversionTest", T, intfConstructAndCastTest) {
        // Conversions where the floating point values should be exactly the same.
        // 0x9f5b is a randomly chosen small value.
        for (int i = 0; i < 110; ++i) {  // 110 = 126 - #bits in 0x9f5b
            //SCOPED_TRACE(::testing::Message() << "i = " << i);

            T float_value = std::ldexp(static_cast<T>(0x9f5b), i);
            turbo::int128 int_value = turbo::int128(0x9f5b) << i;

            REQUIRE_EQ(float_value, static_cast<T>(int_value));
            REQUIRE_EQ(-float_value, static_cast<T>(-int_value));
            REQUIRE_EQ(int_value, turbo::int128(float_value));
            REQUIRE_EQ(-int_value, turbo::int128(-float_value));
        }

        // Round trip conversions with a small sample of randomly generated uint64_t
        // values (less than int64_t max so that value * 2^64 fits into int128).
        uint64_t values[] = {0x6d4492c24fb86199, 0x26ead65e4cb359b5,
                             0x2c43407433ba3fd1, 0x3b574ec668df6b55,
                             0x1c750e55a29f4f0f};
        for (uint64_t value: values) {
            for (int i = 0; i <= 64; ++i) {
            //SCOPED_TRACE(::testing::Message()
            //<< "value = " << value << "; i = " << i);

                T fvalue = std::ldexp(static_cast<T>(value), i);
                REQUIRE_LT(std::abs(fvalue - static_cast<T>(turbo::int128(fvalue))), 0.01);
                REQUIRE_LT(std::abs(-fvalue - static_cast<T>(-turbo::int128(fvalue))), 0.01);
                REQUIRE_LT(std::abs(-fvalue - static_cast<T>(turbo::int128(-fvalue))), 0.01);
                REQUIRE_LT(std::abs(fvalue - static_cast<T>(-turbo::int128(-fvalue))), 0.01);
            }
        }

        // Round trip conversions with a small sample of random large positive values.
        turbo::int128 large_values[] = {
                turbo::make_int128(0x5b0640d96c7b3d9f, 0xb7a7189e51d18622),
                turbo::make_int128(0x34bed042c6f65270, 0x73b236570669a089),
                turbo::make_int128(0x43deba9e6da12724, 0xf7f0f83da686797d),
                turbo::make_int128(0x71e8d383be4e5589, 0x75c3f96fb00752b6)};
        for (turbo::int128 value: large_values) {
        // Make value have as many significant bits as can be represented by
        // the mantissa, also making sure the highest and lowest bit in the range
        // are set.
            value >>= (127 - std::numeric_limits<T>::digits);
            value |= turbo::int128(1) << (std::numeric_limits<T>::digits - 1);
            value |= 1;
            for (int i = 0; i < 127 - std::numeric_limits<T>::digits; ++i) {
                turbo::int128 int_value = value << i;
                REQUIRE_EQ(int_value,
                           static_cast<turbo::int128>(static_cast<T>(int_value)));
                REQUIRE_EQ(-int_value,
                           static_cast<turbo::int128>(static_cast<T>(-int_value)));
            }
        }

        // Small sample of checks that rounding is toward zero
        REQUIRE_EQ(0, turbo::int128(T(0.1)));
        REQUIRE_EQ(17, turbo::int128(T(17.8)));
        REQUIRE_EQ(0, turbo::int128(T(-0.8)));
        REQUIRE_EQ(-53, turbo::int128(T(-53.1)));
        REQUIRE_EQ(0, turbo::int128(T(0.5)));
        REQUIRE_EQ(0, turbo::int128(T(-0.5)));
        T just_lt_one = std::nexttoward(T(1), T(0));
        REQUIRE_EQ(0, turbo::int128(just_lt_one));
        T just_gt_minus_one = std::nexttoward(T(-1), T(0));
        REQUIRE_EQ(0, turbo::int128(just_gt_minus_one));

        // Check limits
        REQUIRE_LT(std::abs(std::ldexp(static_cast<T>(1), 127) -
                            static_cast<T>(turbo::int128_max())), 0.01);
        REQUIRE_LT(std::abs(-std::ldexp(static_cast<T>(1), 127) -
                            static_cast<T>(turbo::int128_min())), 0.01);
    }

    TEST_CASE_TEMPLATE_APPLY(intfConstructAndCastTest, FloatingPointTypes);
}  // namespace
#endif