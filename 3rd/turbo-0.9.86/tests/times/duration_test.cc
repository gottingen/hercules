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

#if defined(_MSC_VER)
#include <winsock2.h>  // for timeval
#endif

#include <chrono>  // NOLINT(build/c++11)
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <random>
#include <string>

#include "turbo/testing/test.h"
#include "turbo/times/time.h"

namespace {

    constexpr int64_t kint64max = std::numeric_limits<int64_t>::max();
    constexpr int64_t kint64min = std::numeric_limits<int64_t>::min();

// Approximates the given number of years. This is only used to make some test
// code more readable.
    turbo::Duration ApproxYears(int64_t n) { return turbo::Duration::hours(n) * 365 * 24; }


    TEST_CASE("Duration, ConstExpr") {
        constexpr turbo::Duration d0 = turbo::Duration::zero();
        static_assert(d0 == turbo::Duration::zero(), "Duration::zero()");
        constexpr turbo::Duration d1 = turbo::Duration::seconds(1);
        static_assert(d1 == turbo::Duration::seconds(1), "seconds(1)");
        static_assert(d1 != turbo::Duration::zero(), "seconds(1)");
        constexpr turbo::Duration d2 = turbo::Duration::infinite();
        static_assert(d2 == turbo::Duration::infinite(), "Duration::infinite()");
        static_assert(d2 != turbo::Duration::zero(), "Duration::infinite()");
    }

    TEST_CASE("Duration, ValueSemantics") {
        // If this compiles, the test passes.
        constexpr turbo::Duration a;      // Default construction
        constexpr turbo::Duration b = a;  // Copy construction
        constexpr turbo::Duration c(b);   // Copy construction (again)

        turbo::Duration d;
        d = c;  // Assignment
    }

    TEST_CASE("Duration, Factories") {
        constexpr turbo::Duration zero = turbo::Duration::zero();
        constexpr turbo::Duration nano = turbo::Duration::nanoseconds(1);
        constexpr turbo::Duration micro = turbo::Duration::microseconds(1);
        constexpr turbo::Duration milli = turbo::Duration::milliseconds(1);
        constexpr turbo::Duration sec = turbo::Duration::seconds(1);
        constexpr turbo::Duration min = turbo::Duration::minutes(1);
        constexpr turbo::Duration hour = turbo::Duration::hours(1);

        REQUIRE_EQ(zero, turbo::Duration());
        REQUIRE_EQ(zero, turbo::Duration::seconds(0));
        REQUIRE_EQ(nano, turbo::Duration::nanoseconds(1));
        REQUIRE_EQ(micro, turbo::Duration::nanoseconds(1000));
        REQUIRE_EQ(milli, turbo::Duration::microseconds(1000));
        REQUIRE_EQ(sec, turbo::Duration::milliseconds(1000));
        REQUIRE_EQ(min, turbo::Duration::seconds(60));
        REQUIRE_EQ(hour, turbo::Duration::minutes(60));

        // Tests factory limits
        const turbo::Duration inf = turbo::Duration::infinite();

        REQUIRE_GT(inf, turbo::Duration::seconds(kint64max));
        REQUIRE_LT(-inf, turbo::Duration::seconds(kint64min));
        REQUIRE_LT(-inf, turbo::Duration::seconds(-kint64max));

        REQUIRE_EQ(inf, turbo::Duration::minutes(kint64max));
        REQUIRE_EQ(-inf, turbo::Duration::minutes(kint64min));
        REQUIRE_EQ(-inf, turbo::Duration::minutes(-kint64max));
        REQUIRE_GT(inf, turbo::Duration::minutes(kint64max / 60));
        REQUIRE_LT(-inf, turbo::Duration::minutes(kint64min / 60));
        REQUIRE_LT(-inf, turbo::Duration::minutes(-kint64max / 60));

        REQUIRE_EQ(inf, turbo::Duration::hours(kint64max));
        REQUIRE_EQ(-inf, turbo::Duration::hours(kint64min));
        REQUIRE_EQ(-inf, turbo::Duration::hours(-kint64max));
        REQUIRE_GT(inf, turbo::Duration::hours(kint64max / 3600));
        REQUIRE_LT(-inf, turbo::Duration::hours(kint64min / 3600));
        REQUIRE_LT(-inf, turbo::Duration::hours(-kint64max / 3600));
    }

    TEST_CASE("Duration, ToConversion") {
#define TEST_DURATION_CONVERSION(UNIT)                                  \
  do {                                                                  \
    const turbo::Duration d = turbo::Duration::UNIT(1.5);                           \
    constexpr turbo::Duration z = turbo::Duration::zero();                  \
    constexpr turbo::Duration inf = turbo::Duration::infinite();            \
    constexpr double dbl_inf = std::numeric_limits<double>::infinity(); \
    REQUIRE_EQ(kint64min, (-inf).to_##UNIT());                    \
    REQUIRE_EQ(-1, (-d).to_##UNIT());                             \
    REQUIRE_EQ(0, (z).to_##UNIT());                               \
    REQUIRE_EQ(1, (d).to_##UNIT());                               \
    REQUIRE_EQ(kint64max, (inf).to_##UNIT());                     \
    REQUIRE_EQ(-dbl_inf, (-inf).to_##UNIT<double>());                    \
    REQUIRE_EQ(-1.5, (-d).to_##UNIT<double>());                          \
    REQUIRE_EQ(0, (z).to_##UNIT<double>());                              \
    REQUIRE_EQ(1.5, (d).to_##UNIT<double>());                            \
    REQUIRE_EQ(dbl_inf, (inf).to_##UNIT<double>());                      \
  } while (0)

        TEST_DURATION_CONVERSION(nanoseconds);
        TEST_DURATION_CONVERSION(microseconds);
        TEST_DURATION_CONVERSION(milliseconds);
        TEST_DURATION_CONVERSION(seconds);
        TEST_DURATION_CONVERSION(minutes);
        TEST_DURATION_CONVERSION(hours);

#undef TEST_DURATION_CONVERSION
    }

    template<int64_t N>
    void TestToConversion() {
        constexpr turbo::Duration nano = turbo::Duration::nanoseconds(N);
        REQUIRE_EQ(N, (nano).to_nanoseconds());
        REQUIRE_EQ(0, (nano).to_microseconds());
        REQUIRE_EQ(0, (nano).to_milliseconds());
        REQUIRE_EQ(0, (nano).to_seconds());
        REQUIRE_EQ(0, (nano).to_minutes());
        REQUIRE_EQ(0, (nano).to_hours());
        const turbo::Duration micro = turbo::Duration::microseconds(N);
        REQUIRE_EQ(N * 1000, (micro).to_nanoseconds());
        REQUIRE_EQ(N, (micro).to_microseconds());
        REQUIRE_EQ(0, (micro).to_milliseconds());
        REQUIRE_EQ(0, (micro).to_seconds());
        REQUIRE_EQ(0, (micro).to_minutes());
        REQUIRE_EQ(0, (micro).to_hours());
        const turbo::Duration milli = turbo::Duration::milliseconds(N);
        REQUIRE_EQ(N * 1000 * 1000, (milli).to_nanoseconds());
        REQUIRE_EQ(N * 1000, (milli).to_microseconds());
        REQUIRE_EQ(N, (milli).to_milliseconds());
        REQUIRE_EQ(0, (milli).to_seconds());
        REQUIRE_EQ(0, (milli).to_minutes());
        REQUIRE_EQ(0, (milli).to_hours());
        const turbo::Duration sec = turbo::Duration::seconds(N);
        REQUIRE_EQ(N * 1000 * 1000 * 1000, (sec).to_nanoseconds());
        REQUIRE_EQ(N * 1000 * 1000, (sec).to_microseconds());
        REQUIRE_EQ(N * 1000,(sec).to_milliseconds());
        REQUIRE_EQ(N, (sec).to_seconds());
        REQUIRE_EQ(0, (sec).to_minutes());
        REQUIRE_EQ(0, (sec).to_hours());
        const turbo::Duration min = turbo::Duration::minutes(N);
        REQUIRE_EQ(N * 60 * 1000 * 1000 * 1000, (min).to_nanoseconds());
        REQUIRE_EQ(N * 60 * 1000 * 1000, (min).to_microseconds());
        REQUIRE_EQ(N * 60 * 1000, (min).to_milliseconds());
        REQUIRE_EQ(N * 60, (min).to_seconds());
        REQUIRE_EQ(N, (min).to_minutes());
        REQUIRE_EQ(0, (min).to_hours());
        const turbo::Duration hour = turbo::Duration::hours(N);
        REQUIRE_EQ(N * 60 * 60 * 1000 * 1000 * 1000, (hour).to_nanoseconds());
        REQUIRE_EQ(N * 60 * 60 * 1000 * 1000, (hour).to_microseconds());
        REQUIRE_EQ(N * 60 * 60 * 1000, (hour).to_milliseconds());
        REQUIRE_EQ(N * 60 * 60, (hour).to_seconds());
        REQUIRE_EQ(N * 60, (hour).to_minutes());
        REQUIRE_EQ(N, (hour).to_hours());
    }

    TEST_CASE("Duration, ToConversionDeprecated") {
        TestToConversion<43>();
        TestToConversion<1>();
        TestToConversion<0>();
        TestToConversion<-1>();
        TestToConversion<-43>();
    }

    template<int64_t N>
    void TestFromChronoBasicEquality() {
        using std::chrono::nanoseconds;
        using std::chrono::microseconds;
        using std::chrono::milliseconds;
        using std::chrono::seconds;
        using std::chrono::minutes;
        using std::chrono::hours;

        static_assert(turbo::Duration::nanoseconds(N) == turbo::Duration::from_chrono(nanoseconds(N)), "");
        static_assert(turbo::Duration::microseconds(N) == turbo::Duration::from_chrono(microseconds(N)), "");
        static_assert(turbo::Duration::milliseconds(N) == turbo::Duration::from_chrono(milliseconds(N)), "");
        static_assert(turbo::Duration::seconds(N) == turbo::Duration::from_chrono(seconds(N)), "");
        static_assert(turbo::Duration::minutes(N) == turbo::Duration::from_chrono(minutes(N)), "");
        static_assert(turbo::Duration::hours(N) == turbo::Duration::from_chrono(hours(N)), "");
    }

    TEST_CASE("Duration, from_chrono") {
        TestFromChronoBasicEquality<-123>();
        TestFromChronoBasicEquality<-1>();
        TestFromChronoBasicEquality<0>();
        TestFromChronoBasicEquality<1>();
        TestFromChronoBasicEquality<123>();

        // minutes (might, depending on the platform) saturate at +inf.
        const auto chrono_minutes_max = std::chrono::minutes::max();
        const auto minutes_max = turbo::Duration::from_chrono(chrono_minutes_max);
        const int64_t minutes_max_count = chrono_minutes_max.count();
        if (minutes_max_count > kint64max / 60) {
            REQUIRE_EQ(turbo::Duration::infinite(), minutes_max);
        } else {
            REQUIRE_EQ(turbo::Duration::minutes(minutes_max_count), minutes_max);
        }

        // minutes (might, depending on the platform) saturate at -inf.
        const auto chrono_minutes_min = std::chrono::minutes::min();
        const auto minutes_min = turbo::Duration::from_chrono(chrono_minutes_min);
        const int64_t minutes_min_count = chrono_minutes_min.count();
        if (minutes_min_count < kint64min / 60) {
            REQUIRE_EQ(-turbo::Duration::infinite(), minutes_min);
        } else {
            REQUIRE_EQ(turbo::Duration::minutes(minutes_min_count), minutes_min);
        }

        // hours (might, depending on the platform) saturate at +inf.
        const auto chrono_hours_max = std::chrono::hours::max();
        const auto hours_max = turbo::Duration::from_chrono(chrono_hours_max);
        const int64_t hours_max_count = chrono_hours_max.count();
        if (hours_max_count > kint64max / 3600) {
            REQUIRE_EQ(turbo::Duration::infinite(), hours_max);
        } else {
            REQUIRE_EQ(turbo::Duration::hours(hours_max_count), hours_max);
        }

        // hours (might, depending on the platform) saturate at -inf.
        const auto chrono_hours_min = std::chrono::hours::min();
        const auto hours_min = turbo::Duration::from_chrono(chrono_hours_min);
        const int64_t hours_min_count = chrono_hours_min.count();
        if (hours_min_count < kint64min / 3600) {
            REQUIRE_EQ(-turbo::Duration::infinite(), hours_min);
        } else {
            REQUIRE_EQ(turbo::Duration::hours(hours_min_count), hours_min);
        }
    }

    template<int64_t N>
    void TestToChrono() {
        using std::chrono::nanoseconds;
        using std::chrono::microseconds;
        using std::chrono::milliseconds;
        using std::chrono::seconds;
        using std::chrono::minutes;
        using std::chrono::hours;

        REQUIRE_EQ(nanoseconds(N), (turbo::Duration::nanoseconds(N)).to_chrono_nanoseconds());
        REQUIRE_EQ(microseconds(N), (turbo::Duration::microseconds(N)).to_chrono_microseconds());
        REQUIRE_EQ(milliseconds(N),(turbo::Duration::milliseconds(N)).to_chrono_milliseconds());
        REQUIRE_EQ(seconds(N), (turbo::Duration::seconds(N)).to_chrono_seconds());

        constexpr auto turbo_minutes = turbo::Duration::minutes(N);
        auto chrono_minutes = minutes(N);
        if (turbo_minutes == -turbo::Duration::infinite()) {
            chrono_minutes = minutes::min();
        } else if (turbo_minutes == turbo::Duration::infinite()) {
            chrono_minutes = minutes::max();
        }
        REQUIRE_EQ(chrono_minutes, (turbo_minutes).to_chrono_minutes());

        constexpr auto turbo_hours = turbo::Duration::hours(N);
        auto chrono_hours = hours(N);
        if (turbo_hours == -turbo::Duration::infinite()) {
            chrono_hours = hours::min();
        } else if (turbo_hours == turbo::Duration::infinite()) {
            chrono_hours = hours::max();
        }
        REQUIRE_EQ(chrono_hours, (turbo_hours).to_chrono_hours());
    }

    TEST_CASE("Duration, ToChrono") {
        using std::chrono::nanoseconds;
        using std::chrono::microseconds;
        using std::chrono::milliseconds;
        using std::chrono::seconds;
        using std::chrono::minutes;
        using std::chrono::hours;

        TestToChrono<kint64min>();
        TestToChrono<-1>();
        TestToChrono<0>();
        TestToChrono<1>();
        TestToChrono<kint64max>();

        // Verify truncation toward zero.
        const auto tick = turbo::Duration::nanoseconds(1) / 4;
        REQUIRE_EQ(nanoseconds(0), (tick).to_chrono_nanoseconds());
        REQUIRE_EQ(nanoseconds(0), (-tick).to_chrono_nanoseconds());
        REQUIRE_EQ(microseconds(0), (tick).to_chrono_microseconds());
        REQUIRE_EQ(microseconds(0), (-tick).to_chrono_microseconds());
        REQUIRE_EQ(milliseconds(0), (tick).to_chrono_milliseconds());
        REQUIRE_EQ(milliseconds(0), (-tick).to_chrono_milliseconds());
        REQUIRE_EQ(seconds(0), (tick).to_chrono_seconds());
        REQUIRE_EQ(seconds(0), (-tick).to_chrono_seconds());
        REQUIRE_EQ(minutes(0), (tick).to_chrono_minutes());
        REQUIRE_EQ(minutes(0),(-tick).to_chrono_minutes());
        REQUIRE_EQ(hours(0), (tick).to_chrono_hours());
        REQUIRE_EQ(hours(0),(-tick).to_chrono_hours());

        // Verifies +/- infinity saturation at max/min.
        constexpr auto inf = turbo::Duration::infinite();
        REQUIRE_EQ(nanoseconds::min(),(-inf).to_chrono_nanoseconds());
        REQUIRE_EQ(nanoseconds::max(), (inf).to_chrono_nanoseconds());
        REQUIRE_EQ(microseconds::min(), (-inf).to_chrono_microseconds());
        REQUIRE_EQ(microseconds::max(), (inf).to_chrono_microseconds());
        REQUIRE_EQ(milliseconds::min(), (-inf).to_chrono_milliseconds());
        REQUIRE_EQ(milliseconds::max(), (inf).to_chrono_milliseconds());
        REQUIRE_EQ(seconds::min(), (-inf).to_chrono_seconds());
        REQUIRE_EQ(seconds::max(), (inf).to_chrono_seconds());
        REQUIRE_EQ(minutes::min(),(-inf).to_chrono_minutes());
        REQUIRE_EQ(minutes::max(), (inf).to_chrono_minutes());
        REQUIRE_EQ(hours::min(), (-inf).to_chrono_hours());
        REQUIRE_EQ(hours::max(), (inf).to_chrono_hours());
    }

    TEST_CASE("Duration, FactoryOverloads") {

        enum E {
            kOne = 1
        };
#define TEST_FACTORY_OVERLOADS(NAME)                                          \
  REQUIRE_EQ(1, NAME(kOne) / NAME(kOne));                                      \
  REQUIRE_EQ(1, NAME(static_cast<int8_t>(1)) / NAME(1));                       \
  REQUIRE_EQ(1, NAME(static_cast<int16_t>(1)) / NAME(1));                      \
  REQUIRE_EQ(1, NAME(static_cast<int32_t>(1)) / NAME(1));                      \
  REQUIRE_EQ(1, NAME(static_cast<int64_t>(1)) / NAME(1));                      \
  REQUIRE_EQ(1, NAME(static_cast<uint8_t>(1)) / NAME(1));                      \
  REQUIRE_EQ(1, NAME(static_cast<uint16_t>(1)) / NAME(1));                     \
  REQUIRE_EQ(1, NAME(static_cast<uint32_t>(1)) / NAME(1));                     \
  REQUIRE_EQ(1, NAME(static_cast<uint64_t>(1)) / NAME(1));                     \
  REQUIRE_EQ(NAME(1) / 2, NAME(static_cast<float>(0.5)));                      \
  REQUIRE_EQ(NAME(1) / 2, NAME(static_cast<double>(0.5)));                     \
  REQUIRE_EQ(1.5, (NAME(static_cast<float>(1.5))).safe_float_mod(NAME(1))); \
  REQUIRE_EQ(1.5, (NAME(static_cast<double>(1.5))).safe_float_mod( NAME(1)));

        TEST_FACTORY_OVERLOADS(turbo::Duration::nanoseconds);
        TEST_FACTORY_OVERLOADS(turbo::Duration::microseconds);
        TEST_FACTORY_OVERLOADS(turbo::Duration::milliseconds);
        TEST_FACTORY_OVERLOADS(turbo::Duration::seconds);
        TEST_FACTORY_OVERLOADS(turbo::Duration::minutes);
        TEST_FACTORY_OVERLOADS(turbo::Duration::hours);

#undef TEST_FACTORY_OVERLOADS

        REQUIRE_EQ(turbo::Duration::milliseconds(1500), turbo::Duration::seconds(1.5));
        REQUIRE_LT(turbo::Duration::nanoseconds(1), turbo::Duration::nanoseconds(1.5));
        REQUIRE_GT(turbo::Duration::nanoseconds(2), turbo::Duration::nanoseconds(1.5));

        const double dbl_inf = std::numeric_limits<double>::infinity();
        REQUIRE_EQ(turbo::Duration::infinite(), turbo::Duration::nanoseconds(dbl_inf));
        REQUIRE_EQ(turbo::Duration::infinite(), turbo::Duration::microseconds(dbl_inf));
        REQUIRE_EQ(turbo::Duration::infinite(), turbo::Duration::milliseconds(dbl_inf));
        REQUIRE_EQ(turbo::Duration::infinite(), turbo::Duration::seconds(dbl_inf));
        REQUIRE_EQ(turbo::Duration::infinite(), turbo::Duration::minutes(dbl_inf));
        REQUIRE_EQ(turbo::Duration::infinite(), turbo::Duration::hours(dbl_inf));
        REQUIRE_EQ(-turbo::Duration::infinite(), turbo::Duration::nanoseconds(-dbl_inf));
        REQUIRE_EQ(-turbo::Duration::infinite(), turbo::Duration::microseconds(-dbl_inf));
        REQUIRE_EQ(-turbo::Duration::infinite(), turbo::Duration::milliseconds(-dbl_inf));
        REQUIRE_EQ(-turbo::Duration::infinite(), turbo::Duration::seconds(-dbl_inf));
        REQUIRE_EQ(-turbo::Duration::infinite(), turbo::Duration::minutes(-dbl_inf));
        REQUIRE_EQ(-turbo::Duration::infinite(), turbo::Duration::hours(-dbl_inf));
    }

    TEST_CASE("Duration, InfinityExamples") {
        // These examples are used in the documentation in time.h. They are
        // written so that they can be copy-n-pasted easily.

        constexpr turbo::Duration inf = turbo::Duration::infinite();
        constexpr turbo::Duration d = turbo::Duration::seconds(1);  // Any finite duration

        REQUIRE_EQ(inf , inf + inf);
        REQUIRE_EQ(inf , inf + d);
        REQUIRE_EQ(inf , inf - inf);
        REQUIRE_EQ(-inf , d - inf);

        REQUIRE_EQ(inf , d * 1e100);
        REQUIRE_EQ(0 , d / inf);  // NOLINT(readability/check)

        // Division by zero returns infinity, or kint64min/MAX where necessary.
        REQUIRE_EQ(inf , d / 0);
        REQUIRE_EQ(kint64max , d / turbo::Duration::zero());
    }

    TEST_CASE("Duration, InfinityComparison") {
        const turbo::Duration inf = turbo::Duration::infinite();
        const turbo::Duration any_dur = turbo::Duration::seconds(1);

        // Equality
        REQUIRE_EQ(inf, inf);
        REQUIRE_EQ(-inf, -inf);
        REQUIRE_NE(inf, -inf);
        REQUIRE_NE(any_dur, inf);
        REQUIRE_NE(any_dur, -inf);

        // Relational
        REQUIRE_GT(inf, any_dur);
        REQUIRE_LT(-inf, any_dur);
        REQUIRE_LT(-inf, inf);
        REQUIRE_GT(inf, -inf);
    }

    TEST_CASE("Duration, InfinityAddition") {
        const turbo::Duration sec_max = turbo::Duration::seconds(kint64max);
        const turbo::Duration sec_min = turbo::Duration::seconds(kint64min);
        const turbo::Duration any_dur = turbo::Duration::seconds(1);
        const turbo::Duration inf = turbo::Duration::infinite();

        // Addition
        REQUIRE_EQ(inf, inf + inf);
        REQUIRE_EQ(inf, inf + -inf);
        REQUIRE_EQ(-inf, -inf + inf);
        REQUIRE_EQ(-inf, -inf + -inf);

        REQUIRE_EQ(inf, inf + any_dur);
        REQUIRE_EQ(inf, any_dur + inf);
        REQUIRE_EQ(-inf, -inf + any_dur);
        REQUIRE_EQ(-inf, any_dur + -inf);

        // Interesting case
        turbo::Duration almost_inf = sec_max + turbo::Duration::nanoseconds(999999999);
        REQUIRE_GT(inf, almost_inf);
        almost_inf += -turbo::Duration::nanoseconds(999999999);
        REQUIRE_GT(inf, almost_inf);

        // Addition overflow/underflow
        REQUIRE_EQ(inf, sec_max + turbo::Duration::seconds(1));
        REQUIRE_EQ(inf, sec_max + sec_max);
        REQUIRE_EQ(-inf, sec_min + -turbo::Duration::seconds(1));
        REQUIRE_EQ(-inf, sec_min + -sec_max);

        // For reference: IEEE 754 behavior
        const double dbl_inf = std::numeric_limits<double>::infinity();
        REQUIRE(std::isinf(dbl_inf + dbl_inf));
        REQUIRE(std::isnan(dbl_inf + -dbl_inf));  // We return inf
        REQUIRE(std::isnan(-dbl_inf + dbl_inf));  // We return inf
        REQUIRE(std::isinf(-dbl_inf + -dbl_inf));
    }

    TEST_CASE("Duration, InfinitySubtraction") {
        const turbo::Duration sec_max = turbo::Duration::seconds(kint64max);
        const turbo::Duration sec_min = turbo::Duration::seconds(kint64min);
        const turbo::Duration any_dur = turbo::Duration::seconds(1);
        const turbo::Duration inf = turbo::Duration::infinite();

        // Subtraction
        REQUIRE_EQ(inf, inf - inf);
        REQUIRE_EQ(inf, inf - -inf);
        REQUIRE_EQ(-inf, -inf - inf);
        REQUIRE_EQ(-inf, -inf - -inf);

        REQUIRE_EQ(inf, inf - any_dur);
        REQUIRE_EQ(-inf, any_dur - inf);
        REQUIRE_EQ(-inf, -inf - any_dur);
        REQUIRE_EQ(inf, any_dur - -inf);

        // Subtraction overflow/underflow
        REQUIRE_EQ(inf, sec_max - -turbo::Duration::seconds(1));
        REQUIRE_EQ(inf, sec_max - -sec_max);
        REQUIRE_EQ(-inf, sec_min - turbo::Duration::seconds(1));
        REQUIRE_EQ(-inf, sec_min - sec_max);

        // Interesting case
        turbo::Duration almost_neg_inf = sec_min;
        REQUIRE_LT(-inf, almost_neg_inf);
        almost_neg_inf -= -turbo::Duration::nanoseconds(1);
        REQUIRE_LT(-inf, almost_neg_inf);

        // For reference: IEEE 754 behavior
        const double dbl_inf = std::numeric_limits<double>::infinity();
        REQUIRE(std::isnan(dbl_inf - dbl_inf));  // We return inf
        REQUIRE(std::isinf(dbl_inf - -dbl_inf));
        REQUIRE(std::isinf(-dbl_inf - dbl_inf));
        REQUIRE(std::isnan(-dbl_inf - -dbl_inf));  // We return inf
    }

    TEST_CASE("Duration, InfinityMultiplication") {
        const turbo::Duration sec_max = turbo::Duration::seconds(kint64max);
        const turbo::Duration sec_min = turbo::Duration::seconds(kint64min);
        const turbo::Duration inf = turbo::Duration::infinite();

#define TEST_INF_MUL_WITH_TYPE(T)                                     \
  REQUIRE_EQ(inf, inf * static_cast<T>(2));                            \
  REQUIRE_EQ(-inf, inf * static_cast<T>(-2));                          \
  REQUIRE_EQ(-inf, -inf * static_cast<T>(2));                          \
  REQUIRE_EQ(inf, -inf * static_cast<T>(-2));                          \
  REQUIRE_EQ(inf, inf * static_cast<T>(0));                            \
  REQUIRE_EQ(-inf, -inf * static_cast<T>(0));                          \
  REQUIRE_EQ(inf, sec_max * static_cast<T>(2));                        \
  REQUIRE_EQ(inf, sec_min * static_cast<T>(-2));                       \
  REQUIRE_EQ(inf, (sec_max / static_cast<T>(2)) * static_cast<T>(3));  \
  REQUIRE_EQ(-inf, sec_max * static_cast<T>(-2));                      \
  REQUIRE_EQ(-inf, sec_min * static_cast<T>(2));                       \
  REQUIRE_EQ(-inf, (sec_min / static_cast<T>(2)) * static_cast<T>(3));

        TEST_INF_MUL_WITH_TYPE(int64_t);  // NOLINT(readability/function)
        TEST_INF_MUL_WITH_TYPE(double);   // NOLINT(readability/function)

#undef TEST_INF_MUL_WITH_TYPE

        const double dbl_inf = std::numeric_limits<double>::infinity();
        REQUIRE_EQ(inf, inf * dbl_inf);
        REQUIRE_EQ(-inf, -inf * dbl_inf);
        REQUIRE_EQ(-inf, inf * -dbl_inf);
        REQUIRE_EQ(inf, -inf * -dbl_inf);

        const turbo::Duration any_dur = turbo::Duration::seconds(1);
        REQUIRE_EQ(inf, any_dur * dbl_inf);
        REQUIRE_EQ(-inf, -any_dur * dbl_inf);
        REQUIRE_EQ(-inf, any_dur * -dbl_inf);
        REQUIRE_EQ(inf, -any_dur * -dbl_inf);

        // Fixed-point multiplication will produce a finite value, whereas floating
        // point fuzziness will overflow to inf.
        REQUIRE_NE(turbo::Duration::infinite(), turbo::Duration::seconds(1) * kint64max);
        REQUIRE_EQ(inf, turbo::Duration::seconds(1) * static_cast<double>(kint64max));
        REQUIRE_NE(-turbo::Duration::infinite(), turbo::Duration::seconds(1) * kint64min);
        REQUIRE_EQ(-inf, turbo::Duration::seconds(1) * static_cast<double>(kint64min));

        // Note that sec_max * or / by 1.0 overflows to inf due to the 53-bit
        // limitations of double.
        REQUIRE_NE(inf, sec_max);
        REQUIRE_NE(inf, sec_max / 1);
        REQUIRE_EQ(inf, sec_max / 1.0);
        REQUIRE_NE(inf, sec_max * 1);
        REQUIRE_EQ(inf, sec_max * 1.0);
    }

    TEST_CASE("Duration, InfinityDivision") {
        const turbo::Duration sec_max = turbo::Duration::seconds(kint64max);
        const turbo::Duration sec_min = turbo::Duration::seconds(kint64min);
        const turbo::Duration inf = turbo::Duration::infinite();

        // Division of Duration by a double
#define TEST_INF_DIV_WITH_TYPE(T)            \
  REQUIRE_EQ(inf, inf / static_cast<T>(2));   \
  REQUIRE_EQ(-inf, inf / static_cast<T>(-2)); \
  REQUIRE_EQ(-inf, -inf / static_cast<T>(2)); \
  REQUIRE_EQ(inf, -inf / static_cast<T>(-2));

        TEST_INF_DIV_WITH_TYPE(int64_t);  // NOLINT(readability/function)
        TEST_INF_DIV_WITH_TYPE(double);   // NOLINT(readability/function)

#undef TEST_INF_DIV_WITH_TYPE

        // Division of Duration by a double overflow/underflow
        REQUIRE_EQ(inf, sec_max / 0.5);
        REQUIRE_EQ(inf, sec_min / -0.5);
        REQUIRE_EQ(inf, ((sec_max / 0.5) + turbo::Duration::seconds(1)) / 0.5);
        REQUIRE_EQ(-inf, sec_max / -0.5);
        REQUIRE_EQ(-inf, sec_min / 0.5);
        REQUIRE_EQ(-inf, ((sec_min / 0.5) - turbo::Duration::seconds(1)) / 0.5);

        const double dbl_inf = std::numeric_limits<double>::infinity();
        REQUIRE_EQ(inf, inf / dbl_inf);
        REQUIRE_EQ(-inf, inf / -dbl_inf);
        REQUIRE_EQ(-inf, -inf / dbl_inf);
        REQUIRE_EQ(inf, -inf / -dbl_inf);

        const turbo::Duration any_dur = turbo::Duration::seconds(1);
        REQUIRE_EQ(turbo::Duration::zero(), any_dur / dbl_inf);
        REQUIRE_EQ(turbo::Duration::zero(), any_dur / -dbl_inf);
        REQUIRE_EQ(turbo::Duration::zero(), -any_dur / dbl_inf);
        REQUIRE_EQ(turbo::Duration::zero(), -any_dur / -dbl_inf);
    }

    TEST_CASE("Duration, InfinityModulus") {
        const turbo::Duration sec_max = turbo::Duration::seconds(kint64max);
        const turbo::Duration any_dur = turbo::Duration::seconds(1);
        const turbo::Duration inf = turbo::Duration::infinite();

        REQUIRE_EQ(inf, inf % inf);
        REQUIRE_EQ(inf, inf % -inf);
        REQUIRE_EQ(-inf, -inf % -inf);
        REQUIRE_EQ(-inf, -inf % inf);

        REQUIRE_EQ(any_dur, any_dur % inf);
        REQUIRE_EQ(any_dur, any_dur % -inf);
        REQUIRE_EQ(-any_dur, -any_dur % inf);
        REQUIRE_EQ(-any_dur, -any_dur % -inf);

        REQUIRE_EQ(inf, inf % -any_dur);
        REQUIRE_EQ(inf, inf % any_dur);
        REQUIRE_EQ(-inf, -inf % -any_dur);
        REQUIRE_EQ(-inf, -inf % any_dur);

        // Remainder isn't affected by overflow.
        REQUIRE_EQ(turbo::Duration::zero(), sec_max % turbo::Duration::seconds(1));
        REQUIRE_EQ(turbo::Duration::zero(), sec_max % turbo::Duration::milliseconds(1));
        REQUIRE_EQ(turbo::Duration::zero(), sec_max % turbo::Duration::microseconds(1));
        REQUIRE_EQ(turbo::Duration::zero(), sec_max % turbo::Duration::nanoseconds(1));
        REQUIRE_EQ(turbo::Duration::zero(), sec_max % turbo::Duration::nanoseconds(1) / 4);
    }

    TEST_CASE("Duration, InfinityIDiv") {
        const turbo::Duration sec_max = turbo::Duration::seconds(kint64max);
        const turbo::Duration any_dur = turbo::Duration::seconds(1);
        const turbo::Duration inf = turbo::Duration::infinite();
        const double dbl_inf = std::numeric_limits<double>::infinity();

        // safe_int_mod (int64_t return value + a remainer)
        turbo::Duration rem = turbo::Duration::zero();
        REQUIRE_EQ(kint64max, turbo::safe_int_mod(inf, inf, &rem));
        REQUIRE_EQ(inf, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(kint64max, turbo::safe_int_mod(-inf, -inf, &rem));
        REQUIRE_EQ(-inf, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(kint64max, turbo::safe_int_mod(inf, any_dur, &rem));
        REQUIRE_EQ(inf, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(0, turbo::safe_int_mod(any_dur, inf, &rem));
        REQUIRE_EQ(any_dur, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(kint64max, turbo::safe_int_mod(-inf, -any_dur, &rem));
        REQUIRE_EQ(-inf, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(0, turbo::safe_int_mod(-any_dur, -inf, &rem));
        REQUIRE_EQ(-any_dur, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(kint64min, turbo::safe_int_mod(-inf, inf, &rem));
        REQUIRE_EQ(-inf, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(kint64min, turbo::safe_int_mod(inf, -inf, &rem));
        REQUIRE_EQ(inf, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(kint64min, turbo::safe_int_mod(-inf, any_dur, &rem));
        REQUIRE_EQ(-inf, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(0, turbo::safe_int_mod(-any_dur, inf, &rem));
        REQUIRE_EQ(-any_dur, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(kint64min, turbo::safe_int_mod(inf, -any_dur, &rem));
        REQUIRE_EQ(inf, rem);

        rem = turbo::Duration::zero();
        REQUIRE_EQ(0, turbo::safe_int_mod(any_dur, -inf, &rem));
        REQUIRE_EQ(any_dur, rem);

        // safe_int_mod overflow/underflow
        rem = any_dur;
        REQUIRE_EQ(kint64max,
                   turbo::safe_int_mod(sec_max, turbo::Duration::nanoseconds(1) / 4, &rem));
        REQUIRE_EQ(sec_max - turbo::Duration::nanoseconds(kint64max) / 4, rem);

        rem = any_dur;
        REQUIRE_EQ(kint64max,
                   turbo::safe_int_mod(sec_max, turbo::Duration::milliseconds(1), &rem));
        REQUIRE_EQ(sec_max - turbo::Duration::milliseconds(kint64max), rem);

        rem = any_dur;
        REQUIRE_EQ(kint64max,
                   turbo::safe_int_mod(-sec_max, -turbo::Duration::milliseconds(1), &rem));
        REQUIRE_EQ(-sec_max + turbo::Duration::milliseconds(kint64max), rem);

        rem = any_dur;
        REQUIRE_EQ(kint64min,
                   turbo::safe_int_mod(-sec_max, turbo::Duration::milliseconds(1), &rem));
        REQUIRE_EQ(-sec_max - turbo::Duration::milliseconds(kint64min), rem);

        rem = any_dur;
        REQUIRE_EQ(kint64min,
                   turbo::safe_int_mod(sec_max, -turbo::Duration::milliseconds(1), &rem));
        REQUIRE_EQ(sec_max + turbo::Duration::milliseconds(kint64min), rem);

        //
        // operator/(Duration, Duration) is a wrapper for safe_int_mod().
        //

        // IEEE 754 says inf / inf should be nan, but int64_t doesn't have
        // nan so we'll return kint64max/kint64min instead.
        REQUIRE(std::isnan(dbl_inf / dbl_inf));
        REQUIRE_EQ(kint64max, inf / inf);
        REQUIRE_EQ(kint64max, -inf / -inf);
        REQUIRE_EQ(kint64min, -inf / inf);
        REQUIRE_EQ(kint64min, inf / -inf);

        REQUIRE(std::isinf(dbl_inf / 2.0));
        REQUIRE_EQ(kint64max, inf / any_dur);
        REQUIRE_EQ(kint64max, -inf / -any_dur);
        REQUIRE_EQ(kint64min, -inf / any_dur);
        REQUIRE_EQ(kint64min, inf / -any_dur);

        REQUIRE_EQ(0.0, 2.0 / dbl_inf);
        REQUIRE_EQ(0, any_dur / inf);
        REQUIRE_EQ(0, any_dur / -inf);
        REQUIRE_EQ(0, -any_dur / inf);
        REQUIRE_EQ(0, -any_dur / -inf);
        REQUIRE_EQ(0, turbo::Duration::zero() / inf);

        // Division of Duration by a Duration overflow/underflow
        REQUIRE_EQ(kint64max, sec_max / turbo::Duration::milliseconds(1));
        REQUIRE_EQ(kint64max, -sec_max / -turbo::Duration::milliseconds(1));
        REQUIRE_EQ(kint64min, -sec_max / turbo::Duration::milliseconds(1));
        REQUIRE_EQ(kint64min, sec_max / -turbo::Duration::milliseconds(1));
    }

    TEST_CASE("Duration, InfinityFDiv") {
        const turbo::Duration any_dur = turbo::Duration::seconds(1);
        const turbo::Duration inf = turbo::Duration::infinite();
        const double dbl_inf = std::numeric_limits<double>::infinity();

        REQUIRE_EQ(dbl_inf, inf.safe_float_mod(inf));
        REQUIRE_EQ(dbl_inf, (-inf).safe_float_mod( -inf));
        REQUIRE_EQ(dbl_inf, (inf).safe_float_mod(any_dur));
        REQUIRE_EQ(0.0, (any_dur).safe_float_mod(inf));
        REQUIRE_EQ(dbl_inf, (-inf).safe_float_mod(-any_dur));
        REQUIRE_EQ(0.0, (-any_dur).safe_float_mod(-inf));

        REQUIRE_EQ(-dbl_inf, (-inf).safe_float_mod(inf));
        REQUIRE_EQ(-dbl_inf, (inf).safe_float_mod(-inf));
        REQUIRE_EQ(-dbl_inf, (-inf).safe_float_mod(any_dur));
        REQUIRE_EQ(0.0, (-any_dur).safe_float_mod(inf));
        REQUIRE_EQ(-dbl_inf, (inf).safe_float_mod(-any_dur));
        REQUIRE_EQ(0.0, (any_dur).safe_float_mod(-inf));
    }

    TEST_CASE("Duration, DivisionByZero") {
        const turbo::Duration zero = turbo::Duration::zero();
        const turbo::Duration inf = turbo::Duration::infinite();
        const turbo::Duration any_dur = turbo::Duration::seconds(1);
        const double dbl_inf = std::numeric_limits<double>::infinity();
        const double dbl_denorm = std::numeric_limits<double>::denorm_min();

        // Operator/(Duration, double)
        REQUIRE_EQ(inf, zero / 0.0);
        REQUIRE_EQ(-inf, zero / -0.0);
        REQUIRE_EQ(inf, any_dur / 0.0);
        REQUIRE_EQ(-inf, any_dur / -0.0);
        REQUIRE_EQ(-inf, -any_dur / 0.0);
        REQUIRE_EQ(inf, -any_dur / -0.0);

        // Tests dividing by a number very close to, but not quite zero.
        REQUIRE_EQ(zero, zero / dbl_denorm);
        REQUIRE_EQ(zero, zero / -dbl_denorm);
        REQUIRE_EQ(inf, any_dur / dbl_denorm);
        REQUIRE_EQ(-inf, any_dur / -dbl_denorm);
        REQUIRE_EQ(-inf, -any_dur / dbl_denorm);
        REQUIRE_EQ(inf, -any_dur / -dbl_denorm);

        // IDiv
        turbo::Duration rem = zero;
        REQUIRE_EQ(kint64max, turbo::safe_int_mod(zero, zero, &rem));
        REQUIRE_EQ(inf, rem);

        rem = zero;
        REQUIRE_EQ(kint64max, turbo::safe_int_mod(any_dur, zero, &rem));
        REQUIRE_EQ(inf, rem);

        rem = zero;
        REQUIRE_EQ(kint64min, turbo::safe_int_mod(-any_dur, zero, &rem));
        REQUIRE_EQ(-inf, rem);

        // Operator/(Duration, Duration)
        REQUIRE_EQ(kint64max, zero / zero);
        REQUIRE_EQ(kint64max, any_dur / zero);
        REQUIRE_EQ(kint64min, -any_dur / zero);

        // FDiv
        REQUIRE_EQ(dbl_inf, (zero).safe_float_mod( zero));
        REQUIRE_EQ(dbl_inf, (any_dur).safe_float_mod(zero));
        REQUIRE_EQ(-dbl_inf, (-any_dur).safe_float_mod( zero));
    }
    bool is_infinit(const turbo::Duration& d) {
        return d == turbo::Duration::infinite() || d == -turbo::Duration::infinite();
    }
    TEST_CASE("Duration, NaN") {
        // Note that IEEE 754 does not define the behavior of a nan's sign when it is
        // copied, so the code below allows for either + or - infinite_duration.
#define TEST_NAN_HANDLING(NAME, NAN)           \
  do {                                         \
    auto x = NAME(NAN);                        \
    REQUIRE(is_infinit(x));        \
    auto y = NAME(42);                         \
    y *= NAN;                                  \
    REQUIRE(is_infinit(y));        \
    auto z = NAME(42);                         \
    z /= NAN;                                  \
    REQUIRE(is_infinit(z));        \
  } while (0)

        const double nan = std::numeric_limits<double>::quiet_NaN();
        TEST_NAN_HANDLING(turbo::Duration::nanoseconds, nan);
        TEST_NAN_HANDLING(turbo::Duration::microseconds, nan);
        TEST_NAN_HANDLING(turbo::Duration::milliseconds, nan);
        TEST_NAN_HANDLING(turbo::Duration::seconds, nan);
        TEST_NAN_HANDLING(turbo::Duration::minutes, nan);
        TEST_NAN_HANDLING(turbo::Duration::hours, nan);

        TEST_NAN_HANDLING(turbo::Duration::nanoseconds, -nan);
        TEST_NAN_HANDLING(turbo::Duration::microseconds, -nan);
        TEST_NAN_HANDLING(turbo::Duration::milliseconds, -nan);
        TEST_NAN_HANDLING(turbo::Duration::seconds, -nan);
        TEST_NAN_HANDLING(turbo::Duration::minutes, -nan);
        TEST_NAN_HANDLING(turbo::Duration::hours, -nan);

#undef TEST_NAN_HANDLING
    }

    TEST_CASE("Duration, Range") {
        const turbo::Duration range = ApproxYears(100 * 1e9);
        const turbo::Duration range_future = range;
        const turbo::Duration range_past = -range;

        REQUIRE_LT(range_future, turbo::Duration::infinite());
        REQUIRE_GT(range_past, -turbo::Duration::infinite());

        const turbo::Duration full_range = range_future - range_past;
        REQUIRE_GT(full_range, turbo::Duration::zero());
        REQUIRE_LT(full_range, turbo::Duration::infinite());

        const turbo::Duration neg_full_range = range_past - range_future;
        REQUIRE_LT(neg_full_range, turbo::Duration::zero());
        REQUIRE_GT(neg_full_range, -turbo::Duration::infinite());

        REQUIRE_LT(neg_full_range, full_range);
        REQUIRE_EQ(neg_full_range, -full_range);
    }

    TEST_CASE("Duration, RelationalOperators") {
#define TEST_REL_OPS(UNIT)               \
  static_assert(UNIT(2) == UNIT(2), ""); \
  static_assert(UNIT(1) != UNIT(2), ""); \
  static_assert(UNIT(1) < UNIT(2), "");  \
  static_assert(UNIT(3) > UNIT(2), "");  \
  static_assert(UNIT(1) <= UNIT(2), ""); \
  static_assert(UNIT(2) <= UNIT(2), ""); \
  static_assert(UNIT(3) >= UNIT(2), ""); \
  static_assert(UNIT(2) >= UNIT(2), "");

        TEST_REL_OPS(turbo::Duration::nanoseconds);
        TEST_REL_OPS(turbo::Duration::microseconds);
        TEST_REL_OPS(turbo::Duration::milliseconds);
        TEST_REL_OPS(turbo::Duration::seconds);
        TEST_REL_OPS(turbo::Duration::minutes);
        TEST_REL_OPS(turbo::Duration::hours);

#undef TEST_REL_OPS
    }

    TEST_CASE("Duration, Addition") {
#define TEST_ADD_OPS(UNIT)                  \
  do {                                      \
    REQUIRE_EQ(UNIT(2), UNIT(1) + UNIT(1));  \
    REQUIRE_EQ(UNIT(1), UNIT(2) - UNIT(1));  \
    REQUIRE_EQ(UNIT(0), UNIT(2) - UNIT(2));  \
    REQUIRE_EQ(UNIT(-1), UNIT(1) - UNIT(2)); \
    REQUIRE_EQ(UNIT(-2), UNIT(0) - UNIT(2)); \
    REQUIRE_EQ(UNIT(-2), UNIT(1) - UNIT(3)); \
    turbo::Duration a = UNIT(1);             \
    a += UNIT(1);                           \
    REQUIRE_EQ(UNIT(2), a);                  \
    a -= UNIT(1);                           \
    REQUIRE_EQ(UNIT(1), a);                  \
  } while (0)

        TEST_ADD_OPS(turbo::Duration::nanoseconds);
        TEST_ADD_OPS(turbo::Duration::microseconds);
        TEST_ADD_OPS(turbo::Duration::milliseconds);
        TEST_ADD_OPS(turbo::Duration::seconds);
        TEST_ADD_OPS(turbo::Duration::minutes);
        TEST_ADD_OPS(turbo::Duration::hours);

#undef TEST_ADD_OPS

        REQUIRE_EQ(turbo::Duration::seconds(2), turbo::Duration::seconds(3) - 2 * turbo::Duration::milliseconds(500));
        REQUIRE_EQ(turbo::Duration::seconds(2) + turbo::Duration::milliseconds(500),
                   turbo::Duration::seconds(3) - turbo::Duration::milliseconds(500));

        REQUIRE_EQ(turbo::Duration::seconds(1) + turbo::Duration::milliseconds(998),
                   turbo::Duration::milliseconds(999) + turbo::Duration::milliseconds(999));

        REQUIRE_EQ(turbo::Duration::milliseconds(-1),
                   turbo::Duration::milliseconds(998) - turbo::Duration::milliseconds(999));

        // Tests fractions of a nanoseconds. These are implementation details only.
        REQUIRE_GT(turbo::Duration::nanoseconds(1), turbo::Duration::nanoseconds(1) / 2);
        REQUIRE_EQ(turbo::Duration::nanoseconds(1),
                   turbo::Duration::nanoseconds(1) / 2 + turbo::Duration::nanoseconds(1) / 2);
        REQUIRE_GT(turbo::Duration::nanoseconds(1) / 4, turbo::Duration::nanoseconds(0));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1) / 8, turbo::Duration::nanoseconds(0));

        // Tests subtraction that will cause wrap around of the rep_lo_ bits.
        turbo::Duration d_7_5 = turbo::Duration::seconds(7) + turbo::Duration::milliseconds(500);
        turbo::Duration d_3_7 = turbo::Duration::seconds(3) + turbo::Duration::milliseconds(700);
        turbo::Duration ans_3_8 = turbo::Duration::seconds(3) + turbo::Duration::milliseconds(800);
        REQUIRE_EQ(ans_3_8, d_7_5 - d_3_7);

        // Subtracting min_duration
        turbo::Duration min_dur = turbo::Duration::seconds(kint64min);
        REQUIRE_EQ(turbo::Duration::seconds(0), min_dur - min_dur);
        REQUIRE_EQ(turbo::Duration::seconds(kint64max), turbo::Duration::seconds(-1) - min_dur);
    }

    TEST_CASE("Duration, Negation") {
        // By storing negations of various values in constexpr variables we
        // verify that the initializers are constant expressions.
        constexpr turbo::Duration negated_zero_duration = -turbo::Duration::zero();
        REQUIRE_EQ(negated_zero_duration, turbo::Duration::zero());

        constexpr turbo::Duration negated_infinite_duration =
                -turbo::Duration::infinite();
        REQUIRE_NE(negated_infinite_duration, turbo::Duration::infinite());
        REQUIRE_EQ(-negated_infinite_duration, turbo::Duration::infinite());

        // The public APIs to check if a duration is infinite depend on using
        // -Duration::infinite(), but we're trying to test operator- here, so we
        // need to use the lower-level internal query is_infinite.
        REQUIRE(negated_infinite_duration.is_infinite());

        // The largest Duration is kint64max seconds and kTicksPerSecond - 1 ticks.
        // Using the turbo::time_internal::MakeDuration API is the cleanest way to
        // construct that Duration.
        constexpr turbo::Duration max_duration = turbo::time_internal::MakeDuration(
                kint64max, turbo::time_internal::kTicksPerSecond - 1);
        constexpr turbo::Duration negated_max_duration = -max_duration;
        // The largest negatable value is one tick above the minimum representable;
        // it's the negation of max_duration.
        constexpr turbo::Duration nearly_min_duration =
                turbo::time_internal::MakeDuration(kint64min, int64_t{1});
        constexpr turbo::Duration negated_nearly_min_duration = -nearly_min_duration;

        REQUIRE_EQ(negated_max_duration, nearly_min_duration);
        REQUIRE_EQ(negated_nearly_min_duration, max_duration);
        REQUIRE_EQ(-(-max_duration), max_duration);

        constexpr turbo::Duration min_duration =
                turbo::time_internal::MakeDuration(kint64min);
        constexpr turbo::Duration negated_min_duration = -min_duration;
        REQUIRE_EQ(negated_min_duration, turbo::Duration::infinite());
    }

    TEST_CASE("Duration, AbsoluteValue") {

        REQUIRE_EQ(turbo::Duration::zero(), turbo::Duration::zero().abs());
        REQUIRE_EQ(turbo::Duration::seconds(1), turbo::Duration::seconds(1).abs());
        REQUIRE_EQ(turbo::Duration::seconds(1), (turbo::Duration::seconds(-1)).abs());

        REQUIRE_EQ(turbo::Duration::infinite(), turbo::Duration::infinite().abs());
        REQUIRE_EQ(turbo::Duration::infinite(), (-turbo::Duration::infinite()).abs());

        turbo::Duration max_dur =
                turbo::Duration::seconds(kint64max) + (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1) / 4);
        REQUIRE_EQ(max_dur, max_dur.abs());

        turbo::Duration min_dur = turbo::Duration::seconds(kint64min);
        REQUIRE_EQ(turbo::Duration::infinite(), min_dur.abs());
        REQUIRE_EQ(max_dur, (min_dur + turbo::Duration::nanoseconds(1) / 4).abs());
    }

    TEST_CASE("Duration, Multiplication") {

#define TEST_MUL_OPS(UNIT)                                    \
  do {                                                        \
    REQUIRE_EQ(UNIT(5), UNIT(2) * 2.5);                        \
    REQUIRE_EQ(UNIT(2), UNIT(5) / 2.5);                        \
    REQUIRE_EQ(UNIT(-5), UNIT(-2) * 2.5);                      \
    REQUIRE_EQ(UNIT(-5), -UNIT(2) * 2.5);                      \
    REQUIRE_EQ(UNIT(-5), UNIT(2) * -2.5);                      \
    REQUIRE_EQ(UNIT(-2), UNIT(-5) / 2.5);                      \
    REQUIRE_EQ(UNIT(-2), -UNIT(5) / 2.5);                      \
    REQUIRE_EQ(UNIT(-2), UNIT(5) / -2.5);                      \
    REQUIRE_EQ(UNIT(2), UNIT(11) % UNIT(3));                   \
    turbo::Duration a = UNIT(2);                               \
    a *= 2.5;                                                 \
    REQUIRE_EQ(UNIT(5), a);                                    \
    a /= 2.5;                                                 \
    REQUIRE_EQ(UNIT(2), a);                                    \
    a %= UNIT(1);                                             \
    REQUIRE_EQ(UNIT(0), a);                                    \
    turbo::Duration big = UNIT(1000000000);                    \
    big *= 3;                                                 \
    big /= 3;                                                 \
    REQUIRE_EQ(UNIT(1000000000), big);                         \
    REQUIRE_EQ(-UNIT(2), -UNIT(2));                            \
    REQUIRE_EQ(-UNIT(2), UNIT(2) * -1);                        \
    REQUIRE_EQ(-UNIT(2), -1 * UNIT(2));                        \
    REQUIRE_EQ(-UNIT(-2), UNIT(2));                            \
    REQUIRE_EQ(2, UNIT(2) / UNIT(1));                          \
    turbo::Duration rem;                                       \
    REQUIRE_EQ(2, turbo::safe_int_mod(UNIT(2), UNIT(1), &rem)); \
    REQUIRE_EQ(2.0, UNIT(2).safe_float_mod(UNIT(1)));     \
  } while (0)

        TEST_MUL_OPS(turbo::Duration::nanoseconds);
        TEST_MUL_OPS(turbo::Duration::microseconds);
        TEST_MUL_OPS(turbo::Duration::milliseconds);
        TEST_MUL_OPS(turbo::Duration::seconds);
        TEST_MUL_OPS(turbo::Duration::minutes);
        TEST_MUL_OPS(turbo::Duration::hours);

#undef TEST_MUL_OPS

        // Ensures that multiplication and division by 1 with a maxed-out durations
        // doesn't lose precision.
        turbo::Duration max_dur =
                turbo::Duration::seconds(kint64max) + (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1) / 4);
        turbo::Duration min_dur = turbo::Duration::seconds(kint64min);
        REQUIRE_EQ(max_dur, max_dur * 1);
        REQUIRE_EQ(max_dur, max_dur / 1);
        REQUIRE_EQ(min_dur, min_dur * 1);
        REQUIRE_EQ(min_dur, min_dur / 1);

        // Tests division on a Duration with a large number of significant digits.
        // Tests when the digits span hi and lo as well as only in hi.
        turbo::Duration sigfigs = turbo::Duration::seconds(2000000000) + turbo::Duration::nanoseconds(3);
        REQUIRE_EQ(turbo::Duration::seconds(666666666) + turbo::Duration::nanoseconds(666666667) +
                   turbo::Duration::nanoseconds(1) / 2,
                   sigfigs / 3);
        sigfigs = turbo::Duration::seconds(int64_t{7000000000});
        REQUIRE_EQ(turbo::Duration::seconds(2333333333) + turbo::Duration::nanoseconds(333333333) +
                   turbo::Duration::nanoseconds(1) / 4,
                   sigfigs / 3);

        REQUIRE_EQ(turbo::Duration::seconds(7) + turbo::Duration::milliseconds(500), turbo::Duration::seconds(3) * 2.5);
        REQUIRE_EQ(turbo::Duration::seconds(8) * -1 + turbo::Duration::milliseconds(300),
                   (turbo::Duration::seconds(2) + turbo::Duration::milliseconds(200)) * -3.5);
        REQUIRE_EQ(-turbo::Duration::seconds(8) + turbo::Duration::milliseconds(300),
                   (turbo::Duration::seconds(2) + turbo::Duration::milliseconds(200)) * -3.5);
        REQUIRE_EQ(turbo::Duration::seconds(1) + turbo::Duration::milliseconds(875),
                   (turbo::Duration::seconds(7) + turbo::Duration::milliseconds(500)) / 4);
        REQUIRE_EQ(turbo::Duration::seconds(30),
                   (turbo::Duration::seconds(7) + turbo::Duration::milliseconds(500)) / 0.25);
        REQUIRE_EQ(turbo::Duration::seconds(3),
                   (turbo::Duration::seconds(7) + turbo::Duration::milliseconds(500)) / 2.5);

        // Tests division remainder.
        REQUIRE_EQ(turbo::Duration::nanoseconds(0), turbo::Duration::nanoseconds(7) % turbo::Duration::nanoseconds(1));
        REQUIRE_EQ(turbo::Duration::nanoseconds(0), turbo::Duration::nanoseconds(0) % turbo::Duration::nanoseconds(10));
        REQUIRE_EQ(turbo::Duration::nanoseconds(2), turbo::Duration::nanoseconds(7) % turbo::Duration::nanoseconds(5));
        REQUIRE_EQ(turbo::Duration::nanoseconds(2), turbo::Duration::nanoseconds(2) % turbo::Duration::nanoseconds(5));

        REQUIRE_EQ(turbo::Duration::nanoseconds(1), turbo::Duration::nanoseconds(10) % turbo::Duration::nanoseconds(3));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1),
                   turbo::Duration::nanoseconds(10) % turbo::Duration::nanoseconds(-3));
        REQUIRE_EQ(turbo::Duration::nanoseconds(-1),
                   turbo::Duration::nanoseconds(-10) % turbo::Duration::nanoseconds(3));
        REQUIRE_EQ(turbo::Duration::nanoseconds(-1),
                   turbo::Duration::nanoseconds(-10) % turbo::Duration::nanoseconds(-3));

        REQUIRE_EQ(turbo::Duration::milliseconds(100),
                   turbo::Duration::seconds(1) % turbo::Duration::milliseconds(300));
        REQUIRE_EQ(
                turbo::Duration::milliseconds(300),
                (turbo::Duration::seconds(3) + turbo::Duration::milliseconds(800)) % turbo::Duration::milliseconds(500));

        REQUIRE_EQ(turbo::Duration::nanoseconds(1), turbo::Duration::nanoseconds(1) % turbo::Duration::seconds(1));
        REQUIRE_EQ(turbo::Duration::nanoseconds(-1), turbo::Duration::nanoseconds(-1) % turbo::Duration::seconds(1));
        REQUIRE_EQ(0, turbo::Duration::nanoseconds(-1) / turbo::Duration::seconds(1));  // Actual -1e-9

        // Tests identity a = (a/b)*b + a%b
#define TEST_MOD_IDENTITY(a, b) \
  REQUIRE_EQ((a), ((a) / (b))*(b) + ((a)%(b)))

        TEST_MOD_IDENTITY(turbo::Duration::seconds(0), turbo::Duration::seconds(2));
        TEST_MOD_IDENTITY(turbo::Duration::seconds(1), turbo::Duration::seconds(1));
        TEST_MOD_IDENTITY(turbo::Duration::seconds(1), turbo::Duration::seconds(2));
        TEST_MOD_IDENTITY(turbo::Duration::seconds(2), turbo::Duration::seconds(1));

        TEST_MOD_IDENTITY(turbo::Duration::seconds(-2), turbo::Duration::seconds(1));
        TEST_MOD_IDENTITY(turbo::Duration::seconds(2), turbo::Duration::seconds(-1));
        TEST_MOD_IDENTITY(turbo::Duration::seconds(-2), turbo::Duration::seconds(-1));

        TEST_MOD_IDENTITY(turbo::Duration::nanoseconds(0), turbo::Duration::nanoseconds(2));
        TEST_MOD_IDENTITY(turbo::Duration::nanoseconds(1), turbo::Duration::nanoseconds(1));
        TEST_MOD_IDENTITY(turbo::Duration::nanoseconds(1), turbo::Duration::nanoseconds(2));
        TEST_MOD_IDENTITY(turbo::Duration::nanoseconds(2), turbo::Duration::nanoseconds(1));

        TEST_MOD_IDENTITY(turbo::Duration::nanoseconds(-2), turbo::Duration::nanoseconds(1));
        TEST_MOD_IDENTITY(turbo::Duration::nanoseconds(2), turbo::Duration::nanoseconds(-1));
        TEST_MOD_IDENTITY(turbo::Duration::nanoseconds(-2), turbo::Duration::nanoseconds(-1));

        // Mixed seconds + subseconds
        turbo::Duration mixed_a = turbo::Duration::seconds(1) + turbo::Duration::nanoseconds(2);
        turbo::Duration mixed_b = turbo::Duration::seconds(1) + turbo::Duration::nanoseconds(3);

        TEST_MOD_IDENTITY(turbo::Duration::seconds(0), mixed_a);
        TEST_MOD_IDENTITY(mixed_a, mixed_a);
        TEST_MOD_IDENTITY(mixed_a, mixed_b);
        TEST_MOD_IDENTITY(mixed_b, mixed_a);

        TEST_MOD_IDENTITY(-mixed_a, mixed_b);
        TEST_MOD_IDENTITY(mixed_a, -mixed_b);
        TEST_MOD_IDENTITY(-mixed_a, -mixed_b);

#undef TEST_MOD_IDENTITY
    }

    TEST_CASE("Duration, Truncation") {
        const turbo::Duration d = turbo::Duration::nanoseconds(1234567890);
        const turbo::Duration inf = turbo::Duration::infinite();
        for (int unit_sign: {1, -1}) {  // sign shouldn't matter
            REQUIRE_EQ(turbo::Duration::nanoseconds(1234567890),
                       d.trunc( unit_sign * turbo::Duration::nanoseconds(1)));
            REQUIRE_EQ(turbo::Duration::microseconds(1234567),
                       d.trunc( unit_sign * turbo::Duration::microseconds(1)));
            REQUIRE_EQ(turbo::Duration::milliseconds(1234),
                       d.trunc( unit_sign * turbo::Duration::milliseconds(1)));
            REQUIRE_EQ(turbo::Duration::seconds(1), d.trunc(unit_sign * turbo::Duration::seconds(1)));
            REQUIRE_EQ(inf, inf.trunc(unit_sign * turbo::Duration::seconds(1)));

            REQUIRE_EQ(turbo::Duration::nanoseconds(-1234567890),
                       (-d).trunc( unit_sign * turbo::Duration::nanoseconds(1)));
            REQUIRE_EQ(turbo::Duration::microseconds(-1234567),
                       (-d).trunc( unit_sign * turbo::Duration::microseconds(1)));
            REQUIRE_EQ(turbo::Duration::milliseconds(-1234),
                       (-d).trunc( unit_sign * turbo::Duration::milliseconds(1)));
            REQUIRE_EQ(turbo::Duration::seconds(-1), (-d).trunc( unit_sign * turbo::Duration::seconds(1)));
            REQUIRE_EQ(-inf, (-inf).trunc( unit_sign * turbo::Duration::seconds(1)));
        }
    }

    TEST_CASE("Duration, Fraction") {
        const turbo::Duration d = turbo::Duration::nanoseconds(1234567890);
        const turbo::Duration inf = turbo::Duration::infinite();

            REQUIRE_EQ(turbo::Duration::nanoseconds(0),
                       d.fraction(turbo::Duration::nanoseconds(1)));
            REQUIRE_EQ(turbo::Duration::nanoseconds(890),
                       d.fraction(turbo::Duration::microseconds(1)));
            REQUIRE_EQ(turbo::Duration::nanoseconds(567890),
                       d.fraction(turbo::Duration::milliseconds(1)));
            REQUIRE_EQ(turbo::Duration::nanoseconds(234567890), d.fraction(turbo::Duration::seconds(1)));
    }

    TEST_CASE("Duration, Flooring") {
        const turbo::Duration d = turbo::Duration::nanoseconds(1234567890);
        const turbo::Duration inf = turbo::Duration::infinite();
        for (int unit_sign: {1, -1}) {  // sign shouldn't matter
            REQUIRE_EQ(turbo::Duration::nanoseconds(1234567890),
                       d.floor(unit_sign * turbo::Duration::nanoseconds(1)));
            REQUIRE_EQ(turbo::Duration::microseconds(1234567),
                       d.floor(unit_sign * turbo::Duration::microseconds(1)));
            REQUIRE_EQ(turbo::Duration::milliseconds(1234),
                       d.floor(unit_sign * turbo::Duration::milliseconds(1)));
            REQUIRE_EQ(turbo::Duration::seconds(1), d.floor( unit_sign * turbo::Duration::seconds(1)));
            REQUIRE_EQ(inf, inf.floor(unit_sign * turbo::Duration::seconds(1)));

            REQUIRE_EQ(turbo::Duration::nanoseconds(-1234567890),
                       (-d).floor( unit_sign * turbo::Duration::nanoseconds(1)));
            REQUIRE_EQ(turbo::Duration::microseconds(-1234568),
                       (-d).floor(unit_sign * turbo::Duration::microseconds(1)));
            REQUIRE_EQ(turbo::Duration::milliseconds(-1235),
                       (-d).floor( unit_sign * turbo::Duration::milliseconds(1)));
            REQUIRE_EQ(turbo::Duration::seconds(-2), (-d).floor( unit_sign * turbo::Duration::seconds(1)));
            REQUIRE_EQ(-inf, (-inf).floor( unit_sign * turbo::Duration::seconds(1)));
        }
    }

    TEST_CASE("Duration, Ceiling") {
        const turbo::Duration d = turbo::Duration::nanoseconds(1234567890);
        const turbo::Duration inf = turbo::Duration::infinite();
        for (int unit_sign: {1, -1}) {  // // sign shouldn't matter
            REQUIRE_EQ(turbo::Duration::nanoseconds(1234567890),
                       d.ceil(unit_sign * turbo::Duration::nanoseconds(1)));
            REQUIRE_EQ(turbo::Duration::microseconds(1234568),
                       d.ceil( unit_sign * turbo::Duration::microseconds(1)));
            REQUIRE_EQ(turbo::Duration::milliseconds(1235),
                       d.ceil( unit_sign * turbo::Duration::milliseconds(1)));
            REQUIRE_EQ(turbo::Duration::seconds(2), d.ceil( unit_sign * turbo::Duration::seconds(1)));
            REQUIRE_EQ(inf, inf.ceil( unit_sign * turbo::Duration::seconds(1)));

            REQUIRE_EQ(turbo::Duration::nanoseconds(-1234567890),
                       (-d).ceil( unit_sign * turbo::Duration::nanoseconds(1)));
            REQUIRE_EQ(turbo::Duration::microseconds(-1234567),
                       (-d).ceil( unit_sign * turbo::Duration::microseconds(1)));
            REQUIRE_EQ(turbo::Duration::milliseconds(-1234),
                       (-d).ceil( unit_sign * turbo::Duration::milliseconds(1)));
            REQUIRE_EQ(turbo::Duration::seconds(-1), (-d).ceil( unit_sign * turbo::Duration::seconds(1)));
            REQUIRE_EQ(-inf, (-inf).ceil( unit_sign * turbo::Duration::seconds(1)));
        }
    }

    TEST_CASE("Duration, RoundTripUnits") {
        const int kRange = 100000;

#define ROUND_TRIP_UNIT(U, LOW, HIGH)          \
  do {                                         \
    for (int64_t i = LOW; i < HIGH; ++i) {     \
      turbo::Duration d = turbo::Duration::U(i);           \
      if (d == turbo::Duration::infinite())       \
        REQUIRE_EQ(kint64max, d / turbo::Duration::U(1));  \
      else if (d == -turbo::Duration::infinite()) \
        REQUIRE_EQ(kint64min, d / turbo::Duration::U(1));  \
      else                                     \
        REQUIRE_EQ(i, turbo::Duration::U(i) / turbo::Duration::U(1)); \
    }                                          \
  } while (0)

        ROUND_TRIP_UNIT(nanoseconds, kint64min, kint64min + kRange);
        ROUND_TRIP_UNIT(nanoseconds, -kRange, kRange);
        ROUND_TRIP_UNIT(nanoseconds, kint64max - kRange, kint64max);

        ROUND_TRIP_UNIT(microseconds, kint64min, kint64min + kRange);
        ROUND_TRIP_UNIT(microseconds, -kRange, kRange);
        ROUND_TRIP_UNIT(microseconds, kint64max - kRange, kint64max);

        ROUND_TRIP_UNIT(milliseconds, kint64min, kint64min + kRange);
        ROUND_TRIP_UNIT(milliseconds, -kRange, kRange);
        ROUND_TRIP_UNIT(milliseconds, kint64max - kRange, kint64max);

        ROUND_TRIP_UNIT(seconds, kint64min, kint64min + kRange);
        ROUND_TRIP_UNIT(seconds, -kRange, kRange);
        ROUND_TRIP_UNIT(seconds, kint64max - kRange, kint64max);

        ROUND_TRIP_UNIT(minutes, kint64min / 60, kint64min / 60 + kRange);
        ROUND_TRIP_UNIT(minutes, -kRange, kRange);
        ROUND_TRIP_UNIT(minutes, kint64max / 60 - kRange, kint64max / 60);

        ROUND_TRIP_UNIT(hours, kint64min / 3600, kint64min / 3600 + kRange);
        ROUND_TRIP_UNIT(hours, -kRange, kRange);
        ROUND_TRIP_UNIT(hours, kint64max / 3600 - kRange, kint64max / 3600);

#undef ROUND_TRIP_UNIT
    }

    TEST_CASE("Duration, TruncConversions") {

        const struct {
            timespec ts;
            turbo::Duration d;
        } from_ts[] = {
                {{1,  1},         turbo::Duration::seconds(1) + turbo::Duration::nanoseconds(1)},
                {{1,  0},         turbo::Duration::seconds(1) + turbo::Duration::nanoseconds(0)},
                {{0,  0},         turbo::Duration::seconds(0) + turbo::Duration::nanoseconds(0)},
                {{0,  -1},        turbo::Duration::seconds(0) - turbo::Duration::nanoseconds(1)},
                {{-1, 999999999}, turbo::Duration::seconds(0) - turbo::Duration::nanoseconds(1)},
                {{-1, 1},         turbo::Duration::seconds(-1) + turbo::Duration::nanoseconds(1)},
                {{-1, 0},         turbo::Duration::seconds(-1) + turbo::Duration::nanoseconds(0)},
                {{-1, -1},        turbo::Duration::seconds(-1) - turbo::Duration::nanoseconds(1)},
                {{-2, 999999999}, turbo::Duration::seconds(-1) - turbo::Duration::nanoseconds(1)},
        };
        for (const auto &test: from_ts) {
            REQUIRE_EQ(test.d, turbo::Duration::from_timespec(test.ts));
        }

        const struct {
            timeval tv;
            turbo::Duration d;
        } from_tv[] = {
                {{1,  1},      turbo::Duration::seconds(1) + turbo::Duration::microseconds(1)},
                {{1,  0},      turbo::Duration::seconds(1) + turbo::Duration::microseconds(0)},
                {{0,  0},      turbo::Duration::seconds(0) + turbo::Duration::microseconds(0)},
                {{0,  -1},     turbo::Duration::seconds(0) - turbo::Duration::microseconds(1)},
                {{-1, 999999}, turbo::Duration::seconds(0) - turbo::Duration::microseconds(1)},
                {{-1, 1},      turbo::Duration::seconds(-1) + turbo::Duration::microseconds(1)},
                {{-1, 0},      turbo::Duration::seconds(-1) + turbo::Duration::microseconds(0)},
                {{-1, -1},     turbo::Duration::seconds(-1) - turbo::Duration::microseconds(1)},
                {{-2, 999999}, turbo::Duration::seconds(-1) - turbo::Duration::microseconds(1)},
        };
        for (const auto &test: from_tv) {
            REQUIRE_EQ(test.d, turbo::Duration::from_timeval(test.tv));
        }
    }

    TEST_CASE("Duration, SmallConversions") {
        // Special tests for conversions of small durations.

        REQUIRE_EQ(turbo::Duration::zero(), turbo::Duration::seconds(0));
        // TODO(bww): Is the next one OK?
        REQUIRE_EQ(turbo::Duration::zero(), turbo::Duration::seconds(std::nextafter(0.125e-9, 0)));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1) / 4, turbo::Duration::seconds(0.125e-9));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1) / 4, turbo::Duration::seconds(0.250e-9));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1) / 2, turbo::Duration::seconds(0.375e-9));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1) / 2, turbo::Duration::seconds(0.500e-9));
        REQUIRE_EQ(turbo::Duration::nanoseconds(3) / 4, turbo::Duration::seconds(0.625e-9));
        REQUIRE_EQ(turbo::Duration::nanoseconds(3) / 4, turbo::Duration::seconds(0.750e-9));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1), turbo::Duration::seconds(0.875e-9));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1), turbo::Duration::seconds(1.000e-9));

        REQUIRE_EQ(turbo::Duration::zero(), turbo::Duration::seconds(std::nextafter(-0.125e-9, 0)));
        REQUIRE_EQ(-turbo::Duration::nanoseconds(1) / 4, turbo::Duration::seconds(-0.125e-9));
        REQUIRE_EQ(-turbo::Duration::nanoseconds(1) / 4, turbo::Duration::seconds(-0.250e-9));
        REQUIRE_EQ(-turbo::Duration::nanoseconds(1) / 2, turbo::Duration::seconds(-0.375e-9));
        REQUIRE_EQ(-turbo::Duration::nanoseconds(1) / 2, turbo::Duration::seconds(-0.500e-9));
        REQUIRE_EQ(-turbo::Duration::nanoseconds(3) / 4, turbo::Duration::seconds(-0.625e-9));
        REQUIRE_EQ(-turbo::Duration::nanoseconds(3) / 4, turbo::Duration::seconds(-0.750e-9));
        REQUIRE_EQ(-turbo::Duration::nanoseconds(1), turbo::Duration::seconds(-0.875e-9));
        REQUIRE_EQ(-turbo::Duration::nanoseconds(1), turbo::Duration::seconds(-1.000e-9));
    }

    void VerifyApproxSameAsMul(double time_as_seconds, int *const misses) {
        auto direct_seconds = turbo::Duration::seconds(time_as_seconds);
        auto mul_by_one_second = time_as_seconds * turbo::Duration::seconds(1);
        // These are expected to differ by up to one tick due to fused multiply/add
        // contraction.
        if ((direct_seconds - mul_by_one_second).abs() >
            turbo::time_internal::MakeDuration(0, 1u)) {
            if (*misses > 10) return;
            REQUIRE_LE(++(*misses), 10);
            REQUIRE_EQ(direct_seconds, mul_by_one_second);
        }
    }

    // For a variety of interesting durations, we find the exact point
    // where one double converts to that duration, and the very next double
    // converts to the next duration.  For both of those points, verify that
    // seconds(point) returns a duration near point * seconds(1.0). (They may
    // not be exactly equal due to fused multiply/add contraction.)
    TEST_CASE("Duration, ToDoubleSecondsCheckEdgeCases") {

        constexpr uint32_t kTicksPerSecond = turbo::time_internal::kTicksPerSecond;
        constexpr auto duration_tick = turbo::time_internal::MakeDuration(0, 1u);
        int misses = 0;
        for (int64_t seconds = 0; seconds < 99; ++seconds) {
            uint32_t tick_vals[] = {0, +999, +999999, +999999999, kTicksPerSecond - 1,
                                    0, 1000, 1000000, 1000000000, kTicksPerSecond,
                                    1, 1001, 1000001, 1000000001, kTicksPerSecond + 1,
                                    2, 1002, 1000002, 1000000002, kTicksPerSecond + 2,
                                    3, 1003, 1000003, 1000000003, kTicksPerSecond + 3,
                                    4, 1004, 1000004, 1000000004, kTicksPerSecond + 4,
                                    5, 6, 7, 8, 9};
            for (uint32_t ticks: tick_vals) {
                turbo::Duration s_plus_t = turbo::Duration::seconds(seconds) + ticks * duration_tick;
                for (turbo::Duration d: {s_plus_t, -s_plus_t}) {
                    turbo::Duration after_d = d + duration_tick;
                    REQUIRE_NE(d, after_d);
                    REQUIRE_EQ(after_d - d, duration_tick);

                    double low_edge = d.to_seconds<double>();
                    REQUIRE_EQ(d, turbo::Duration::seconds(low_edge));

                    double high_edge = after_d.to_seconds<double>();
                    REQUIRE_EQ(after_d, turbo::Duration::seconds(high_edge));

                    for (;;) {
                        double midpoint = low_edge + (high_edge - low_edge) / 2;
                        if (midpoint == low_edge || midpoint == high_edge) break;
                        turbo::Duration mid_duration = turbo::Duration::seconds(midpoint);
                        if (mid_duration == d) {
                            low_edge = midpoint;
                        } else {
                            REQUIRE_EQ(mid_duration, after_d);
                            high_edge = midpoint;
                        }
                    }
                    // Now low_edge is the highest double that converts to Duration d,
                    // and high_edge is the lowest double that converts to Duration after_d.
                    VerifyApproxSameAsMul(low_edge, &misses);
                    VerifyApproxSameAsMul(high_edge, &misses);
                }
            }
        }
    }

    TEST_CASE("Duration, ToDoubleSecondsCheckRandom") {
        std::random_device rd;
        std::seed_seq seed({rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()});
        std::mt19937_64 gen(seed);
        // We want doubles distributed from 1/8ns up to 2^63, where
        // as many values are tested from 1ns to 2ns as from 1sec to 2sec,
        // so even distribute along a log-scale of those values, and
        // exponentiate before using them.  (9.223377e+18 is just slightly
        // out of bounds for turbo::Duration.)
        std::uniform_real_distribution<double> uniform(std::log(0.125e-9),
                                                       std::log(9.223377e+18));
        int misses = 0;
        for (int i = 0; i < 1000000; ++i) {
            double d = std::exp(uniform(gen));
            VerifyApproxSameAsMul(d, &misses);
            VerifyApproxSameAsMul(-d, &misses);
        }
    }

    TEST_CASE("Duration, ConversionSaturation") {
        turbo::Duration d;

        const auto max_timeval_sec =
                std::numeric_limits<decltype(timeval::tv_sec)>::max();
        const auto min_timeval_sec =
                std::numeric_limits<decltype(timeval::tv_sec)>::min();
        timeval tv;
        tv.tv_sec = max_timeval_sec;
        tv.tv_usec = 999998;
        d = turbo::Duration::from_timeval(tv);
        tv = d.to_timeval();
        REQUIRE_EQ(max_timeval_sec, tv.tv_sec);
        REQUIRE_EQ(999998, tv.tv_usec);
        d += turbo::Duration::microseconds(1);
        tv = d.to_timeval();
        REQUIRE_EQ(max_timeval_sec, tv.tv_sec);
        REQUIRE_EQ(999999, tv.tv_usec);
        d += turbo::Duration::microseconds(1);  // no effect
        tv = d.to_timeval();
        REQUIRE_EQ(max_timeval_sec, tv.tv_sec);
        REQUIRE_EQ(999999, tv.tv_usec);

        tv.tv_sec = min_timeval_sec;
        tv.tv_usec = 1;
        d = turbo::Duration::from_timeval(tv);
        tv = d.to_timeval();
        REQUIRE_EQ(min_timeval_sec, tv.tv_sec);
        REQUIRE_EQ(1, tv.tv_usec);
        d -= turbo::Duration::microseconds(1);
        tv = d.to_timeval();
        REQUIRE_EQ(min_timeval_sec, tv.tv_sec);
        REQUIRE_EQ(0, tv.tv_usec);
        d -= turbo::Duration::microseconds(1);  // no effect
        tv = d.to_timeval();
        REQUIRE_EQ(min_timeval_sec, tv.tv_sec);
        REQUIRE_EQ(0, tv.tv_usec);

        const auto max_timespec_sec =
                std::numeric_limits<decltype(timespec::tv_sec)>::max();
        const auto min_timespec_sec =
                std::numeric_limits<decltype(timespec::tv_sec)>::min();
        timespec ts;
        ts.tv_sec = max_timespec_sec;
        ts.tv_nsec = 999999998;
        d = turbo::Duration::from_timespec(ts);
        ts = d.to_timespec();
        REQUIRE_EQ(max_timespec_sec, ts.tv_sec);
        REQUIRE_EQ(999999998, ts.tv_nsec);
        d += turbo::Duration::nanoseconds(1);
        ts = d.to_timespec();
        REQUIRE_EQ(max_timespec_sec, ts.tv_sec);
        REQUIRE_EQ(999999999, ts.tv_nsec);
        d += turbo::Duration::nanoseconds(1);  // no effect
        ts = d.to_timespec();
        REQUIRE_EQ(max_timespec_sec, ts.tv_sec);
        REQUIRE_EQ(999999999, ts.tv_nsec);

        ts.tv_sec = min_timespec_sec;
        ts.tv_nsec = 1;
        d = turbo::Duration::from_timespec(ts);
        ts = d.to_timespec();
        REQUIRE_EQ(min_timespec_sec, ts.tv_sec);
        REQUIRE_EQ(1, ts.tv_nsec);
        d -= turbo::Duration::nanoseconds(1);
        ts = d.to_timespec();
        REQUIRE_EQ(min_timespec_sec, ts.tv_sec);
        REQUIRE_EQ(0, ts.tv_nsec);
        d -= turbo::Duration::nanoseconds(1);  // no effect
        ts = d.to_timespec();
        REQUIRE_EQ(min_timespec_sec, ts.tv_sec);
        REQUIRE_EQ(0, ts.tv_nsec);
    }

    TEST_CASE("Duration, format_duration") {
        // Example from Go's docs.
        REQUIRE_EQ("72h3m0.5s",
                   (turbo::Duration::hours(72) + turbo::Duration::minutes(3) +
                                          turbo::Duration::milliseconds(500)).to_string());
        // Go's largest time: 2540400h10m10.000000000s
        REQUIRE_EQ("2540400h10m10s",
                   (turbo::Duration::hours(2540400) + turbo::Duration::minutes(10) +
                                          turbo::Duration::seconds(10)).to_string());

        REQUIRE_EQ("0", (turbo::Duration::zero()).to_string());
        REQUIRE_EQ("0", (turbo::Duration::seconds(0)).to_string());
        REQUIRE_EQ("0", (turbo::Duration::nanoseconds(0)).to_string());

        REQUIRE_EQ("1ns", (turbo::Duration::nanoseconds(1)).to_string());
        REQUIRE_EQ("1us", (turbo::Duration::microseconds(1)).to_string());
        REQUIRE_EQ("1ms", (turbo::Duration::milliseconds(1)).to_string());
        REQUIRE_EQ("1s", (turbo::Duration::seconds(1)).to_string());
        REQUIRE_EQ("1m", (turbo::Duration::minutes(1)).to_string());
        REQUIRE_EQ("1h", (turbo::Duration::hours(1)).to_string());

        REQUIRE_EQ("1h1m", (turbo::Duration::hours(1) + turbo::Duration::minutes(1)).to_string());
        REQUIRE_EQ("1h1s", (turbo::Duration::hours(1) + turbo::Duration::seconds(1)).to_string());
        REQUIRE_EQ("1m1s", (turbo::Duration::minutes(1) + turbo::Duration::seconds(1)).to_string());

        REQUIRE_EQ("1h0.25s",
                   (turbo::Duration::hours(1) + turbo::Duration::milliseconds(250)).to_string());
        REQUIRE_EQ("1m0.25s",
                   (turbo::Duration::minutes(1) + turbo::Duration::milliseconds(250)).to_string());
        REQUIRE_EQ("1h1m0.25s",
                   (turbo::Duration::hours(1) + turbo::Duration::minutes(1) +
                                          turbo::Duration::milliseconds(250)).to_string());
        REQUIRE_EQ("1h0.0005s",
                   (turbo::Duration::hours(1) + turbo::Duration::microseconds(500)).to_string());
        REQUIRE_EQ("1h0.0000005s",
                   (turbo::Duration::hours(1) + turbo::Duration::nanoseconds(500)).to_string());

        // Subsecond special case.
        REQUIRE_EQ("1.5ns", (turbo::Duration::nanoseconds(1) +
                                                   turbo::Duration::nanoseconds(1) / 2).to_string());
        REQUIRE_EQ("1.25ns", (turbo::Duration::nanoseconds(1) +
                                                    turbo::Duration::nanoseconds(1) / 4).to_string());
        REQUIRE_EQ("1ns", (turbo::Duration::nanoseconds(1) +
                                                 turbo::Duration::nanoseconds(1) / 9).to_string());
        REQUIRE_EQ("1.2us", (turbo::Duration::microseconds(1) +
                                                   turbo::Duration::nanoseconds(200)).to_string());
        REQUIRE_EQ("1.2ms", (turbo::Duration::milliseconds(1) +
                                                   turbo::Duration::microseconds(200)).to_string());
        REQUIRE_EQ("1.0002ms", (turbo::Duration::milliseconds(1) +
                                                      turbo::Duration::nanoseconds(200)).to_string());
        REQUIRE_EQ("1.00001ms", (turbo::Duration::milliseconds(1) +
                                                       turbo::Duration::nanoseconds(10)).to_string());
        REQUIRE_EQ("1.000001ms",
                   (turbo::Duration::milliseconds(1) + turbo::Duration::nanoseconds(1)).to_string());

        // Negative durations.
        REQUIRE_EQ("-1ns", (turbo::Duration::nanoseconds(-1)).to_string());
        REQUIRE_EQ("-1us", (turbo::Duration::microseconds(-1)).to_string());
        REQUIRE_EQ("-1ms", (turbo::Duration::milliseconds(-1)).to_string());
        REQUIRE_EQ("-1s", (turbo::Duration::seconds(-1)).to_string());
        REQUIRE_EQ("-1m", (turbo::Duration::minutes(-1)).to_string());
        REQUIRE_EQ("-1h", (turbo::Duration::hours(-1)).to_string());

        REQUIRE_EQ("-1h1m",
                   (-(turbo::Duration::hours(1) + turbo::Duration::minutes(1))).to_string());
        REQUIRE_EQ("-1h1s",
                   (-(turbo::Duration::hours(1) + turbo::Duration::seconds(1))).to_string());
        REQUIRE_EQ("-1m1s",
                   (-(turbo::Duration::minutes(1) + turbo::Duration::seconds(1))).to_string());

        REQUIRE_EQ("-1ns", (turbo::Duration::nanoseconds(-1)).to_string());
        REQUIRE_EQ("-1.2us", (
                -(turbo::Duration::microseconds(1) + turbo::Duration::nanoseconds(200))).to_string());
        REQUIRE_EQ("-1.2ms", (
                -(turbo::Duration::milliseconds(1) + turbo::Duration::microseconds(200))).to_string());
        REQUIRE_EQ("-1.0002ms", (-(turbo::Duration::milliseconds(1) +
                                                         turbo::Duration::nanoseconds(200))).to_string());
        REQUIRE_EQ("-1.00001ms", (-(turbo::Duration::milliseconds(1) +
                                                          turbo::Duration::nanoseconds(10))).to_string());
        REQUIRE_EQ("-1.000001ms", (-(turbo::Duration::milliseconds(1) +
                                                           turbo::Duration::nanoseconds(1))).to_string());

        //
        // Interesting corner cases.
        //

        const turbo::Duration qns = turbo::Duration::nanoseconds(1) / 4;
        const turbo::Duration max_dur =
                turbo::Duration::seconds(kint64max) + (turbo::Duration::seconds(1) - qns);
        const turbo::Duration min_dur = turbo::Duration::seconds(kint64min);

        REQUIRE_EQ("0.25ns", (qns).to_string());
        REQUIRE_EQ("-0.25ns", (-qns).to_string());
        REQUIRE_EQ("2562047788015215h30m7.99999999975s",
                   (max_dur).to_string());
        REQUIRE_EQ("-2562047788015215h30m8s", (min_dur).to_string());

        // Tests printing full precision from units that print using safe_float_mod
        REQUIRE_EQ("55.00000000025s", (turbo::Duration::seconds(55) + qns).to_string());
        REQUIRE_EQ("55.00000025ms",
                   (turbo::Duration::milliseconds(55) + qns).to_string());
        REQUIRE_EQ("55.00025us", (turbo::Duration::microseconds(55) + qns).to_string());
        REQUIRE_EQ("55.25ns", (turbo::Duration::nanoseconds(55) + qns).to_string());

        // Formatting infinity
        REQUIRE_EQ("inf", (turbo::Duration::infinite()).to_string());
        REQUIRE_EQ("-inf", (-turbo::Duration::infinite()).to_string());

        // Formatting approximately +/- 100 billion years
        const turbo::Duration huge_range = ApproxYears(100000000000);
        REQUIRE_EQ("876000000000000h", (huge_range).to_string());
        REQUIRE_EQ("-876000000000000h", (-huge_range).to_string());

        REQUIRE_EQ("876000000000000h0.999999999s",
                   (huge_range +
                                          (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1))).to_string());
        REQUIRE_EQ("876000000000000h0.9999999995s",
                   (
                           huge_range + (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1) / 2)).to_string());
        REQUIRE_EQ("876000000000000h0.99999999975s",
                   (
                           huge_range + (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1) / 4)).to_string());

        REQUIRE_EQ("-876000000000000h0.999999999s",
                   (-huge_range -
                                          (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1))).to_string());
        REQUIRE_EQ("-876000000000000h0.9999999995s",
                   (
                           -huge_range - (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1) / 2)).to_string());
        REQUIRE_EQ("-876000000000000h0.99999999975s",
                   (
                           -huge_range - (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1) / 4)).to_string());
    }

    TEST_CASE("Duration, parse_duration") {

        turbo::Duration d;

        // No specified unit. Should only work for zero and infinity.
        REQUIRE(d.parse_duration("0"));
        REQUIRE_EQ(turbo::Duration::zero(), d);
        REQUIRE(d.parse_duration("+0"));
        REQUIRE_EQ(turbo::Duration::zero(), d);
        REQUIRE(d.parse_duration("-0"));
        REQUIRE_EQ(turbo::Duration::zero(), d);

        REQUIRE(d.parse_duration("inf"));
        REQUIRE_EQ(turbo::Duration::infinite(), d);
        REQUIRE(d.parse_duration("+inf"));
        REQUIRE_EQ(turbo::Duration::infinite(), d);
        REQUIRE(d.parse_duration("-inf"));
        REQUIRE_EQ(-turbo::Duration::infinite(), d);
        REQUIRE_FALSE(d.parse_duration("infBlah"));

        // Illegal input forms.
        REQUIRE_FALSE(d.parse_duration(""));
        REQUIRE_FALSE(d.parse_duration("0.0"));
        REQUIRE_FALSE(d.parse_duration(".0"));
        REQUIRE_FALSE(d.parse_duration("."));
        REQUIRE_FALSE(d.parse_duration("01"));
        REQUIRE_FALSE(d.parse_duration("1"));
        REQUIRE_FALSE(d.parse_duration("-1"));
        REQUIRE_FALSE(d.parse_duration("2"));
        REQUIRE_FALSE(d.parse_duration("2 s"));
        REQUIRE_FALSE(d.parse_duration(".s"));
        REQUIRE_FALSE(d.parse_duration("-.s"));
        REQUIRE_FALSE(d.parse_duration("s"));
        REQUIRE_FALSE(d.parse_duration(" 2s"));
        REQUIRE_FALSE(d.parse_duration("2s "));
        REQUIRE_FALSE(d.parse_duration(" 2s "));
        REQUIRE_FALSE(d.parse_duration("2mt"));
        REQUIRE_FALSE(d.parse_duration("1e3s"));

        // One unit type.
        REQUIRE(d.parse_duration("1ns"));
        REQUIRE_EQ(turbo::Duration::nanoseconds(1), d);
        REQUIRE(d.parse_duration("1us"));
        REQUIRE_EQ(turbo::Duration::microseconds(1), d);
        REQUIRE(d.parse_duration("1ms"));
        REQUIRE_EQ(turbo::Duration::milliseconds(1), d);
        REQUIRE(d.parse_duration("1s"));
        REQUIRE_EQ(turbo::Duration::seconds(1), d);
        REQUIRE(d.parse_duration("2m"));
        REQUIRE_EQ(turbo::Duration::minutes(2), d);
        REQUIRE(d.parse_duration("2h"));
        REQUIRE_EQ(turbo::Duration::hours(2), d);

        // Huge counts of a unit.
        REQUIRE(d.parse_duration("9223372036854775807us"));
        REQUIRE_EQ(turbo::Duration::microseconds(9223372036854775807), d);
        REQUIRE(d.parse_duration("-9223372036854775807us"));
        REQUIRE_EQ(turbo::Duration::microseconds(-9223372036854775807), d);

        // Multiple units.
        REQUIRE(d.parse_duration("2h3m4s"));
        REQUIRE_EQ(turbo::Duration::hours(2) + turbo::Duration::minutes(3) + turbo::Duration::seconds(4), d);
        REQUIRE(d.parse_duration("3m4s5us"));
        REQUIRE_EQ(turbo::Duration::minutes(3) + turbo::Duration::seconds(4) + turbo::Duration::microseconds(5), d);
        REQUIRE(d.parse_duration("2h3m4s5ms6us7ns"));
        REQUIRE_EQ(turbo::Duration::hours(2) + turbo::Duration::minutes(3) + turbo::Duration::seconds(4) +
                   turbo::Duration::milliseconds(5) + turbo::Duration::microseconds(6) +
                   turbo::Duration::nanoseconds(7),
                   d);

        // Multiple units out of order.
        REQUIRE(d.parse_duration("2us3m4s5h"));
        REQUIRE_EQ(turbo::Duration::hours(5) + turbo::Duration::minutes(3) + turbo::Duration::seconds(4) +
                   turbo::Duration::microseconds(2),
                   d);

        // Fractional values of units.
        REQUIRE(d.parse_duration("1.5ns"));
        REQUIRE_EQ(1.5 * turbo::Duration::nanoseconds(1), d);
        REQUIRE(d.parse_duration("1.5us"));
        REQUIRE_EQ(1.5 * turbo::Duration::microseconds(1), d);
        REQUIRE(d.parse_duration("1.5ms"));
        REQUIRE_EQ(1.5 * turbo::Duration::milliseconds(1), d);
        REQUIRE(d.parse_duration("1.5s"));
        REQUIRE_EQ(1.5 * turbo::Duration::seconds(1), d);
        REQUIRE(d.parse_duration("1.5m"));
        REQUIRE_EQ(1.5 * turbo::Duration::minutes(1), d);
        REQUIRE(d.parse_duration("1.5h"));
        REQUIRE_EQ(1.5 * turbo::Duration::hours(1), d);

        // Huge fractional counts of a unit.
        REQUIRE(d.parse_duration("0.4294967295s"));
        REQUIRE_EQ(turbo::Duration::nanoseconds(429496729) + turbo::Duration::nanoseconds(1) / 2, d);
        REQUIRE(d.parse_duration("0.429496729501234567890123456789s"));
        REQUIRE_EQ(turbo::Duration::nanoseconds(429496729) + turbo::Duration::nanoseconds(1) / 2, d);

        // Negative durations.
        REQUIRE(d.parse_duration("-1s"));
        REQUIRE_EQ(turbo::Duration::seconds(-1), d);
        REQUIRE(d.parse_duration("-1m"));
        REQUIRE_EQ(turbo::Duration::minutes(-1), d);
        REQUIRE(d.parse_duration("-1h"));
        REQUIRE_EQ(turbo::Duration::hours(-1), d);

        REQUIRE(d.parse_duration("-1h2s"));
        REQUIRE_EQ(-(turbo::Duration::hours(1) + turbo::Duration::seconds(2)), d);
        REQUIRE_FALSE(d.parse_duration("1h-2s"));
        REQUIRE_FALSE(d.parse_duration("-1h-2s"));
        REQUIRE_FALSE(d.parse_duration("-1h -2s"));
    }

    TEST_CASE("Duration, FormatParseRoundTrip") {
#define TEST_PARSE_ROUNDTRIP(d)                \
  do {                                         \
    std::string s = (d).to_string();   \
    turbo::Duration dur;                        \
    REQUIRE(dur.parse_duration(s)); \
    REQUIRE_EQ(d, dur);                         \
  } while (0)

        TEST_PARSE_ROUNDTRIP(turbo::Duration::nanoseconds(1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::microseconds(1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::milliseconds(1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::seconds(1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::minutes(1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::hours(1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::hours(1) + turbo::Duration::nanoseconds(2));

        TEST_PARSE_ROUNDTRIP(turbo::Duration::nanoseconds(-1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::microseconds(-1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::milliseconds(-1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::seconds(-1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::minutes(-1));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::hours(-1));

        TEST_PARSE_ROUNDTRIP(turbo::Duration::hours(-1) + turbo::Duration::nanoseconds(2));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::hours(1) + turbo::Duration::nanoseconds(-2));
        TEST_PARSE_ROUNDTRIP(turbo::Duration::hours(-1) + turbo::Duration::nanoseconds(-2));

        TEST_PARSE_ROUNDTRIP(turbo::Duration::nanoseconds(1) +turbo::Duration::nanoseconds(1) / 4);  // 1.25ns

        const turbo::Duration huge_range = ApproxYears(100000000000);
        TEST_PARSE_ROUNDTRIP(huge_range);
        TEST_PARSE_ROUNDTRIP(huge_range + (turbo::Duration::seconds(1) - turbo::Duration::nanoseconds(1)));

#undef TEST_PARSE_ROUNDTRIP
    }

}  // namespace
