// Copyright 2023 The Elastic-AI Authors.
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
//
// Created by jeff on 24-1-14.
//

#ifndef TURBO_TIMES_DURATION_H_
#define TURBO_TIMES_DURATION_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <cstdint>
#include <ctime>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <string_view>
#include "turbo/platform/port.h"

namespace turbo {
    class Duration;
    namespace time_internal {

        int64_t safe_int_mod(bool satq, Duration num, Duration den, Duration *rem);

        constexpr int64_t GetRepHi(Duration d);

        constexpr uint32_t GetRepLo(Duration d);

        constexpr Duration MakeDuration(int64_t hi,
                                        uint32_t lo);

        constexpr Duration MakeDuration(int64_t hi,
                                        int64_t lo);

        inline Duration make_pos_double_duration(double n);

        constexpr int64_t kTicksPerNanosecond = 4;
        constexpr int64_t kTicksPerSecond = 1000 * 1000 * 1000 * kTicksPerNanosecond;

        template<std::intmax_t N>
        constexpr Duration from_int64(int64_t v,
                                      std::ratio<1, N>);

        constexpr Duration from_int64(int64_t v,
                                      std::ratio<60>);

        constexpr Duration from_int64(int64_t v,
                                      std::ratio<3600>);

        template<typename T>
        using EnableIfIntegral = typename std::enable_if<
                std::is_integral<T>::value || std::is_enum<T>::value, int>::type;
        template<typename T>
        using EnableIfFloat =
                typename std::enable_if<std::is_floating_point<T>::value, int>::type;
        template<typename T>
        using SelectInt64Double = typename std::enable_if_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, double>, T>;

        template<typename T>
        using EnableIfFoundation = typename std::enable_if_t<
                std::is_integral_v<T> || std::is_enum_v<T> || std::is_floating_point_v<T>, int>;

        // IsValidRep64<T>(0) is true if the expression `int64_t{std::declval<T>()}` is
        // valid. That is, if a T can be assigned to an int64_t without narrowing.
        template<typename T>
        constexpr auto IsValidRep64(int) -> decltype(int64_t{std::declval<T>()} == 0) {
            return true;
        }

        template<typename T>
        constexpr auto IsValidRep64(char) -> bool {
            return false;
        }

    }  // namespace time_internal
    /**
     * @ingroup turbo_times_duration
     * @brief The `turbo::Duration` class represents a signed, fixed-length amount of time.
     *        A `Duration` is generated using a unit-specific factory function, or is
     *        the result of subtracting one `turbo::Time` from another. Durations behave
     *        like unit-safe integers and they support all the natural integer-like
     *        arithmetic operations. Arithmetic overflows and saturates at +/- infinity.
     *        `Duration` should be passed by value rather than const reference.
     *
     *        Factory functions `nanoseconds()`, `microseconds()`, `milliseconds()`,
     *        `seconds()`, `minutes()`, `hours()` and `infinite()` allow for
     *        creation of constexpr `Duration` values
     *
     *        Examples:
     *        @code
     *        constexpr turbo::Duration ten_ns = turbo::Duration::nanoseconds(10);
     *        constexpr turbo::Duration min = turbo::Duration::minutes(1);
     *        constexpr turbo::Duration hour = turbo::Duration::hours(1);
     *        turbo::Duration dur = 60 * min;  // dur == hour
     *        turbo::Duration half_sec = turbo::Duration::milliseconds(500);
     *        turbo::Duration quarter_sec = 0.25 * turbo::Duration::seconds(1);
     *        @endcode
     *
     *        `Duration` values can be easily converted to an integral number of units
     *        using the division operator.
     *
     *        Example:
     *        @code
     *        constexpr turbo::Duration dur = turbo::Duration::milliseconds(1500);
     *        int64_t ns = dur / turbo::Duration::nanoseconds(1);   // ns == 1500000000
     *        int64_t ms = dur / turbo::Duration::milliseconds(1);  // ms == 1500
     *        int64_t sec = dur / turbo::Duration::seconds(1);    // sec == 1 (subseconds truncated)
     *        int64_t min = dur / turbo::Duration::minutes(1);    // min == 0
     *        @endcode
     *
     *        See the `safe_int_mod()` and `safe_float_mod()` functions below for details on
     *        how to access the fractional parts of the quotient.
     *
     *        Alternatively, conversions can be performed using helpers such as
     *        `to_int64_microseconds()` and `to_double_seconds()`.
     */
    class Duration {
    public:
        // Value semantics.
        constexpr Duration() = default;

        // Copyable.
#if !defined(__clang__) && defined(_MSC_VER) && _MSC_VER < 1930
        // Explicitly defining the constexpr copy constructor avoids an MSVC bug.
        constexpr Duration(const Duration& d)
            : rep_hi_(d.rep_hi_), rep_lo_(d.rep_lo_) {}
#else

        constexpr Duration(const Duration &d) = default;

#endif

        Duration &operator=(const Duration &d) = default;

        // Compound assignment operators.
        Duration &operator+=(Duration d);

        Duration &operator-=(Duration d);

        Duration &operator*=(int64_t r);

        Duration &operator*=(double r);

        Duration &operator/=(int64_t r);

        Duration &operator/=(double r);

        Duration &operator%=(Duration rhs);

        // Overloads that forward to either the int64_t or double overloads above.
        // Integer operands must be representable as int64_t.
        template<typename T, time_internal::EnableIfIntegral<T> = 0>
        Duration &operator*=(T r) {
            int64_t x = r;
            return *this *= x;
        }

        template<typename T, time_internal::EnableIfIntegral<T> = 0>
        Duration &operator/=(T r) {
            int64_t x = r;
            return *this /= x;
        }

        template<typename T, time_internal::EnableIfFloat<T> = 0>
        Duration &operator*=(T r) {
            double x = r;
            return *this *= x;
        }

        template<typename T, time_internal::EnableIfFloat<T> = 0>
        Duration &operator/=(T r) {
            double x = r;
            return *this /= x;
        }

        template<typename H>
        friend H hash_value(H h, Duration d) {
            return H::combine(std::move(h), d.rep_hi_, d.rep_lo_);
        }

    public:
        ///// attributes /////
        constexpr bool is_infinite() const;

        constexpr bool is_zero() const;

        constexpr bool is_negative() const;

        constexpr bool is_positive() const;

    public:
        ////// getters //////
        template<typename T = int64_t>
        constexpr time_internal::SelectInt64Double<T> to_nanoseconds() const;

        template<typename T = int64_t>
        constexpr time_internal::SelectInt64Double<T> to_microseconds() const;

        template<typename T = int64_t>
        constexpr time_internal::SelectInt64Double<T> to_milliseconds() const;

        template<typename T = int64_t>
        constexpr time_internal::SelectInt64Double<T> to_seconds() const;

        template<typename T = int64_t>
        constexpr time_internal::SelectInt64Double<T> to_minutes() const;

        template<typename T = int64_t>
        constexpr time_internal::SelectInt64Double<T> to_hours() const;

        template<typename T>
        constexpr T to_chrono_duration() const;

        constexpr std::chrono::nanoseconds to_chrono_nanoseconds() const;

        constexpr std::chrono::microseconds to_chrono_microseconds() const;

        constexpr std::chrono::milliseconds to_chrono_milliseconds() const;

        constexpr std::chrono::seconds to_chrono_seconds() const;

        constexpr std::chrono::minutes to_chrono_minutes() const;

        constexpr std::chrono::hours to_chrono_hours() const;

        /**
         * @ingroup turbo_times_time_point
         * @brief  similar to `duration_from_timespec()`
         * @see `duration_from_timespec()`
         * @param d
         * @return
         */
        constexpr timespec to_timespec() const;

        /**
         * @ingroup turbo_times_time_point
         * @brief  similar to `duration_from_timespec()`
         * @see `duration_from_timespec()`
         * @param d
         * @return
         */
        constexpr timeval to_timeval() const;

        /////// transform ///////
        /**
         * @brief Divides a `Duration` numerator into a fractional number of units of a
         *        `Duration` denominator.
         *        See also the notes on `infinite()` below regarding the behavior of
         *        division involving zero and infinite durations.
         *        @see `turbo::safe_int_mod()` for a version that returns the quotient and remainder.
         *        Example:
         *        @code
         *        double d = turbo::Duration::milliseconds(1500).safe_float_mod(turbo::Duration::seconds(1));
         *        // d == 1.5
         *        @endcode
         * @param den the denominator
         * @return the quotient
         */
        constexpr double safe_float_mod(Duration den) const;

        /**
         * @ingroup turbo_times_duration
         * @brief Truncates a duration (toward zero) to a multiple of a non-zero unit.
         *        Example:
         *        @code
         *        turbo::Duration d = turbo::Duration::nanoseconds(123456789);
         *        turbo::Duration a = turbo::trunc(d, turbo::Duration::microseconds(1));  // 123456us
         *        @endcode
         * @param d
         * @param unit
         * @return
         */
        Duration trunc(Duration unit) const;

        /**
         * @ingroup turbo_times_duration
         * @brief Floors a duration using the passed duration unit to its largest value not
         *        greater than the duration.
         *        Example:
         *        @code
         *        turbo::Duration d = turbo::Duration::nanoseconds(123456789);
         *        turbo::Duration b = turbo::floor(d, turbo::Duration::microseconds(1));  // 123456us
         *        @endcode
         * @param d
         * @param unit
         * @return
         */
        Duration floor(Duration unit) const;

        /**
         * @ingroup turbo_times_duration
         * @brief Returns the ceiling of a duration using the passed duration unit to its
         *        smallest value not less than the duration.
         *        Example:
         *        @code
         *        turbo::Duration d = turbo::Duration::nanoseconds(123456789);
         *        turbo::Duration c = turbo::ceil(d, turbo::Duration::microseconds(1));   // 123457us
         *        @endcode
         * @param d
         * @param unit
         * @return
         */
        Duration ceil(Duration unit) const;

        /**
         * @ingroup turbo_times_duration
         * @brief Returns the nearest integral mod of a duration using the passed duration
         *       unit to its nearest value.
         *       Example:
         *       @code
         *       turbo::Duration d = turbo::Duration::nanoseconds(123456789);
         *       turbo::Duration e = d.fraction(turbo::Duration::microseconds(1));  // 789ns
         *       @endcode
         *       @see `turbo::safe_int_mod()` for a version that returns the quotient and remainder.
         * @param unit
         * @return
         */
        Duration fraction(Duration unit) const;

        // Returns an infinite Duration with the opposite sign.
        // REQUIRES: d.is_infinite()
        constexpr Duration opposite_infinity() const;

        /**
     * @ingroup turbo_times_duration
     * @brief Returns a string representing the duration in the form "72h3m0.5s".
     *        Returns "inf" or "-inf" for +/- `Duration::infinite()`.
     * @param d
     * @return
     */
        std::string to_string() const;

        /**
     * @ingroup turbo_times_duration
     * @brief Parses a duration string consisting of a possibly signed sequence of
     *        decimal numbers, each with an optional fractional part and a unit
     *        suffix.  The valid suffixes are "ns", "us" "ms", "s", "m", and "h".
     *        Simple examples include "300ms", "-1.5h", and "2h45m".  Parses "0" as
     *        `Duration::zero()`. Parses "inf" and "-inf" as +/- `Duration::infinite()`.
     * @param dur_string
     * @param d
     * @return
     */
        bool parse_duration(std::string_view dur_string);

    public:
        /////// creators ///////

        template<typename T, time_internal::EnableIfFoundation<T> = 0>
        static constexpr Duration nanoseconds(T n);

        template<typename T, time_internal::EnableIfFoundation<T> = 0>
        static constexpr Duration microseconds(T n);

        template<typename T, time_internal::EnableIfFoundation<T> = 0>
        static constexpr Duration milliseconds(T n);

        template<typename T, time_internal::EnableIfFoundation<T> = 0>
        static constexpr Duration seconds(T n);

        template<typename T, time_internal::EnableIfFoundation<T> = 0>
        static constexpr Duration minutes(T n);

        template<typename T, time_internal::EnableIfFoundation<T> = 0>
        static constexpr Duration hours(T n);

        // Converts a std::chrono::duration to an turbo::Duration.
        template<typename Rep, typename Period>
        static constexpr Duration from_chrono_duration(const std::chrono::duration<Rep, Period> &d);

        static constexpr Duration from_chrono(const std::chrono::nanoseconds &d);

        static constexpr Duration from_chrono(const std::chrono::microseconds &d);

        static constexpr Duration from_chrono(const std::chrono::milliseconds &d);

        static constexpr Duration from_chrono(const std::chrono::seconds &d);

        static constexpr Duration from_chrono(const std::chrono::minutes &d);

        static constexpr Duration from_chrono(const std::chrono::hours &d);

        /**
         * @ingroup turbo_times_time_point
         * @brief Converts a `timespec` to a `Duration`.
         * @param ts
         * @return
         */
        static constexpr Duration from_timespec(timespec ts);

        /**
         * @ingroup turbo_times_time_point
         * @brief  similar to `duration_from_timespec()`
         * @see `duration_from_timespec()`
         * @param tv
         * @return
         */
        static constexpr Duration from_timeval(timeval tv);

        /**
         * @brief Returns an infinite `Duration`.  To get a `Duration` representing negative
         *        infinity, use `-turbo::Duration::infinite()`.
         *
         *        Duration arithmetic overflows to +/- infinity and saturates. In general,
         *        arithmetic with `Duration` infinities is similar to IEEE 754 infinities
         *        except where IEEE 754 NaN would be involved, in which case +/-
         *        `turbo::Duration::infinite()` is used in place of a "nan" Duration.
         *        Examples:
         *        @code
         *        constexpr turbo::Duration inf = turbo::Duration::infinite();
         *        const turbo::Duration d = ... any finite duration ...
         *
         *        inf == inf + inf
         *        inf == inf + d
         *        inf == inf - inf
         *        -inf == d - inf
         *
         *        inf == d * 1e100
         *        inf == inf / 2
         *        0 == d / inf
         *        INT64_MAX == inf / d
         *
         *        d < inf
         *        -inf < d
         *
         *        // Division by zero returns infinity, or INT64_MIN/MAX where appropriate.
         *        inf == d / 0
         *        INT64_MAX == d / turbo::Duration::zero()
         *        @endcode
         * @return
         */
        static constexpr Duration infinite();

        static constexpr Duration zero();

        constexpr Duration abs() const;

    private:
        template<typename Ratio>
        constexpr int64_t to_int64(Ratio) const;

        // Fastpath implementations for the 6 common duration units.
        constexpr int64_t to_int64(std::nano) const;

        constexpr int64_t to_int64(std::micro) const;

        constexpr int64_t to_int64(std::milli) const;

        constexpr int64_t to_int64(std::ratio<1>) const;

        constexpr int64_t to_int64(std::ratio<60>) const;

        constexpr int64_t to_int64(std::ratio<3600>) const;


    private:
        friend constexpr int64_t time_internal::GetRepHi(Duration d);

        friend constexpr uint32_t time_internal::GetRepLo(Duration d);

        friend constexpr Duration time_internal::MakeDuration(int64_t hi,
                                                              uint32_t lo);

        constexpr Duration(int64_t hi, uint32_t lo) : rep_hi_(hi), rep_lo_(lo) {}

        int64_t rep_hi_{0};
        uint32_t rep_lo_{0};
    };


    // Relational Operators
    constexpr bool operator<(Duration lhs, Duration rhs);

    constexpr bool operator>(Duration lhs, Duration rhs) {
        return rhs < lhs;
    }

    constexpr bool operator>=(Duration lhs, Duration rhs) {
        return !(lhs < rhs);
    }

    constexpr bool operator<=(Duration lhs, Duration rhs) {
        return !(rhs < lhs);
    }

    constexpr bool operator==(Duration lhs, Duration rhs);

    constexpr bool operator!=(Duration lhs, Duration rhs) {
        return !(lhs == rhs);
    }

    // Additive Operators
    constexpr Duration operator-(Duration d);

    inline Duration operator+(Duration lhs, Duration rhs) {
        auto tmp = lhs;
        return tmp += rhs;
    }

    inline Duration operator-(Duration lhs, Duration rhs) {
        auto tmp = lhs;
        return tmp -= rhs;
    }

    // Multiplicative Operators
    // Integer operands must be representable as int64_t.
    template<typename T>
    Duration operator*(Duration lhs, T rhs) {
        auto tmp = lhs;
        return tmp *= rhs;
    }

    template<typename T>
    Duration operator*(T lhs, Duration rhs) {
        auto tmp = rhs;
        return tmp *= lhs;
    }

    template<typename T>
    Duration operator/(Duration lhs, T rhs) {
        auto tmp = lhs;
        return tmp /= rhs;
    }

    inline int64_t operator/(Duration lhs, Duration rhs) {
        auto tmp = lhs;
        return time_internal::safe_int_mod(true, tmp, rhs,
                                           &tmp);  // trunc towards zero
    }

    inline Duration operator%(Duration lhs, Duration rhs) {
        auto tmp = lhs;
        return tmp %= rhs;
    }

    constexpr bool Duration::is_infinite() const {
        return rep_lo_ == ~uint32_t{0};
    }

    constexpr bool Duration::is_zero() const {
        return rep_hi_ == 0 && rep_lo_ == 0;
    }

    constexpr bool Duration::is_negative() const {
        return *this < zero();
    }

    constexpr bool Duration::is_positive() const {
        return *this > zero();
    }

    /**
     * @ingroup turbo_times_duration
     * @brief Divides a numerator `Duration` by a denominator `Duration`, returning the
     *        quotient and remainder. The remainder always has the same sign as the
     *        numerator. The returned quotient and remainder respect the identity:
     *        numerator = denominator * quotient + remainder
     *        Returned quotients are capped to the range of `int64_t`, with the difference
     *        spilling into the remainder to uphold the above identity. This means that the
     *        remainder returned could differ from the remainder returned by
     *        `Duration::operator%` for huge quotients.
     *        See also the notes on `infinite()` below regarding the behavior of
     *        division involving zero and infinite durations.
     *        Example:
     *        @code
     *        constexpr turbo::Duration a = turbo::Duration::seconds(std::numeric_limits<int64_t>::max());  // big
     *        constexpr turbo::Duration b = turbo::Duration::nanoseconds(1);       // small
     *        turbo::Duration rem = a % b;
     *        // rem == turbo::Duration::zero()
     *        // Here, q would overflow int64_t, so rem accounts for the difference.
     *        int64_t q = turbo::safe_int_mod(a, b, &rem);
     *        // q == std::numeric_limits<int64_t>::max(), rem == a - b * q
     *        @endcode
     * @param num the numerator
     * @param den the denominator
     * @param rem the remainder
     * @return the quotient
     */
    inline int64_t safe_int_mod(Duration num, Duration den, Duration *rem) {
        return time_internal::safe_int_mod(true, num, den,
                                           rem);  // trunc towards zero
    }

    // Output stream operator.
    inline std::ostream &operator<<(std::ostream &os, Duration d) {
        return os << d.to_string();
    }

    template<typename T>
    constexpr time_internal::SelectInt64Double<T> Duration::to_nanoseconds() const {
        if constexpr (std::is_same_v<T, int64_t>) {
            if (rep_hi_ >= 0 &&
                rep_hi_ >> 33 == 0) {
                return (rep_hi_ * 1000 * 1000 * 1000) +
                       (rep_lo_ / time_internal::kTicksPerNanosecond);
            }
            return *this / nanoseconds(1);
        } else {
            return safe_float_mod(nanoseconds(1));
        }
    }

    template<typename T>
    constexpr time_internal::SelectInt64Double<T> Duration::to_microseconds() const {
        if constexpr (std::is_same_v<T, int64_t>) {
            if (rep_hi_ >= 0 &&
                rep_hi_ >> 43 == 0) {
                return (rep_hi_ * 1000 * 1000) +
                       (rep_lo_ / (time_internal::kTicksPerNanosecond * 1000));
            }
            return *this / microseconds(1);
        } else {
            return safe_float_mod(microseconds(1));
        }
    }

    template<typename T>
    constexpr time_internal::SelectInt64Double<T> Duration::to_milliseconds() const {
        if constexpr (std::is_same_v<T, int64_t>) {
            if (rep_hi_ >= 0 &&
                rep_hi_ >> 53 == 0) {
                return (rep_hi_ * 1000) +
                       (rep_lo_ / (time_internal::kTicksPerNanosecond * 1000 * 1000));
            }
            return *this / milliseconds(1);
        } else {
            return safe_float_mod(milliseconds(1));
        }
    }

    template<typename T>
    constexpr time_internal::SelectInt64Double<T> Duration::to_seconds() const {
        if constexpr (std::is_same_v<T, int64_t>) {
            int64_t hi = rep_hi_;
            if (is_infinite()) return hi;
            if (hi < 0 && rep_lo_ != 0) ++hi;
            return hi;
        } else {
            return safe_float_mod(seconds(1));
        }
    }

    template<typename T>
    constexpr time_internal::SelectInt64Double<T> Duration::to_minutes() const {
        if constexpr (std::is_same_v<T, int64_t>) {
            int64_t hi = rep_hi_;
            if (is_infinite()) return hi;
            if (hi < 0 && rep_lo_ != 0) ++hi;
            return hi / 60;
        } else {
            return safe_float_mod(minutes(1));
        }
    }

    template<typename T>
    constexpr time_internal::SelectInt64Double<T> Duration::to_hours() const {
        if constexpr (std::is_same_v<T, int64_t>) {
            int64_t hi = rep_hi_;
            if (is_infinite()) return hi;
            if (hi < 0 && rep_lo_ != 0) ++hi;
            return hi / (60 * 60);
        } else {
            return safe_float_mod(hours(1));
        }
    }

    constexpr timespec Duration::to_timespec() const {
        timespec ts{0, 0};
        if (!is_infinite()) {
            int64_t rep_hi = rep_hi_;
            uint32_t rep_lo = rep_lo_;
            if (rep_hi < 0) {
                // Tweak the fields so that unsigned division of rep_lo
                // maps to truncation (towards zero) for the timespec.
                rep_lo += time_internal::kTicksPerNanosecond - 1;
                if (rep_lo >= time_internal::kTicksPerSecond) {
                    rep_hi += 1;
                    rep_lo -= time_internal::kTicksPerSecond;
                }
            }
            ts.tv_sec = static_cast<decltype(ts.tv_sec)>(rep_hi);
            if (ts.tv_sec == rep_hi) {  // no time_t narrowing
                ts.tv_nsec = rep_lo / time_internal::kTicksPerNanosecond;
                return ts;
            }
        }
        if (*this >= zero()) {
            ts.tv_sec = std::numeric_limits<time_t>::max();
            ts.tv_nsec = 1000 * 1000 * 1000 - 1;
        } else {
            ts.tv_sec = std::numeric_limits<time_t>::min();
            ts.tv_nsec = 0;
        }
        return ts;
    }

    constexpr timeval Duration::to_timeval() const {
        timeval tv{0, 0};
        timespec ts = to_timespec();
        if (ts.tv_sec < 0) {
            // Tweak the fields so that positive division of tv_nsec
            // maps to truncation (towards zero) for the timeval.
            ts.tv_nsec += 1000 - 1;
            if (ts.tv_nsec >= 1000 * 1000 * 1000) {
                ts.tv_sec += 1;
                ts.tv_nsec -= 1000 * 1000 * 1000;
            }
        }
        tv.tv_sec = static_cast<decltype(tv.tv_sec)>(ts.tv_sec);
        if (tv.tv_sec != ts.tv_sec) {  // narrowing
            if (ts.tv_sec < 0) {
                tv.tv_sec = std::numeric_limits<decltype(tv.tv_sec)>::min();
                tv.tv_usec = 0;
            } else {
                tv.tv_sec = std::numeric_limits<decltype(tv.tv_sec)>::max();
                tv.tv_usec = 1000 * 1000 - 1;
            }
            return tv;
        }
        tv.tv_usec = static_cast<int>(ts.tv_nsec / 1000);  // suseconds_t
        return tv;
    }

    template<typename T, time_internal::EnableIfFoundation<T>>
    constexpr Duration Duration::nanoseconds(T n) {
        if constexpr (std::is_floating_point_v<T>) {
            return n * time_internal::from_int64(1, std::nano{});
        }
        return time_internal::from_int64(n, std::nano{});
    }

    template<typename T, time_internal::EnableIfFoundation<T>>
    constexpr Duration Duration::microseconds(T n) {
        if constexpr (std::is_floating_point_v<T>) {
            return n * time_internal::from_int64(1, std::micro{});
        }
        return time_internal::from_int64(n, std::micro{});
    }

    template<typename T, time_internal::EnableIfFoundation<T>>
    constexpr Duration Duration::milliseconds(T n) {
        if constexpr (std::is_floating_point_v<T>) {
            return n * time_internal::from_int64(1, std::milli{});
        }
        return time_internal::from_int64(n, std::milli{});
    }

    template<typename T, time_internal::EnableIfFoundation<T>>
    constexpr Duration Duration::seconds(T n) {
        if constexpr (std::is_floating_point_v<T>) {
            if (n >= 0) {  // Note: `NaN >= 0` is false.
                if (n >= static_cast<T>((std::numeric_limits<int64_t>::max)())) {
                    return Duration::infinite();
                }
                return time_internal::make_pos_double_duration(n);
            } else {
                if (std::isnan(n))
                    return std::signbit(n) ? -Duration::infinite() : Duration::infinite();
                if (n <= (std::numeric_limits<int64_t>::min)()) return -Duration::infinite();
                return -time_internal::make_pos_double_duration(-n);
            }
        }
        return time_internal::from_int64(n, std::ratio<1>{});
    }

    template<typename T, time_internal::EnableIfFoundation<T>>
    constexpr Duration Duration::minutes(T n) {
        if constexpr (std::is_floating_point_v<T>) {
            return n * time_internal::from_int64(1, std::ratio<60>{});
        }
        return time_internal::from_int64(n, std::ratio<60>{});
    }

    template<typename T, time_internal::EnableIfFoundation<T>>
    constexpr Duration Duration::hours(T n) {
        if constexpr (std::is_floating_point_v<T>) {
            return n * time_internal::from_int64(1, std::ratio<3600>{});
        }
        return time_internal::from_int64(n, std::ratio<3600>{});
    }

    constexpr Duration Duration::infinite() {
        return time_internal::MakeDuration((std::numeric_limits<int64_t>::max)(),
                                           ~uint32_t{0});
    }

    constexpr Duration Duration::zero() {
        return Duration{};
    }

    constexpr Duration Duration::abs() const {
        return is_negative() ? -*this : *this;
    }

    constexpr double Duration::safe_float_mod(Duration den) const {
        // Arithmetic with infinity is sticky.
        if (is_infinite() || den == Duration::zero()) {
            return (is_negative()) == (den < Duration::zero())
                   ? std::numeric_limits<double>::infinity()
                   : -std::numeric_limits<double>::infinity();
        }
        if (den.is_infinite()) return 0.0;

        double a = static_cast<double>(rep_hi_) * time_internal::kTicksPerSecond + rep_lo_;
        double b = static_cast<double>(time_internal::GetRepHi(den)) * time_internal::kTicksPerSecond +
                   time_internal::GetRepLo(den);
        return a / b;
    }

    template<typename Ratio>
    constexpr int64_t Duration::to_int64(Ratio) const {
        // Note: This may be used on MSVC, which may have a system_clock period of
        // std::ratio<1, 10 * 1000 * 1000>
        return (*this * Ratio::den / Ratio::num).to_seconds();
    }

    // Fastpath implementations for the 6 common duration units.
    constexpr int64_t Duration::to_int64(std::nano) const {
        return to_nanoseconds();
    }

    constexpr int64_t Duration::to_int64(std::micro) const {
        return to_microseconds();
    }

    constexpr int64_t Duration::to_int64(std::milli) const {
        return to_milliseconds();
    }

    constexpr int64_t Duration::to_int64(std::ratio<1>) const {
        return to_seconds();
    }

    constexpr int64_t Duration::to_int64(std::ratio<60>) const {
        return to_minutes();
    }

    constexpr int64_t Duration::to_int64(std::ratio<3600>) const {
        return to_hours();
    }

    // Converts an turbo::Duration to a chrono duration of type T.
    template<typename T>
    constexpr T Duration::to_chrono_duration() const {
        using Rep = typename T::rep;
        using Period = typename T::period;
        static_assert(time_internal::IsValidRep64<Rep>(0), "duration::rep is invalid");
        if (is_infinite())
            return *this < Duration::zero() ? (T::min)() : (T::max)();
        const auto v = to_int64(Period{});
        if (v > (std::numeric_limits<Rep>::max)()) return (T::max)();
        if (v < (std::numeric_limits<Rep>::min)()) return (T::min)();
        return T{v};
    }

    constexpr std::chrono::nanoseconds Duration::to_chrono_nanoseconds() const {
        return to_chrono_duration<std::chrono::nanoseconds>();
    }

    constexpr std::chrono::microseconds Duration::to_chrono_microseconds() const {
        return to_chrono_duration<std::chrono::microseconds>();
    }

    constexpr std::chrono::milliseconds Duration::to_chrono_milliseconds() const {
        return to_chrono_duration<std::chrono::milliseconds>();
    }

    constexpr std::chrono::seconds Duration::to_chrono_seconds() const {
        return to_chrono_duration<std::chrono::seconds>();
    }

    constexpr std::chrono::minutes Duration::to_chrono_minutes() const {
        return to_chrono_duration<std::chrono::minutes>();
    }

    constexpr std::chrono::hours Duration::to_chrono_hours() const {
        return to_chrono_duration<std::chrono::hours>();
    }

    // Returns an infinite Duration with the opposite sign.
    // REQUIRES: d.is_infinite()
    constexpr Duration Duration::opposite_infinity() const {
        return rep_hi_ < 0
               ? time_internal::MakeDuration((std::numeric_limits<int64_t>::max)(), ~uint32_t{0})
               : time_internal::MakeDuration((std::numeric_limits<int64_t>::min)(),
                                             ~uint32_t{0});
    }


    namespace time_internal {
        // Creates a Duration with a given representation.
        // REQUIRES: hi,lo is a valid representation of a Duration as specified
        // in time/duration.cc.
        constexpr Duration MakeDuration(int64_t hi,
                                        uint32_t lo = 0) {
            return Duration(hi, lo);
        }

        constexpr Duration MakeDuration(int64_t hi,
                                        int64_t lo) {
            return MakeDuration(hi, static_cast<uint32_t>(lo));
        }

        // Make a Duration value from a floating-point number, as long as that number
        // is in the range [ 0 .. numeric_limits<int64_t>::max ), that is, as long as
        // it's positive and can be converted to int64_t without risk of UB.
        inline Duration make_pos_double_duration(double n) {
            const int64_t int_secs = static_cast<int64_t>(n);
            const uint32_t ticks = static_cast<uint32_t>(
                    std::round((n - static_cast<double>(int_secs)) * kTicksPerSecond));
            return ticks < kTicksPerSecond
                   ? MakeDuration(int_secs, ticks)
                   : MakeDuration(int_secs + 1, ticks - kTicksPerSecond);
        }

        // Creates a normalized Duration from an almost-normalized (sec,ticks)
        // pair. sec may be positive or negative.  ticks must be in the range
        // -kTicksPerSecond < *ticks < kTicksPerSecond.  If ticks is negative it
        // will be normalized to a positive value in the resulting Duration.
        constexpr Duration MakeNormalizedDuration(
                int64_t sec, int64_t ticks) {
            return (ticks < 0) ? MakeDuration(sec - 1, ticks + kTicksPerSecond)
                               : MakeDuration(sec, ticks);
        }

        // Provide access to the Duration representation.
        constexpr int64_t GetRepHi(Duration d) {
            return d.rep_hi_;
        }

        constexpr uint32_t GetRepLo(Duration d) {
            return d.rep_lo_;
        }

        // Returns (-n)-1 (equivalently -(n+1)) without avoidable overflow.
        constexpr int64_t NegateAndSubtractOne(
                int64_t n) {
            // Note: Good compilers will optimize this expression to ~n when using
            // a two's-complement representation (which is required for int64_t).
            return (n < 0) ? -(n + 1) : (-n) - 1;
        }

        template<std::intmax_t N>
        constexpr Duration from_int64(int64_t v,
                                      std::ratio<1, N>) {
            static_assert(0 < N && N <= 1000000000, "Unsupported ratio");
            // Subsecond ratios cannot overflow.
            return MakeNormalizedDuration(
                    v / N, v % N * kTicksPerNanosecond * 1000 * 1000 * 1000 / N);
        }

        constexpr Duration from_int64(int64_t v,
                                      std::ratio<60>) {
            return (v <= (std::numeric_limits<int64_t>::max)() / 60 &&
                    v >= (std::numeric_limits<int64_t>::min)() / 60)
                   ? MakeDuration(v * 60)
                   : v > 0 ? Duration::infinite() : -Duration::infinite();
        }

        constexpr Duration from_int64(int64_t v,
                                      std::ratio<3600>) {
            return (v <= (std::numeric_limits<int64_t>::max)() / 3600 &&
                    v >= (std::numeric_limits<int64_t>::min)() / 3600)
                   ? MakeDuration(v * 3600)
                   : v > 0 ? Duration::infinite() : -Duration::infinite();
        }

    }  // namespace time_internal
    constexpr bool operator<(Duration lhs,
                             Duration rhs) {
        return time_internal::GetRepHi(lhs) != time_internal::GetRepHi(rhs)
               ? time_internal::GetRepHi(lhs) < time_internal::GetRepHi(rhs)
               : time_internal::GetRepHi(lhs) == (std::numeric_limits<int64_t>::min)()
                 ? time_internal::GetRepLo(lhs) + 1 <
                   time_internal::GetRepLo(rhs) + 1
                 : time_internal::GetRepLo(lhs) < time_internal::GetRepLo(rhs);
    }

    constexpr bool operator==(Duration lhs,
                              Duration rhs) {
        return time_internal::GetRepHi(lhs) == time_internal::GetRepHi(rhs) &&
               time_internal::GetRepLo(lhs) == time_internal::GetRepLo(rhs);
    }

    constexpr Duration operator-(Duration d) {
        // This is a little interesting because of the special cases.
        //
        // If rep_lo_ is zero, we have it easy; it's safe to negate rep_hi_, we're
        // dealing with an integral number of seconds, and the only special case is
        // the maximum negative finite duration, which can't be negated.
        //
        // Infinities stay infinite, and just change direction.
        //
        // Finally we're in the case where rep_lo_ is non-zero, and we can borrow
        // a second's worth of ticks and avoid overflow (as negating int64_t-min + 1
        // is safe).
        return time_internal::GetRepLo(d) == 0
               ? time_internal::GetRepHi(d) ==
                 (std::numeric_limits<int64_t>::min)()
                 ? Duration::infinite()
                 : time_internal::MakeDuration(-time_internal::GetRepHi(d))
               : d.is_infinite()
                 ? d.opposite_infinity()
                 : time_internal::MakeDuration(
                                time_internal::NegateAndSubtractOne(
                                        time_internal::GetRepHi(d)),
                                time_internal::kTicksPerSecond -
                                time_internal::GetRepLo(d));
    }

    // Converts a std::chrono::duration to an turbo::Duration.
    template<typename Rep, typename Period>
    constexpr Duration Duration::from_chrono_duration(const std::chrono::duration<Rep, Period> &d) {
        static_assert(time_internal::IsValidRep64<Rep>(0), "duration::rep is invalid");
        return time_internal::from_int64(int64_t{d.count()}, Period{});
    }

    constexpr Duration Duration::from_chrono(const std::chrono::nanoseconds &d) {
        return from_chrono_duration(d);
    }

    constexpr Duration Duration::from_chrono(const std::chrono::microseconds &d) {
        return from_chrono_duration(d);
    }

    constexpr Duration Duration::from_chrono(
            const std::chrono::milliseconds &d) {
        return from_chrono_duration(d);
    }

    constexpr Duration Duration::from_chrono(
            const std::chrono::seconds &d) {
        return from_chrono_duration(d);
    }

    constexpr Duration Duration::from_chrono(
            const std::chrono::minutes &d) {
        return from_chrono_duration(d);
    }

    constexpr Duration Duration::from_chrono(
            const std::chrono::hours &d) {
        return from_chrono_duration(d);
    }

    constexpr Duration Duration::from_timespec(timespec ts) {
        if (static_cast<uint64_t>(ts.tv_nsec) < 1000 * 1000 * 1000) {
            int64_t ticks = ts.tv_nsec * time_internal::kTicksPerNanosecond;
            return time_internal::MakeDuration(ts.tv_sec, ticks);
        }
        return seconds(ts.tv_sec) + nanoseconds(ts.tv_nsec);
    }

    constexpr Duration Duration::from_timeval(timeval tv) {
        if (static_cast<uint64_t>(tv.tv_usec) < 1000 * 1000) {
            int64_t ticks = tv.tv_usec * 1000 * time_internal::kTicksPerNanosecond;
            return time_internal::MakeDuration(tv.tv_sec, ticks);
        }
        return seconds(tv.tv_sec) + microseconds(tv.tv_usec);
    }

    inline Duration Duration::fraction(Duration unit) const {
        return *this - trunc(unit);
    }

    struct ToMilliseconds {
        int64_t operator()(const turbo::Duration &d) const {
            return d.to_milliseconds();
        }
    };

    struct ToMicroseconds {
        int64_t operator()(const turbo::Duration &d) const {
            return d.to_microseconds();
        }
    };

    struct ToNanoseconds {
        int64_t operator()(const turbo::Duration &d) const {
            return d.to_nanoseconds();
        }
    };

    struct ToSeconds {
        int64_t operator()(const turbo::Duration &d) const {
            return d.to_seconds();
        }
    };

    struct ToDoubleSeconds {
        double operator()(const turbo::Duration &d) const {
            return d.to_seconds<double>();
        }
    };

    struct ToDoubleMilliseconds {
        double operator()(const turbo::Duration &d) const {
            return d.to_milliseconds<double>();
        }
    };

    struct ToDoubleMicroseconds {
        double operator()(const turbo::Duration &d) const {
            return d.to_microseconds<double>();
        }
    };

    struct ToDoubleNanoseconds {
        double operator()(const turbo::Duration &d) const {
            return d.to_nanoseconds<double>();
        }
    };

    struct ToDoubleMinutes {
        double operator()(const turbo::Duration &d) const {
            return d.to_minutes<double>();
        }
    };

    // turbo_parse_flag()
    //
    // Parses a command-line flag string representation `text` into a Duration
    // value. Duration flags must be specified in a format that is valid input for
    // `turbo::ParseDuration()`.
    bool turbo_parse_flag(std::string_view text, Duration *dst, std::string *error);


    // turbo_unparse_flag()
    //
    // Unparses a Duration value into a command-line string representation using
    // the format specified by `turbo::ParseDuration()`.
    std::string turbo_unparse_flag(Duration d);

}  // namespace turbo

#endif  // TURBO_TIMES_DURATION_H_
