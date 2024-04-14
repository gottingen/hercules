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
//

#ifndef TURBO_TIME_TIME_H_
#define TURBO_TIME_TIME_H_

#if !defined(_MSC_VER)

#include <sys/time.h>

#else
// We don't include `winsock2.h` because it drags in `windows.h` and friends,
// and they define conflicting macros like OPAQUE, ERROR, and more. This has the
// potential to break Turbo users.
//
// Instead we only forward declare `timeval` and require Windows users include
// `winsock2.h` themselves. This is both inconsistent and troublesome, but so is
// including 'windows.h' so we are picking the lesser of two evils here.
struct timeval;
#endif

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
#include "turbo/times/civil_time.h"
#include "turbo/times/cctz/time_zone.h"
#include "turbo/times/duration.h"
#include "turbo/base/assume.h"

namespace turbo {

    class Time;      // Defined below
    class TimeZone;  // Defined below
}  // namespace turbo
namespace turbo::time_internal {

    constexpr Time FromUnixDuration(Duration d);

    constexpr Duration ToUnixDuration(Time t);

    // Floors d to the next unit boundary closer to negative infinity.
    constexpr int64_t FloorToUnit(turbo::Duration d, turbo::Duration unit);

}  // namespace turbo::time_internal

namespace turbo {

    // CivilInfo
    //
    // Information about the civil time corresponding to an absolute time.
    // This struct is not intended to represent an instant in time. So, rather
    // than passing a `TimeZone::CivilInfo` to a function, pass an `turbo::Time`
    // and an `turbo::TimeZone`.
    struct CivilInfo {
        CivilInfo() = default;

        CivilSecond cs;
        Duration subsecond;

        constexpr civil_year_t year() const { return cs.year(); }

        constexpr int month() const { return cs.month(); }

        constexpr int mday() const { return cs.day(); }

        constexpr int hour() const { return cs.hour(); }

        constexpr int hour12() const {
            int h = cs.hour();
            if (h > 12) {
                h -= 12;
            }
            return h;
        }

        constexpr const char *ampm() const {
            return cs.hour() < 12 ? "am" : "pm";
        }

        constexpr int minute() const { return cs.minute(); }

        constexpr int second() const { return cs.second(); }

        inline Weekday week_day() const { return get_weekday(cs); }

        inline int wday() const {
            auto wd = get_weekday(cs);
            switch (wd) {
                case Weekday::sunday:
                    return 0;
                case Weekday::monday:
                    return 1;
                case Weekday::tuesday:
                    return 2;
                case Weekday::wednesday:
                    return 3;
                case Weekday::thursday:
                    return 4;
                case Weekday::friday:
                    return 5;
                case Weekday::saturday:
                    return 6;
            }
            TURBO_UNREACHABLE();
        }

        inline int yday() const { return get_year_day(cs); }

        // Note: The following fields exist for backward compatibility
        // with older APIs.  Accessing these fields directly is a sign of
        // imprudent logic in the calling code.  Modern time-related code
        // should only access this data indirectly by way of format_time().
        // These fields are undefined for infinite_future() and infinite_past().
        int offset{0};             // seconds east of UTC
        bool is_dst{0};            // is offset non-standard?
        const char *zone_abbr{nullptr};  // time-zone abbreviation (e.g., "PST")
    };


    /**
     * @brief The `turbo::Time` class represents a specific instant in time. Arithmetic operators
     *        are provided for naturally expressing time calculations. Instances are
     *        created using `turbo::time_now()` and the `turbo::From*()` factory functions that
     *        accept the gamut of other time representations. Formatting and parsing
     *        functions are provided for conversion to and from strings.  `turbo::Time`
     *        should be passed by value rather than const reference.
     *
     *        `turbo::Time` assumes there are 60 seconds in a minute, which means the
     *        underlying time scales must be "smeared" to eliminate leap seconds.
     *        See https://developers.google.com/time/smear.
     *
     *        Even though `turbo::Time` supports a wide range of timestamps, exercise
     *        caution when using values in the distant past. `turbo::Time` uses the
     *        Proleptic Gregorian calendar, which extends the Gregorian calendar backward
     *        to dates before its introduction in 1582.
     *        See https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar
     *        for more information. Use the ICU calendar classes to convert a date in
     *        some other calendar (http://userguide.icu-project.org/datetime/calendar).
     *
     *        Similarly, standardized time zones are a reasonably recent innovation, with
     *        the Greenwich prime meridian being established in 1884. The TZ database
     *        itself does not profess accurate offsets for timestamps prior to 1970. The
     *        breakdown of future timestamps is subject to the whim of regional
     *        governments.
     *
     *        The `turbo::Time` class represents an instant in time as a count of clock
     *        ticks of some granularity (resolution) from some starting point (epoch).
     *
     *        `turbo::Time` uses a resolution that is high enough to avoid loss in
     *        precision, and a range that is wide enough to avoid overflow, when
     *        converting between tick counts in most Google time scales (i.e., resolution
     *        of at least one nanosecond, and range +/-100 billion years).  Conversions
     *        between the time scales are performed by truncating (towards negative
     *        infinity) to the nearest representable point.
     *
     *        Examples:
     *        @code
     *        turbo::Time t1 = ...;
     *        turbo::Time t2 = t1 + turbo::Duration::minutes(2);
     *        turbo::Duration d = t2 - t1;  // == turbo::Duration::minutes(2)
     *        @endcode
     */
    class Time {
    public:
        // Value semantics.

        // Returns the Unix epoch.  However, those reading your code may not know
        // or expect the Unix epoch as the default value, so make your code more
        // readable by explicitly initializing all instances before use.
        //
        // Example:
        //   turbo::Time t = turbo::Time::unix_epoch();
        //   turbo::Time t = turbo::time_now();
        //   turbo::Time t = turbo::Time::from_timeval(tv);
        //   turbo::Time t = turbo::Time::infinite_past();
        constexpr Time() = default;

        // Copyable.
        constexpr Time(const Time &t) = default;

        Time &operator=(const Time &t) = default;

        // Assignment operators.
        Time &operator+=(Duration d) {
            rep_ += d;
            return *this;
        }

        Time &operator-=(Duration d) {
            rep_ -= d;
            return *this;
        }

        template<typename H>
        friend H hash_value(H h, Time t) {
            return H::combine(std::move(h), t.rep_);
        }

    public:
        //////////////////////////// gettter ////////////////////////////
        int64_t to_nanoseconds() const;

        int64_t to_microseconds() const;

        int64_t to_milliseconds() const;

        int64_t to_seconds() const;

        time_t to_time_t() const;

        double to_udate() const;

        int64_t to_universal() const;

        timespec to_timespec() const;

        timeval to_timeval() const;

        Duration fraction(Duration unit) const;

        struct tm to_tm(TimeZone tz) const;

        struct tm to_local_tm() const;

        struct tm to_utc_tm() const;

        CivilInfo to_civil(TimeZone tz) const;

        CivilInfo to_local_civil() const;

        CivilInfo to_utc_civil() const;

        std::chrono::system_clock::time_point to_chrono_time() const;


        /**
         * @ingroup turbo_times_time_zone
         * @brief Formats the given `turbo::Time` in the `turbo::TimeZone` according to the
         *        provided format string. Uses strftime()-like formatting options, with
         *        the following extensions:
         *        - %Ez  - RFC3339-compatible numeric UTC offset (+hh:mm or -hh:mm)
         *        - %E*z - Full-resolution numeric UTC offset (+hh:mm:ss or -hh:mm:ss)
         *        - %E#S - seconds with # digits of fractional precision
         *        - %E*S - seconds with full fractional precision (a literal '*')
         *        - %E#f - Fractional seconds with # digits of precision
         *        - %E*f - Fractional seconds with full precision (a literal '*')
         *        - %E4Y - Four-character years (-999 ... -001, 0000, 0001 ... 9999)
         *        - %ET  - The RFC3339 "date-time" separator "T"
         *        Note that %E0S behaves like %S, and %E0f produces no characters.  In
         *        contrast %E*f always produces at least one digit, which may be '0'.
         *        Note that %Y produces as many characters as it takes to fully render the
         *        year.  A year outside of [-999:9999] when formatted with %E4Y will produce
         *        more than four characters, just like %Y.
         *        We recommend that format strings include the UTC offset (%z, %Ez, or %E*z)
         *        so that the result uniquely identifies a time instant.
         *        Example:
         *        @code
         *        turbo::CivilSecond cs(2013, 1, 2, 3, 4, 5);
         *        turbo::Time t = turbo::Time::from_civil(cs, lax);
         *        std::string f = turbo::format_time("%H:%M:%S", t, lax);  // "03:04:05"
         *        f = turbo::format_time("%H:%M:%E3S", t, lax);  // "03:04:05.000"
         *        @endcode
         *        Note: If the given `turbo::Time` is `turbo::Time::infinite_future()`, the returned
         *        string will be exactly "infinite-future". If the given `turbo::Time` is
         *        `turbo::Time::infinite_past()`, the returned string will be exactly "infinite-past".
         *        In both cases the given format string and `turbo::TimeZone` are ignored.
         *        Example:
         *        @code
         *        turbo::Time t = turbo::Time::infinite_future();
         *        std::string f = turbo::format_time("%H:%M:%S", t, lax);  // "infinite-future"
         *        @endcode
         * @param format
         * @param t
         * @param tz
         * @return
         */
        std::string to_string(std::string_view format, TimeZone tz) const;

        std::string to_string(TimeZone tz) const;

        std::string to_string() const;

        /**
     * @ingroup turbo_times_time_zone
     * @brief Parses an input string according to the provided format string and
     *        returns the corresponding `turbo::Time`. Uses strftime()-like formatting
     *        options, with the same extensions as format_time(), but with the
     *        exceptions that %E#S is interpreted as %E*S, and %E#f as %E*f.  %Ez
     *        and %E*z also accept the same inputs, which (along with %z) includes
     *        'z' and 'Z' as synonyms for +00:00.  %ET accepts either 'T' or 't'.
     *        %Y consumes as many numeric characters as it can, so the matching data
     *        should always be terminated with a non-numeric.  %E4Y always consumes
     *        exactly four characters, including any sign.
     *        Unspecified fields are taken from the default date and time of ...
     *        "1970-01-01 00:00:00.0 +0000"
     *        For example, parsing a string of "15:45" (%H:%M) will return an turbo::Time
     *        that represents "1970-01-01 15:45:00.0 +0000".
     *        Note that since parse_time() returns time instants, it makes the most sense
     *        to parse fully-specified date/time strings that include a UTC offset (%z,
     *        %Ez, or %E*z).
     *        Note also that `turbo::parse_time()` only heeds the fields year, month, day,
     *        hour, minute, (fractional) second, and UTC offset.  Other fields, like
     *        weekday (%a or %A), while parsed for syntactic validity, are ignored
     *        in the conversion.
     *        Date and time fields that are out-of-range will be treated as errors
     *        rather than normalizing them like `turbo::CivilSecond` does.  For example,
     *        it is an error to parse the date "Oct 32, 2013" because 32 is out of range.
     *        A leap second of ":60" is normalized to ":00" of the following minute with
     *        fractional seconds discarded.  The following table shows how the given
     *        seconds and subseconds will be parsed:
     *        "59.x" -> 59.x  // exact
     *        "60.x" -> 00.0  // normalized
     *        "00.x" -> 00.x  // exact
     *        Errors are indicated by returning false and assigning an error message
     *        to the "err" out param if it is non-null.
     *        Note: If the input string is exactly "infinite-future", the returned
     *        `turbo::Time` will be `turbo::Time::infinite_future()` and `true` will be returned.
     *        If the input string is "infinite-past", the returned `turbo::Time` will be
     *        `turbo::Time::infinite_past()` and `true` will be returned.
     * @param format
     * @param input
     * @param time
     * @param err
     * @return
     */
        bool parse_time(std::string_view format, std::string_view input, std::string *err);

        /**
         * @ingroup turbo_times_time_zone
         * @brief Like `parse_time()` above, but if the format string does not contain a UTC
         *        offset specification (%z/%Ez/%E*z) then the input is interpreted in the
         *        given TimeZone.  This means that the input, by itself, does not identify a
         *        unique instant.  Being time-zone dependent, it also admits the possibility
         *        of ambiguity or non-existence, in which case the "pre" time (as defined
         *        by TimeZone::TimeInfo) is returned.  For these reasons we recommend that
         *        all date/time strings include a UTC offset so they're context independent.
         *        Example:
         *        @code
         *        turbo::Time t;
         *        std::string err;
         *        bool b = turbo::parse_time("%Y-%m-%d %H:%M:%S", "2013-10-19 12:34:56",
         *                                   lax, &t, &err);
         *        // b == true && err.empty() && t == 2013-10-19 12:34:56 -0700
         *        @endcode
         * @param format
         * @param input
         * @param tz
         * @param time
         * @param err
         * @return
         */
        bool parse_time(std::string_view format, std::string_view input, TimeZone tz, std::string *err);

    public:
        //////////////////////////// creator ////////////////////////////

        static turbo::Time time_now();

        static constexpr Time from_nanoseconds(int64_t ns);

        static constexpr Time from_microseconds(int64_t us);

        static constexpr Time from_milliseconds(int64_t ms);

        static constexpr Time from_seconds(int64_t s);

        static constexpr Time from_time_t(time_t t);

        static Time from_udate(double udate);

        static Time from_universal(int64_t universal);


        static Time from_timespec(timespec ts);

        static Time from_timeval(timeval tv);

        static Time from_chrono(const std::chrono::system_clock::time_point &tp);

        static Time from_civil(CivilSecond ct, TimeZone tz);

        static constexpr Time unix_epoch();

        static constexpr Time universal_epoch();

        static constexpr Time infinite_future();

        static constexpr Time infinite_past();

        static Time from_tm(const struct tm &tm, TimeZone tz);

        static Time from_date_time(int64_t year, int mon, int day, int hour,
                                   int min, int sec, TimeZone tz);

    private:
        friend constexpr Time time_internal::FromUnixDuration(Duration d);

        friend constexpr Duration time_internal::ToUnixDuration(Time t);

        friend constexpr bool operator<(Time lhs, Time rhs);

        friend constexpr bool operator==(Time lhs, Time rhs);

        friend Duration operator-(Time lhs, Time rhs);

        constexpr explicit Time(Duration rep) : rep_(rep) {}

        Duration rep_;
    };

    // Relational Operators
    constexpr bool operator<(Time lhs, Time rhs) {
        return lhs.rep_ < rhs.rep_;
    }

    constexpr bool operator>(Time lhs, Time rhs) {
        return rhs < lhs;
    }

    constexpr bool operator>=(Time lhs, Time rhs) {
        return !(lhs < rhs);
    }

    constexpr bool operator<=(Time lhs, Time rhs) {
        return !(rhs < lhs);
    }

    constexpr bool operator==(Time lhs, Time rhs) {
        return lhs.rep_ == rhs.rep_;
    }

    constexpr bool operator!=(Time lhs, Time rhs) {
        return !(lhs == rhs);
    }

    // Additive Operators
    inline Time operator+(Time lhs, Duration rhs) {
        return lhs += rhs;
    }

    inline Time operator+(Duration lhs, Time rhs) {
        return rhs += lhs;
    }

    inline Time operator-(Time lhs, Duration rhs) {
        return lhs -= rhs;
    }

    inline Duration operator-(Time lhs, Time rhs) {
        return lhs.rep_ - rhs.rep_;
    }

    constexpr Time Time::unix_epoch() { return Time(); }

    constexpr Time Time::universal_epoch() {
        // 719162 is the number of days from 0001-01-01 to 1970-01-01,
        // assuming the Gregorian calendar.
        return Time(time_internal::MakeDuration(-24 * 719162 * int64_t{3600}, uint32_t{0}));
    }

    constexpr Time Time::infinite_future() {
        return Time(time_internal::MakeDuration((std::numeric_limits<int64_t>::max)(),
                                                ~uint32_t{0}));
    }

    constexpr Time Time::infinite_past() {
        return Time(time_internal::MakeDuration((std::numeric_limits<int64_t>::min)(),
                                                ~uint32_t{0}));
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief The `turbo::TimeZone` is an opaque, small, value-type class representing a
     *        geo-political region within which particular rules are used for converting
     *        between absolute and civil times (see https://git.io/v59Ly). `turbo::TimeZone`
     *        values are named using the TZ identifiers from the IANA Time Zone Database,
     *        such as "America/Los_Angeles" or "Australia/Sydney". `turbo::TimeZone` values
     *        are created from factory functions such as `turbo::load_time_zone()`. Note:
     *        strings like "PST" and "EDT" are not valid TZ identifiers. Prefer to pass by
     *        value rather than const reference.
     *
     *        For more on the fundamental concepts of time zones, absolute times, and civil
     *        times, see https://github.com/google/cctz#fundamental-concepts
     *
     *        Examples:
     *        @code
     *        turbo::TimeZone utc = turbo::utc_time_zone();
     *        turbo::TimeZone pst = turbo::fixed_time_zone(-8 * 60 * 60);
     *        turbo::TimeZone loc = turbo::local_time_zone();
     *        turbo::TimeZone lax;
     *        if (!turbo::load_time_zone("America/Los_Angeles", &lax)) {
     *              // handle error case
     *        }
     *        @endcode
     *        See also:
     *        - https://github.com/google/cctz
     *        - https://www.iana.org/time-zones
     *        - https://en.wikipedia.org/wiki/Zoneinfo
     */
    class TimeZone {
    public:
        explicit TimeZone(time_internal::cctz::time_zone tz) : cz_(tz) {}

        TimeZone() = default;  // UTC, but prefer utc_time_zone() to be explicit.

        // Copyable.
        TimeZone(const TimeZone &) = default;

        TimeZone &operator=(const TimeZone &) = default;

        explicit operator time_internal::cctz::time_zone() const { return cz_; }

        std::string name() const { return cz_.name(); }

        /**
         * @brief Returns the civil time for this TimeZone at a certain `turbo::Time`.
         *        If the input time is infinite, the output civil second will be set to
         *        CivilSecond::max() or min(), and the subsecond will be infinite.
         *
         *        Example:
         *        @code
         *        const auto epoch = lax.At(turbo::Time::unix_epoch());
         *        // epoch.cs == 1969-12-31 16:00:00
         *        // epoch.subsecond == turbo::Duration::zero()
         *        // epoch.offset == -28800
         *        // epoch.is_dst == false
         *        // epoch.abbr == "PST"
         *        @endcode
         * @param t the time to convert
         * @return the converted time
         */
        CivilInfo at(Time t) const;

        /**
         * @brief Information about the absolute times corresponding to a civil time.
         *        (Subseconds must be handled separately.)
         *
         *        It is possible for a caller to pass a civil-time value that does
         *        not represent an actual or unique instant in time (due to a shift
         *        in UTC offset in the TimeZone, which results in a discontinuity in
         *        the civil-time components). For example, a daylight-saving-time
         *        transition skips or repeats civil times---in the United States,
         *        March 13, 2011 02:15 never occurred, while November 6, 2011 01:15
         *        occurred twice---so requests for such times are not well-defined.
         *        To account for these possibilities, `turbo::TimeZone::TimeInfo` is
         *        richer than just a single `turbo::Time`.
         */
        struct TimeInfo {
            enum CivilKind {
                UNIQUE,    // the civil time was singular (pre == trans == post)
                SKIPPED,   // the civil time did not exist (pre >= trans > post)
                REPEATED,  // the civil time was ambiguous (pre < trans <= post)
            } kind;
            Time pre;    // time calculated using the pre-transition offset
            Time trans;  // when the civil-time discontinuity occurred
            Time post;   // time calculated using the post-transition offset
        };

        /**
         * @brief Returns the absolute time(s) for this TimeZone at a certain `turbo::CivilSecond`.
         *        When the civil time is skipped or repeated, returns times calculated using
         *        the pre-transition and post-transition UTC offsets, plus the transition time
         *        itself.
         *
         *        Examples:
         *        @code
         *        const auto jan01 = lax.At(turbo::CivilSecond(2011, 1, 1, 0, 0, 0));
         *        // jan01.kind == TimeZone::TimeInfo::UNIQUE
         *        // jan01.pre    is 2011-01-01 00:00:00 -0800
         *        // jan01.trans  is 2011-01-01 00:00:00 -0800
         *        // jan01.post   is 2011-01-01 00:00:00 -0800
         *
         *        // A Spring DST transition, when there is a gap in civil time
         *        const auto mar13 = lax.At(turbo::CivilSecond(2011, 3, 13, 2, 15, 0));
         *        // mar13.kind == TimeZone::TimeInfo::SKIPPED
         *        // mar13.pre   is 2011-03-13 03:15:00 -0700
         *        // mar13.trans is 2011-03-13 03:00:00 -0700
         *        // mar13.post  is 2011-03-13 01:15:00 -0800
         *
         *        // A Fall DST transition, when civil times are repeated
         *        const auto nov06 = lax.At(turbo::CivilSecond(2011, 11, 6, 1, 15, 0));
         *        // nov06.kind == TimeZone::TimeInfo::REPEATED
         *        // nov06.pre   is 2011-11-06 01:15:00 -0700
         *        // nov06.trans is 2011-11-06 01:00:00 -0800
         *        // nov06.post  is 2011-11-06 01:15:00 -0800
         *        @endcode
         * @param cs the civil time to convert
         * @return the converted time
         */
        TimeInfo at(CivilSecond ct) const;

        /**
         * @brief Finds the time of the next offset change in this time zone.
         *       By definition, `NextTransition(t, &trans)` returns false when `t` is
         *       `infinite_future()`. If the zone has no transitions, the result will
         *       also be false no matter what the argument.
         *
         *       Otherwise, when `t` is `infinite_past()`, `NextTransition(t, &trans)`
         *       returns true and sets `trans` to the first recorded transition. Chains
         *       of calls to `NextTransition()/PrevTransition()` will eventually return
         *       false, but it is unspecified exactly when `NextTransition(t, &trans)`
         *       jumps to false, or what time is set by `PrevTransition(t, &trans)` for
         *       a very distant `t`.
         *
         * @note Enumeration of time-zone transitions is for informational purposes only.
         *      Modern time-related code should not care about when offset changes occur.
         *      Example:
         *      @code
         *      turbo::TimeZone nyc;
         *      if (!turbo::load_time_zone("America/New_York", &nyc)) { ... }
         *      const auto now = turbo::time_now();
         *      auto t = turbo::Time::infinite_past();
         *      turbo::TimeZone::CivilTransition trans;
         *      while (t <= now && nyc.NextTransition(t, &trans)) {
         *          // transition: trans.from -> trans.to
         *          t = nyc.At(trans.to).trans;
         *      }
         *     @endcode
         */
        struct CivilTransition {
            CivilSecond from;  // the civil time we jump from
            CivilSecond to;    // the civil time we jump to
        };

        bool NextTransition(Time t, CivilTransition *trans) const;

        bool PrevTransition(Time t, CivilTransition *trans) const;

        template<typename H>
        friend H hash_value(H h, TimeZone tz) {
            return H::combine(std::move(h), tz.cz_);
        }

    private:
        friend bool operator==(TimeZone a, TimeZone b) { return a.cz_ == b.cz_; }

        friend bool operator!=(TimeZone a, TimeZone b) { return a.cz_ != b.cz_; }

        friend std::ostream &operator<<(std::ostream &os, TimeZone tz) {
            return os << tz.name();
        }

        time_internal::cctz::time_zone cz_;
    };

    /**
     * @ingroup turbo_times_time_zone
     * @brief Loads the named zone. May perform I/O on the initial load of the named
     *        zone. If the name is invalid, or some other kind of error occurs, returns
     *        `false` and `*tz` is set to the UTC time zone.
     * @param name
     * @param tz
     * @return
     */
    inline bool load_time_zone(std::string_view name, TimeZone *tz) {
        if (name == "localtime") {
            *tz = TimeZone(time_internal::cctz::local_time_zone());
            return true;
        }
        time_internal::cctz::time_zone cz;
        const bool b = time_internal::cctz::load_time_zone(std::string(name), &cz);
        *tz = TimeZone(cz);
        return b;
    }


    /**`
     * @ingroup turbo_times_time_zone
     * @brief Returns a TimeZone that is a fixed offset (seconds east) from UTC.
     *        Note: If the absolute value of the offset is greater than 24 hours
     *        you'll get UTC (i.e., no offset) instead.
     * @param seconds
     * @return
     */
    inline TimeZone fixed_time_zone(int seconds) {
        return TimeZone(
                time_internal::cctz::fixed_time_zone(std::chrono::seconds(seconds)));
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief Convenience method returning the UTC time zone.
     * @return
     */
    inline TimeZone utc_time_zone() {
        return TimeZone(time_internal::cctz::utc_time_zone());
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief Convenience method returning the local time zone, or UTC if there is
     *        no configured local zone.  Warning: Be wary of using local_time_zone(),
     *        and particularly so in a server process, as the zone configured for the
     *        local machine should be irrelevant.  Prefer an explicit zone name.
     * @return
     */
    inline TimeZone local_time_zone() {
        return TimeZone(time_internal::cctz::local_time_zone());
    }

    inline CivilInfo to_civil_info(Time t,
                                   TimeZone tz) {
        return tz.at(t);  // already a CivilSecond
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief Helpers for TimeZone::At(Time) to return particularly aligned civil times.
     *       Example:
     *       @code
     *       turbo::Time t = ...;
     *       turbo::TimeZone tz = ...;
     *       const auto cd = turbo::to_civil_day(t, tz);
     *       @endcode
     * @param t the time to convert
     * @param tz the time zone to use
     * @return the converted time
     */
    inline CivilSecond to_civil_second(Time t,
                                       TimeZone tz) {
        return tz.at(t).cs;  // already a CivilSecond
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief  similar to `to_civil_second()`
     * @see `to_civil_second()`
     * @param t
     * @param tz
     * @return
     */
    inline CivilMinute to_civil_minute(Time t,
                                       TimeZone tz) {
        return CivilMinute(tz.at(t).cs);
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief  similar to `to_civil_second()`
     * @see `to_civil_second()`
     * @param t
     * @param tz
     * @return
     */
    inline CivilHour to_civil_hour(Time t, TimeZone tz) {
        return CivilHour(tz.at(t).cs);
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief  similar to `to_civil_second()`
     * @see `to_civil_second()`
     * @param t
     * @param tz
     * @return
     */
    inline CivilDay to_civil_day(Time t, TimeZone tz) {
        return CivilDay(tz.at(t).cs);
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief  similar to `to_civil_second()`
     * @see `to_civil_second()`
     * @param t
     * @param tz
     * @return
     */
    inline CivilMonth to_civil_month(Time t,
                                     TimeZone tz) {
        return CivilMonth(tz.at(t).cs);
    }

    /**
     * @ingroup turbo_times_time_zone
     * @brief  similar to `to_civil_second()`
     * @see `to_civil_second()`
     * @param t
     * @param tz
     * @return
     */
    inline CivilYear to_civil_year(Time t, TimeZone tz) {
        return CivilYear(tz.at(t).cs);
    }

    // TimeConversion
    //
    // An `turbo::TimeConversion` represents the conversion of year, month, day,
    // hour, minute, and second values (i.e., a civil time), in a particular
    // `turbo::TimeZone`, to a time instant (an absolute time), as returned by
    // `turbo::convert_date_time()`. Legacy version of `turbo::TimeZone::TimeInfo`.
    //
    // Deprecated. Use `turbo::TimeZone::TimeInfo`.
    struct
    TimeConversion {
        Time pre;    // time calculated using the pre-transition offset
        Time trans;  // when the civil-time discontinuity occurred
        Time post;   // time calculated using the post-transition offset

        enum Kind {
            UNIQUE,    // the civil time was singular (pre == trans == post)
            SKIPPED,   // the civil time did not exist
            REPEATED,  // the civil time was ambiguous
        };
        Kind kind;

        bool normalized;  // input values were outside their valid ranges
    };

    // convert_date_time()
    //
    // Legacy version of `turbo::TimeZone::At(turbo::CivilSecond)` that takes
    // the civil time as six, separate values (YMDHMS).
    //
    // The input month, day, hour, minute, and second values can be outside
    // of their valid ranges, in which case they will be "normalized" during
    // the conversion.
    //
    // Example:
    //
    //   // "October 32" normalizes to "November 1".
    //   turbo::TimeConversion tc =
    //       turbo::convert_date_time(2013, 10, 32, 8, 30, 0, lax);
    //   // tc.kind == TimeConversion::UNIQUE && tc.normalized == true
    //   // turbo::to_civil_day(tc.pre, tz).month() == 11
    //   // turbo::to_civil_day(tc.pre, tz).day() == 1
    //
    // Deprecated. Use `turbo::TimeZone::At(CivilSecond)`.
    TimeConversion convert_date_time(int64_t year, int mon, int day, int hour,
                                     int min, int sec, TimeZone tz);

    // from_date_time()
    //
    // A convenience wrapper for `turbo::convert_date_time()` that simply returns
    // the "pre" `turbo::Time`.  That is, the unique result, or the instant that
    // is correct using the pre-transition offset (as if the transition never
    // happened).
    //
    // Example:
    //
    //   turbo::Time t = turbo::from_date_time(2017, 9, 26, 9, 30, 0, lax);
    //   // t = 2017-09-26 09:30:00 -0700
    //
    // Deprecated. Use `turbo::Time::from_civil(CivilSecond, TimeZone)`. Note that the
    // behavior of `from_civil()` differs from `from_date_time()` for skipped civil
    // times. If you care about that see `turbo::TimeZone::At(turbo::CivilSecond)`.
    /**
     * @ingroup turbo_times_time_zone
     * @brief  similar to `convert_date_time()` but returns the "pre" `turbo::Time`
     *         (the unique result, or the instant that is correct using the pre-transition
     *         offset (as if the transition never happened)).
     *         Deprecated. Use `turbo::Time::from_civil(CivilSecond, TimeZone)`. Note that the
     *         behavior of `from_civil()` differs from `from_date_time()` for skipped civil
     *         times. If you care about that see `turbo::TimeZone::At(turbo::CivilSecond)`.
     *         Example:
     *         @code
     *         turbo::Time t = turbo::from_date_time(2017, 9, 26, 9, 30, 0, lax);
     *         // t = 2017-09-26 09:30:00 -0700
     *         @endcode
     *
     * @see `convert_date_time()`
     * @param year
     * @param mon
     * @param day
     * @param hour
     * @param min
     * @param sec
     * @param tz
     * @return
     */
    inline Time Time::from_date_time(int64_t year, int mon, int day, int hour,
                                     int min, int sec, TimeZone tz) {
        return convert_date_time(year, mon, day, hour, min, sec, tz).pre;
    }

    // RFC3339_full
    // RFC3339_sec
    //
    // format_time()/parse_time() format specifiers for RFC3339 date/time strings,
    // with trailing zeros trimmed or with fractional seconds omitted altogether.
    //
    // Note that RFC3339_sec[] matches an ISO 8601 extended format for date and
    // time with UTC offset.  Also note the use of "%Y": RFC3339 mandates that
    // years have exactly four digits, but we allow them to take their natural
    // width.
    TURBO_DLL extern const char RFC3339_full[];  // %Y-%m-%d%ET%H:%M:%E*S%Ez
    TURBO_DLL extern const char RFC3339_sec[];   // %Y-%m-%d%ET%H:%M:%S%Ez

    // RFC1123_full
    // RFC1123_no_wday
    //
    // format_time()/parse_time() format specifiers for RFC1123 date/time strings.
    TURBO_DLL extern const char RFC1123_full[];     // %a, %d %b %E4Y %H:%M:%S %z
    TURBO_DLL extern const char RFC1123_no_wday[];  // %d %b %E4Y %H:%M:%S %z



    // Convenience functions that format the given time using the RFC3339_full
    // format.  The first overload uses the provided TimeZone, while the second
    // uses local_time_zone().


    // Output stream operator.
    inline std::ostream &operator<<(std::ostream &os, Time t) {
        return os << t.to_string();
    }

    // ============================================================================
    // Implementation Details Follow
    // ============================================================================

    namespace time_internal {

        // Map between a Time and a Duration since the Unix epoch.  Note that these
        // functions depend on the above mentioned choice of the Unix epoch for the
        // Time representation (and both need to be Time friends).  Without this
        // knowledge, we would need to add-in/subtract-out unix_epoch() respectively.
        constexpr Time FromUnixDuration(Duration d) {
            return Time(d);
        }

        constexpr Duration ToUnixDuration(Time t) {
            return t.rep_;
        }


    }  // namespace time_internal

    constexpr Time Time::from_nanoseconds(int64_t ns) {
        return time_internal::FromUnixDuration(Duration::nanoseconds(ns));
    }

    constexpr Time Time::from_microseconds(int64_t us) {
        return time_internal::FromUnixDuration(Duration::microseconds(us));
    }

    constexpr Time Time::from_milliseconds(int64_t ms) {
        return time_internal::FromUnixDuration(Duration::milliseconds(ms));
    }

    constexpr Time Time::from_seconds(int64_t s) {
        return time_internal::FromUnixDuration(Duration::seconds(s));
    }

    constexpr Time Time::from_time_t(time_t t) {
        return time_internal::FromUnixDuration(Duration::seconds(t));
    }

    inline Time Time::from_civil(CivilSecond ct,
                                 TimeZone tz) {
        const auto ti = tz.at(ct);
        if (ti.kind == TimeZone::TimeInfo::SKIPPED) return ti.trans;
        return ti.pre;
    }

    inline Duration Time::fraction(Duration unit) const {
        return rep_.fraction(unit);
    }

    // turbo_parse_flag()
    //
    // Parses the command-line flag string representation `text` into a Time value.
    // Time flags must be specified in a format that matches turbo::RFC3339_full.
    //
    // For example:
    //
    //   --start_time=2016-01-02T03:04:05.678+08:00
    //
    // Note: A UTC offset (or 'Z' indicating a zero-offset from UTC) is required.
    //
    // Additionally, if you'd like to specify a time as a count of
    // seconds/milliseconds/etc from the Unix epoch, use an turbo::Duration flag
    // and add that duration to turbo::UnixEpoch() to get an turbo::Time.
        bool turbo_parse_flag(std::string_view text, Time* t, std::string* error);

    // turbo_unparse_flag()
    //
    // Unparses a Time value into a command-line string representation using
    // the format specified by `turbo::ParseTime()`.
        std::string turbo_unparse_flag(Time t);
}  // namespace turbo

#endif  // TURBO_TIME_TIME_H_
