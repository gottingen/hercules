// Copyright 2018 The Turbo Authors.
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

#ifndef TURBO_TIME_CIVIL_TIME_H_
#define TURBO_TIME_CIVIL_TIME_H_

#include <iosfwd>
#include <string>

#include "turbo/platform/port.h"
#include "turbo/strings/string_view.h"
#include "turbo/times/cctz/civil_time.h"

namespace turbo {

    namespace time_internal {
        struct second_tag : cctz::detail::second_tag {
        };
        struct minute_tag : second_tag, cctz::detail::minute_tag {
        };
        struct hour_tag : minute_tag, cctz::detail::hour_tag {
        };
        struct day_tag : hour_tag, cctz::detail::day_tag {
        };
        struct month_tag : day_tag, cctz::detail::month_tag {
        };
        struct year_tag : month_tag, cctz::detail::year_tag {
        };
    }  // namespace time_internal

    /**
     * @ingroup turbo_times_civil_time
     * @brief Each of these civil-time types is a simple value type with the same
     *       interface for construction and the same six accessors for each of the civil
     *       time fields (year, month, day, hour, minute, and second, aka YMDHMS). These
     *       classes differ only in their alignment, which is indicated by the type name
     *       and specifies the field on which arithmetic operates.
     *       CONSTRUCTION
     *       Each of the civil-time types can be constructed in two ways: by directly
     *       passing to the constructor up to six integers representing the YMDHMS fields,
     *       or by copying the YMDHMS fields from a differently aligned civil-time type.
     *       Omitted fields are assigned their minimum valid value. hours, minutes, and
     *       seconds will be set to 0, month and day will be set to 1. Since there is no
     *       minimum year, the default is 1970.
     *       Examples:
     *       @code
     *       turbo::CivilDay default_value;               // 1970-01-01 00:00:00
     *       turbo::CivilDay a(2015, 2, 3);               // 2015-02-03 00:00:00
     *       turbo::CivilDay b(2015, 2, 3, 4, 5, 6);      // 2015-02-03 00:00:00
     *       turbo::CivilDay c(2015);                     // 2015-01-01 00:00:00
     *       turbo::CivilSecond ss(2015, 2, 3, 4, 5, 6);  // 2015-02-03 04:05:06
     *       turbo::CivilMinute mm(ss);                   // 2015-02-03 04:05:00
     *       turbo::CivilHour hh(mm);                     // 2015-02-03 04:00:00
     *       turbo::CivilDay d(hh);                       // 2015-02-03 00:00:00
     *       turbo::CivilMonth m(d);                      // 2015-02-01 00:00:00
     *       turbo::CivilYear y(m);                       // 2015-01-01 00:00:00
     *       m = turbo::CivilMonth(y);                    // 2015-01-01 00:00:00
     *       d = turbo::CivilDay(m);                      // 2015-01-01 00:00:00
     *       hh = turbo::CivilHour(d);                    // 2015-01-01 00:00:00
     *       mm = turbo::CivilMinute(hh);                 // 2015-01-01 00:00:00
     *       ss = turbo::CivilSecond(mm);                 // 2015-01-01 00:00:00
     *       @endcode
     *       Each civil-time class is aligned to the civil-time field indicated in the
     *       class's name after normalization. Alignment is performed by setting all the
     *       inferior fields to their minimum valid value (as described above). The
     *       following are examples of how each of the six types would align the fields
     *       representing November 22, 2015 at 12:34:56 in the afternoon. (Note: the
     *       string format used here is not important; it's just a shorthand way of
     *       showing the six YMDHMS fields.)
     *       @code
     *       turbo::CivilSecond   : 2015-11-22 12:34:56
     *       turbo::CivilMinute   : 2015-11-22 12:34:00
     *       turbo::CivilHour     : 2015-11-22 12:00:00
     *       turbo::CivilDay      : 2015-11-22 00:00:00
     *       turbo::CivilMonth    : 2015-11-01 00:00:00
     *       turbo::CivilYear     : 2015-01-01 00:00:00
     *       @endcode
     *
     *       Each civil-time type performs arithmetic on the field to which it is
     *       aligned. This means that adding 1 to an turbo::CivilDay increments the day
     *       field (normalizing as necessary), and subtracting 7 from an turbo::CivilMonth
     *       operates on the month field (normalizing as necessary). All arithmetic
     *       produces a valid civil time. Difference requires two similarly aligned
     *       civil-time objects and returns the scalar answer in units of the objects'
     *       alignment. For example, the difference between two turbo::CivilHour objects
     *       will give an answer in units of civil hours.
     *       ALIGNMENT CONVERSION
     *       The alignment of a civil-time object cannot change, but the object may be
     *       used to construct a new object with a different alignment. This is referred
     *       to as "realigning". When realigning to a type with the same or more
     *       precision (e.g., turbo::CivilDay -> turbo::CivilSecond), the conversion may be
     *       performed implicitly since no information is lost. However, if information
     *       could be discarded (e.g., CivilSecond -> CivilDay), the conversion must
     *       be explicit at the call site.
     *       Examples:
     *       @code
     *       void UseDay(turbo::CivilDay day);
     *       turbo::CivilSecond cs;
     *       UseDay(cs);                  // Won't compile because data may be discarded
     *       UseDay(turbo::CivilDay(cs));  // OK: explicit conversion
     *       turbo::CivilDay cd;
     *       UseDay(cd);                  // OK: no conversion needed
     *       turbo::CivilMonth cm;
     *       UseDay(cm);                  // OK: implicit conversion to turbo::CivilDay
     *       @endcode
     *       NORMALIZATION
     *       Normalization takes invalid values and adjusts them to produce valid values.
     *       Within the civil-time library, integer arguments passed to the Civil*
     *       constructors may be out-of-range, in which case they are normalized by
     *       carrying overflow into a field of courser granularity to produce valid
     *       civil-time objects. This normalization enables natural arithmetic on
     *       constructor arguments without worrying about the field's range.
     *       Examples:
     *       @code
     *       // Out-of-range; normalized to 2016-11-01
     *       turbo::CivilDay d(2016, 10, 32);
     *       // Out-of-range, negative: normalized to 2016-10-30T23
     *       turbo::CivilHour h1(2016, 10, 31, -1);
     *       // Normalization is cumulative: normalized to 2016-10-30T23
     *       turbo::CivilHour h2(2016, 10, 32, -25);
     *       @endcode
     *       Note: If normalization is undesired, you can signal an error by comparing
     *       the constructor arguments to the normalized values returned by the YMDHMS
     *       properties.
     *       COMPARISON
     *       Comparison between civil-time objects considers all six YMDHMS fields,
     *       regardless of the type's alignment. Comparison between differently aligned
     *       civil-time types is allowed.
     *       Examples:
     *       @code
     *       turbo::CivilDay feb_3(2015, 2, 3);  // 2015-02-03 00:00:00
     *       turbo::CivilDay mar_4(2015, 3, 4);  // 2015-03-04 00:00:00
     *       // feb_3 < mar_4
     *       // turbo::CivilYear(feb_3) == turbo::CivilYear(mar_4)
     *       turbo::CivilSecond feb_3_noon(2015, 2, 3, 12, 0, 0);  // 2015-02-03 12:00:00
     *       // feb_3 < feb_3_noon
     *       // feb_3 == turbo::CivilDay(feb_3_noon)
     *       // Iterates all the days of February 2015.
     *       for (turbo::CivilDay d(2015, 2, 1); d < turbo::CivilMonth(2015, 3); ++d) {
     *       // ...
     *       }
     *       @endcode
     *       ARITHMETIC
     *       Civil-time types support natural arithmetic operators such as addition,
     *       subtraction, and difference. Arithmetic operates on the civil-time field
     *       indicated in the type's name. Difference operators require arguments with
     *       the same alignment and return the answer in units of the alignment.
     *       Example:
     *       @code
     *       turbo::CivilDay a(2015, 2, 3);
     *       ++a;                              // 2015-02-04 00:00:00
     *       --a;                              // 2015-02-03 00:00:00
     *       turbo::CivilDay b = a + 1;         // 2015-02-04 00:00:00
     *       turbo::CivilDay c = 1 + b;         // 2015-02-05 00:00:00
     *       int n = c - a;                    // n = 2 (civil days)
     *       int m = c - turbo::CivilMonth(c);  // Won't compile: different types.
     *       @endcode
     *       ACCESSORS
     *       Each civil-time type has accessors for all six of the civil-time fields:
     *       year, month, day, hour, minute, and second.
     *       civil_year_t year()
     *       int          month()
     *       int          day()
     *       int          hour()
     *       int          minute()
     *       int          second()
     *       Recall that fields inferior to the type's alignment will be set to their
     *       minimum valid value.
     *       Example:
     *       @code
     *       turbo::CivilDay d(2015, 6, 28);
     *       // d.year() == 2015
     *       // d.month() == 6
     *       // d.day() == 28
     *       // d.hour() == 0
     *       // d.minute() == 0
     *       // d.second() == 0
     *       @endcode
     *       CASE STUDY: Adding a month to January 31.
     *       One of the classic questions that arises when considering a civil time
     *       library (or a date library or a date/time library) is this:
     *       "What is the result of adding a month to January 31?"
     *       This is an interesting question because it is unclear what is meant by a
     *       "month", and several different answers are possible, depending on context:
     *       1. March 3 (or 2 if a leap year), if "add a month" means to add a month to
     *       the current month, and adjust the date to overflow the extra days into
     *       March. In this case the result of "February 31" would be normalized as
     *       within the civil-time library.
     *       2. February 28 (or 29 if a leap year), if "add a month" means to add a
     *       month, and adjust the date while holding the resulting month constant.
     *       In this case, the result of "February 31" would be truncated to the last
     *       day in February.
     *       3. An error. The caller may get some error, an exception, an invalid date
     *       object, or perhaps return `false`. This may make sense because there is
     *       no single unambiguously correct answer to the question.
     *       Practically speaking, any answer that is not what the programmer intended
     *       is the wrong answer.
     *       The Turbo time library avoids this problem by making it impossible to
     *       ask ambiguous questions. All civil-time objects are aligned to a particular
     *       civil-field boundary (such as aligned to a year, month, day, hour, minute,
     *       or second), and arithmetic operates on the field to which the object is
     *       aligned. This means that in order to "add a month" the object must first be
     *       aligned to a month boundary, which is equivalent to the first day of that
     *       month.
     *       Of course, there are ways to compute an answer the question at hand using
     *       this Turbo time library, but they require the programmer to be explicit
     *       about the answer they expect. To illustrate, let's see how to compute all
     *       three of the above possible answers to the question of "Jan 31 plus 1
     *       month":
     *       Example:
     *       @code
     *       const turbo::CivilDay d(2015, 1, 31);
     *       // Answer 1:
     *       // Add 1 to the month field in the constructor, and rely on normalization.
     *       const auto normalized = turbo::CivilDay(d.year(), d.month() + 1, d.day());
     *       // normalized == 2015-03-03 (aka Feb 31)
     *       // Answer 2:
     *       // Add 1 to month field, capping to the end of next month.
     *       const auto next_month = turbo::CivilMonth(d) + 1;
     *       const auto last_day_of_next_month = turbo::CivilDay(next_month + 1) - 1;
     *       const auto capped = std::min(normalized, last_day_of_next_month);
     *       // capped == 2015-02-28
     *       // Answer 3:
     *       // Signal an error if the normalized answer is not in next month.
     *       if (turbo::CivilMonth(normalized) != next_month) {
     *       // error, month overflow
     *       }
     *       @endcode
     *       @see turbo::CivilSecond
     *       @see turbo::CivilMinute
     *       @see turbo::CivilHour
     *       @see turbo::CivilDay
     *       @see turbo::CivilMonth
     *       @see turbo::CivilYear
     *       @see turbo::civil_year_t
     *       @see turbo::civil_diff_t
     *       @see turbo::Weekday
     *       @see turbo::get_weekday
     *       @see turbo::next_weekday
     *       @see turbo::prev_weekday
     *       @see turbo::get_year_day
     *       @see turbo::GetMonthDay
     *       @see turbo::GetYearWeek
     *       @see turbo::GetMonthWeek
     *       @see turbo::GetYearWeekDay
     *       @see turbo::GetMonthWeekDay
     *       @see turbo::GetYearWeekDayOrdinal
     *       @see turbo::GetMonthWeekDayOrdinal
     *       @see turbo::GetYearWeekOrdinal
     *       @see turbo::GetMonthWeekOrdinal
     *       @see turbo::GetYearWeekOrdinalSunday
     *       @see turbo::GetMonthWeekOrdinalSunday
     *       @see turbo::GetYearWeekSunday
     *       @see turbo::GetMonthWeekSunday
     *       @see turbo::GetYearWeekSundayOrdinal
     *       @see turbo::GetMonthWeekSundayOrdinal
     *       @see turbo::GetYearWeekMonday
     *       @see turbo::GetMonthWeekMonday
     *       @see turbo::GetYearWeekMondayOrdinal
     *       @see turbo::GetMonthWeekMondayOrdinal
     *
     */
    using CivilSecond =
            time_internal::cctz::detail::civil_time<time_internal::second_tag>;
    using CivilMinute =
            time_internal::cctz::detail::civil_time<time_internal::minute_tag>;
    using CivilHour =
            time_internal::cctz::detail::civil_time<time_internal::hour_tag>;
    using CivilDay =
            time_internal::cctz::detail::civil_time<time_internal::day_tag>;
    using CivilMonth =
            time_internal::cctz::detail::civil_time<time_internal::month_tag>;
    using CivilYear =
            time_internal::cctz::detail::civil_time<time_internal::year_tag>;

    /**
     * @ingroup turbo_times_civil_time
     * @brief Type alias of a civil-time year value. This type is guaranteed to (at least)
     *       support any year value supported by `time_t`.
     *       Example:
     *       @code
     *       turbo::CivilSecond cs = ...;
     *       turbo::civil_year_t y = cs.year();
     *       cs = turbo::CivilSecond(y, 1, 1, 0, 0, 0);  // CivilSecond(CivilYear(cs))
     *       @endcode
     *
     */
    using civil_year_t = time_internal::cctz::year_t;

    /**
     * @ingroup turbo_times_civil_time
     * @brief Type alias of the difference between two civil-time values.
     *       This type is used to indicate arguments that are not
     *       normalized (such as parameters to the civil-time constructors), the results
     *       of civil-time subtraction, or the operand to civil-time addition.
     *       Example:
     *       @code
     *       turbo::civil_diff_t n_sec = cs1 - cs2;             // cs1 == cs2 + n_sec;
     *       @endcode
     *
     */
    using civil_diff_t = time_internal::cctz::diff_t;

    /**
     * @ingroup turbo_times_civil_time
     * @brief The Weekday enum class represents the civil-time concept of a "weekday" with
     *       members for all days of the week.
     *       Example:
     *       @code
     *       turbo::Weekday wd = turbo::Weekday::thursday;
     *       @endcode
     *
     */
    using Weekday = time_internal::cctz::weekday;

    /**
     * @ingroup turbo_times_civil_time
     * @brief Returns the turbo::Weekday for the given (realigned) civil-time value.
     *       Example:
     *       @code
     *       turbo::CivilDay a(2015, 8, 13);
     *       turbo::Weekday wd = turbo::get_weekday(a);  // wd == turbo::Weekday::thursday
     *       @endcode
     *
     */
    inline Weekday get_weekday(CivilSecond cs) {
        return time_internal::cctz::get_weekday(cs);
    }

    /**
     * @ingroup turbo_times_civil_time
     * @brief Returns the turbo::CivilDay that strictly follows a given turbo::CivilDay, and
     *       that falls on the given turbo::Weekday.
     *       Example, given the following month:
     *       @code
     *       August 2015
     *       Su Mo Tu We Th Fr Sa
     *       1
     *       2  3  4  5  6  7  8
     *       9 10 11 12 13 14 15
     *       16 17 18 19 20 21 22
     *       23 24 25 26 27 28 29
     *       30 31
     *       turbo::CivilDay a(2015, 8, 13);
     *       // turbo::get_weekday(a) == turbo::Weekday::thursday
     *       turbo::CivilDay b = turbo::next_weekday(a, turbo::Weekday::thursday);
     *       // b = 2015-08-20
     *
     *       turbo::CivilDay d = ...
     *       // Gets the following Thursday if d is not already Thursday
     *       turbo::CivilDay thurs1 = turbo::next_weekday(d - 1, turbo::Weekday::thursday);
     *       // Gets the previous Thursday if d is not already Thursday
     *       turbo::CivilDay thurs2 = turbo::prev_weekday(d + 1, turbo::Weekday::thursday);
     *       @endcode
     *
     */
    inline CivilDay next_weekday(CivilDay cd, Weekday wd) {
        return CivilDay(time_internal::cctz::next_weekday(cd, wd));
    }

    /**
     * @ingroup turbo_times_civil_time
     * @brief similar to next_weekday, but returns the previous weekday
     * @see turbo::next_weekday
     */
    inline CivilDay prev_weekday(CivilDay cd, Weekday wd) {
        return CivilDay(time_internal::cctz::prev_weekday(cd, wd));
    }

    /**
     * @ingroup turbo_times_civil_time
     * @brief Returns the day-of-year for the given (realigned) civil-time value.
     *       Example:
     *       @code
     *       turbo::CivilDay a(2015, 1, 1);
     *       int yd_jan_1 = turbo::get_year_day(a);   // yd_jan_1 = 1
     *       turbo::CivilDay b(2015, 12, 31);
     *       int yd_dec_31 = turbo::get_year_day(b);  // yd_dec_31 = 365
     *       @endcode
     *
     */
    inline int get_year_day(CivilSecond cs) {
        return time_internal::cctz::get_yearday(cs);
    }

    /**
     * @ingroup turbo_times_civil_time
     * @brief Formats the given civil-time value into a string value of the following
     *       format:
     *       Type        | format
     *       ---------------------------------
     *       CivilSecond | YYYY-MM-DDTHH:MM:SS
     *       CivilMinute | YYYY-MM-DDTHH:MM
     *       CivilHour   | YYYY-MM-DDTHH
     *       CivilDay    | YYYY-MM-DD
     *       CivilMonth  | YYYY-MM
     *       CivilYear   | YYYY
     *       Example:
     *       @code
     *       turbo::CivilDay d = turbo::CivilDay(1969, 7, 20);
     *       std::string day_string = turbo::format_civil_time(d);  // "1969-07-20"
     *       @endcode
     *
     */
    std::string format_civil_time(CivilSecond c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for format_civil_time
     */
    std::string format_civil_time(CivilMinute c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for format_civil_time
     */
    std::string format_civil_time(CivilHour c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for format_civil_time
     */
    std::string format_civil_time(CivilDay c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for format_civil_time
     */
    std::string format_civil_time(CivilMonth c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for format_civil_time
     */
    std::string format_civil_time(CivilYear c);


    /**
     * @ingroup turbo_times_civil_time
     * @brief Parses a civil-time value from the specified `std::string_view` into the
     *       passed output parameter. Returns `true` upon successful parsing.
     *       The expected form of the input string is as follows:
     *       Type        | format
     *       ---------------------------------
     *       CivilSecond | YYYY-MM-DDTHH:MM:SS
     *       CivilMinute | YYYY-MM-DDTHH:MM
     *       CivilHour   | YYYY-MM-DDTHH
     *       CivilDay    | YYYY-MM-DD
     *       CivilMonth  | YYYY-MM
     *       CivilYear   | YYYY
     *       Example:
     *       @code
     *       turbo::CivilDay d;
     *       bool ok = turbo::parse_civil_time("2018-01-02", &d); // OK
     *       @endcode
     *       Note that parsing will fail if the string's format does not match the
     *       expected type exactly. `parse_lenient_civil_time()` below is more lenient.
     *
     */
    bool parse_civil_time(std::string_view s, CivilSecond *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_civil_time
     */
    bool parse_civil_time(std::string_view s, CivilMinute *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_civil_time
     */
    bool parse_civil_time(std::string_view s, CivilHour *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_civil_time
     */
    bool parse_civil_time(std::string_view s, CivilDay *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_civil_time
     */
    bool parse_civil_time(std::string_view s, CivilMonth *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_civil_time
     */
    bool parse_civil_time(std::string_view s, CivilYear *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief Parses any of the formats accepted by `turbo::parse_civil_time()`, but is more
     *      lenient if the format of the string does not exactly match the associated
     *      type.
     *
     *      Example:
     *      @code
     *      turbo::CivilDay d;
     *      bool ok = turbo::parse_lenient_civil_time("1969-07-20", &d); // OK
     *      ok = turbo::parse_lenient_civil_time("1969-07-20T10", &d);   // OK: T10 floored
     *      ok = turbo::parse_lenient_civil_time("1969-07", &d);   // OK: day defaults to 1
     *      @endcode
     * @param s The string to parse.
     * @param c The output parameter to store the parsed civil-time value.
     * @return `true` if the string was parsed successfully.
     */
    bool parse_lenient_civil_time(std::string_view s, CivilSecond *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_lenient_civil_time
     */
    bool parse_lenient_civil_time(std::string_view s, CivilMinute *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_lenient_civil_time
     */
    bool parse_lenient_civil_time(std::string_view s, CivilHour *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_lenient_civil_time
     */
    bool parse_lenient_civil_time(std::string_view s, CivilDay *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_lenient_civil_time
     */
    bool parse_lenient_civil_time(std::string_view s, CivilMonth *c);

    /**
     * @ingroup turbo_times_civil_time
     * @brief  overload for parse_lenient_civil_time
     */
    bool parse_lenient_civil_time(std::string_view s, CivilYear *c);

    namespace time_internal {  // For functions found via ADL on civil-time tags.

        // Streaming Operators
        //
        // Each civil-time type may be sent to an output stream using operator<<().
        // The result matches the string produced by `format_civil_time()`.
        //
        // Example:
        //
        //   turbo::CivilDay d = turbo::CivilDay(1969, 7, 20);
        //   std::cout << "Date is: " << d << "\n";
        //
        std::ostream &operator<<(std::ostream &os, CivilYear y);

        std::ostream &operator<<(std::ostream &os, CivilMonth m);

        std::ostream &operator<<(std::ostream &os, CivilDay d);

        std::ostream &operator<<(std::ostream &os, CivilHour h);

        std::ostream &operator<<(std::ostream &os, CivilMinute m);

        std::ostream &operator<<(std::ostream &os, CivilSecond s);


    }  // namespace time_internal

    // turbo_parse_flag()
    //
    // Parses the command-line flag string representation `s` into a civil-time
    // value. Flags must be specified in a format that is valid for
    // `turbo::parse_lenient_civil_time()`.
    bool turbo_parse_flag(std::string_view s, CivilSecond* c, std::string* error);
    bool turbo_parse_flag(std::string_view s, CivilMinute* c, std::string* error);
    bool turbo_parse_flag(std::string_view s, CivilHour* c, std::string* error);
    bool turbo_parse_flag(std::string_view s, CivilDay* c, std::string* error);
    bool turbo_parse_flag(std::string_view s, CivilMonth* c, std::string* error);
    bool turbo_parse_flag(std::string_view s, CivilYear* c, std::string* error);

    // turbo_unparse_flag()
    //
    // Unparses a civil-time value into a command-line string representation using
    // the format specified by `turbo::parse_civil_time()`.
    std::string turbo_unparse_flag(CivilSecond c);
    std::string turbo_unparse_flag(CivilMinute c);
    std::string turbo_unparse_flag(CivilHour c);
    std::string turbo_unparse_flag(CivilDay c);
    std::string turbo_unparse_flag(CivilMonth c);
    std::string turbo_unparse_flag(CivilYear c);
}  // namespace turbo

#endif  // TURBO_TIME_CIVIL_TIME_H_
