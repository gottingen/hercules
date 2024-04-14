// Copyright 2020 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include "turbo/times/cctz/civil_time.h"

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include "turbo/platform/port.h"
#include "turbo/testing/test.h"

namespace turbo {
    namespace time_internal {
        namespace cctz {

            namespace {

                template<typename T>
                std::string format(const T &t) {
                    std::stringstream ss;
                    ss << t;
                    return ss.str();
                }

            }  // namespace

#if __cpp_constexpr >= 201304 || (defined(_MSC_VER) && _MSC_VER >= 1910)
// Construction constexpr tests

            TEST_CASE("CivilTime, Normal") {
                constexpr civil_second css(2016, 1, 28, 17, 14, 12);
                static_assert(css.second() == 12, "Normal.second");
                constexpr civil_minute cmm(2016, 1, 28, 17, 14);
                static_assert(cmm.minute() == 14, "Normal.minute");
                constexpr civil_hour chh(2016, 1, 28, 17);
                static_assert(chh.hour() == 17, "Normal.hour");
                constexpr civil_day cd(2016, 1, 28);
                static_assert(cd.day() == 28, "Normal.day");
                constexpr civil_month cm(2016, 1);
                static_assert(cm.month() == 1, "Normal.month");
                constexpr civil_year cy(2016);
                static_assert(cy.year() == 2016, "Normal.year");
            }

            TEST_CASE("CivilTime, Conversion") {
                constexpr civil_year cy(2016);
                static_assert(cy.year() == 2016, "Conversion.year");
                constexpr civil_month cm(cy);
                static_assert(cm.month() == 1, "Conversion.month");
                constexpr civil_day cd(cm);
                static_assert(cd.day() == 1, "Conversion.day");
                constexpr civil_hour chh(cd);
                static_assert(chh.hour() == 0, "Conversion.hour");
                constexpr civil_minute cmm(chh);
                static_assert(cmm.minute() == 0, "Conversion.minute");
                constexpr civil_second css(cmm);
                static_assert(css.second() == 0, "Conversion.second");
            }

// Normalization constexpr tests

            TEST_CASE("CivilTime, Normalized") {
                constexpr civil_second cs(2016, 1, 28, 17, 14, 12);
                static_assert(cs.year() == 2016, "Normalized.year");
                static_assert(cs.month() == 1, "Normalized.month");
                static_assert(cs.day() == 28, "Normalized.day");
                static_assert(cs.hour() == 17, "Normalized.hour");
                static_assert(cs.minute() == 14, "Normalized.minute");
                static_assert(cs.second() == 12, "Normalized.second");
            }

            TEST_CASE("CivilTime, SecondOverflow") {
                constexpr civil_second cs(2016, 1, 28, 17, 14, 121);
                static_assert(cs.year() == 2016, "SecondOverflow.year");
                static_assert(cs.month() == 1, "SecondOverflow.month");
                static_assert(cs.day() == 28, "SecondOverflow.day");
                static_assert(cs.hour() == 17, "SecondOverflow.hour");
                static_assert(cs.minute() == 16, "SecondOverflow.minute");
                static_assert(cs.second() == 1, "SecondOverflow.second");
            }

            TEST_CASE("CivilTime, SecondUnderflow") {
                constexpr civil_second cs(2016, 1, 28, 17, 14, -121);
                static_assert(cs.year() == 2016, "SecondUnderflow.year");
                static_assert(cs.month() == 1, "SecondUnderflow.month");
                static_assert(cs.day() == 28, "SecondUnderflow.day");
                static_assert(cs.hour() == 17, "SecondUnderflow.hour");
                static_assert(cs.minute() == 11, "SecondUnderflow.minute");
                static_assert(cs.second() == 59, "SecondUnderflow.second");
            }

            TEST_CASE("CivilTime, MinuteOverflow") {
                constexpr civil_second cs(2016, 1, 28, 17, 121, 12);
                static_assert(cs.year() == 2016, "MinuteOverflow.year");
                static_assert(cs.month() == 1, "MinuteOverflow.month");
                static_assert(cs.day() == 28, "MinuteOverflow.day");
                static_assert(cs.hour() == 19, "MinuteOverflow.hour");
                static_assert(cs.minute() == 1, "MinuteOverflow.minute");
                static_assert(cs.second() == 12, "MinuteOverflow.second");
            }

            TEST_CASE("CivilTime, MinuteUnderflow") {
                constexpr civil_second cs(2016, 1, 28, 17, -121, 12);
                static_assert(cs.year() == 2016, "MinuteUnderflow.year");
                static_assert(cs.month() == 1, "MinuteUnderflow.month");
                static_assert(cs.day() == 28, "MinuteUnderflow.day");
                static_assert(cs.hour() == 14, "MinuteUnderflow.hour");
                static_assert(cs.minute() == 59, "MinuteUnderflow.minute");
                static_assert(cs.second() == 12, "MinuteUnderflow.second");
            }

            TEST_CASE("CivilTime, HourOverflow") {
                constexpr civil_second cs(2016, 1, 28, 49, 14, 12);
                static_assert(cs.year() == 2016, "HourOverflow.year");
                static_assert(cs.month() == 1, "HourOverflow.month");
                static_assert(cs.day() == 30, "HourOverflow.day");
                static_assert(cs.hour() == 1, "HourOverflow.hour");
                static_assert(cs.minute() == 14, "HourOverflow.minute");
                static_assert(cs.second() == 12, "HourOverflow.second");
            }

            TEST_CASE("CivilTime, HourUnderflow") {
                constexpr civil_second cs(2016, 1, 28, -49, 14, 12);
                static_assert(cs.year() == 2016, "HourUnderflow.year");
                static_assert(cs.month() == 1, "HourUnderflow.month");
                static_assert(cs.day() == 25, "HourUnderflow.day");
                static_assert(cs.hour() == 23, "HourUnderflow.hour");
                static_assert(cs.minute() == 14, "HourUnderflow.minute");
                static_assert(cs.second() == 12, "HourUnderflow.second");
            }

            TEST_CASE("CivilTime, MonthOverflow") {
                constexpr civil_second cs(2016, 25, 28, 17, 14, 12);
                static_assert(cs.year() == 2018, "MonthOverflow.year");
                static_assert(cs.month() == 1, "MonthOverflow.month");
                static_assert(cs.day() == 28, "MonthOverflow.day");
                static_assert(cs.hour() == 17, "MonthOverflow.hour");
                static_assert(cs.minute() == 14, "MonthOverflow.minute");
                static_assert(cs.second() == 12, "MonthOverflow.second");
            }

            TEST_CASE("CivilTime, MonthUnderflow") {
                constexpr civil_second cs(2016, -25, 28, 17, 14, 12);
                static_assert(cs.year() == 2013, "MonthUnderflow.year");
                static_assert(cs.month() == 11, "MonthUnderflow.month");
                static_assert(cs.day() == 28, "MonthUnderflow.day");
                static_assert(cs.hour() == 17, "MonthUnderflow.hour");
                static_assert(cs.minute() == 14, "MonthUnderflow.minute");
                static_assert(cs.second() == 12, "MonthUnderflow.second");
            }

            TEST_CASE("CivilTime, C4Overflow") {
                constexpr civil_second cs(2016, 1, 292195, 17, 14, 12);
                static_assert(cs.year() == 2816, "C4Overflow.year");
                static_assert(cs.month() == 1, "C4Overflow.month");
                static_assert(cs.day() == 1, "C4Overflow.day");
                static_assert(cs.hour() == 17, "C4Overflow.hour");
                static_assert(cs.minute() == 14, "C4Overflow.minute");
                static_assert(cs.second() == 12, "C4Overflow.second");
            }

            TEST_CASE("CivilTime, C4Underflow") {
                constexpr civil_second cs(2016, 1, -292195, 17, 14, 12);
                static_assert(cs.year() == 1215, "C4Underflow.year");
                static_assert(cs.month() == 12, "C4Underflow.month");
                static_assert(cs.day() == 30, "C4Underflow.day");
                static_assert(cs.hour() == 17, "C4Underflow.hour");
                static_assert(cs.minute() == 14, "C4Underflow.minute");
                static_assert(cs.second() == 12, "C4Underflow.second");
            }

            TEST_CASE("CivilTime, MixedNormalization") {
                constexpr civil_second cs(2016, -42, 122, 99, -147, 4949);
                static_assert(cs.year() == 2012, "MixedNormalization.year");
                static_assert(cs.month() == 10, "MixedNormalization.month");
                static_assert(cs.day() == 4, "MixedNormalization.day");
                static_assert(cs.hour() == 1, "MixedNormalization.hour");
                static_assert(cs.minute() == 55, "MixedNormalization.minute");
                static_assert(cs.second() == 29, "MixedNormalization.second");
            }

            // Relational constexpr tests

            TEST_CASE("CivilTime, Less") {
                constexpr civil_second cs1(2016, 1, 28, 17, 14, 12);
                constexpr civil_second cs2(2016, 1, 28, 17, 14, 13);
                constexpr bool less = cs1 < cs2;
                static_assert(less, "Less");
            }

// Arithmetic constexpr tests

            TEST_CASE("CivilTime, Addition") {
                constexpr civil_second cs1(2016, 1, 28, 17, 14, 12);
                constexpr civil_second cs2 = cs1 + 50;
                static_assert(cs2.year() == 2016, "Addition.year");
                static_assert(cs2.month() == 1, "Addition.month");
                static_assert(cs2.day() == 28, "Addition.day");
                static_assert(cs2.hour() == 17, "Addition.hour");
                static_assert(cs2.minute() == 15, "Addition.minute");
                static_assert(cs2.second() == 2, "Addition.second");
            }

            TEST_CASE("CivilTime, Subtraction") {
                constexpr civil_second cs1(2016, 1, 28, 17, 14, 12);
                constexpr civil_second cs2 = cs1 - 50;
                static_assert(cs2.year() == 2016, "Subtraction.year");
                static_assert(cs2.month() == 1, "Subtraction.month");
                static_assert(cs2.day() == 28, "Subtraction.day");
                static_assert(cs2.hour() == 17, "Subtraction.hour");
                static_assert(cs2.minute() == 13, "Subtraction.minute");
                static_assert(cs2.second() == 22, "Subtraction.second");
            }

            TEST_CASE("CivilTime, Difference") {
                constexpr civil_day cd1(2016, 1, 28);
                constexpr civil_day cd2(2015, 1, 28);
                constexpr int diff = cd1 - cd2;
                static_assert(diff == 365, "Difference");
            }

            // NOTE: Run this with --copt=-ftrapv to detect overflow problems.
            TEST_CASE("CivilTime, ConstructionWithHugeYear") {
                constexpr civil_hour h(-9223372036854775807, 1, 1, -1);
                static_assert(h.year() == -9223372036854775807 - 1,
                              "ConstructionWithHugeYear");
                static_assert(h.month() == 12, "ConstructionWithHugeYear");
                static_assert(h.day() == 31, "ConstructionWithHugeYear");
                static_assert(h.hour() == 23, "ConstructionWithHugeYear");
            }

            // NOTE: Run this with --copt=-ftrapv to detect overflow problems.
            TEST_CASE("CivilTime, DifferenceWithHugeYear") {
                {
                    constexpr civil_day d1(9223372036854775807, 1, 1);
                    constexpr civil_day d2(9223372036854775807, 12, 31);
                    static_assert(d2 - d1 == 364, "DifferenceWithHugeYear");
                }
                {
                    constexpr civil_day d1(-9223372036854775807 - 1, 1, 1);
                    constexpr civil_day d2(-9223372036854775807 - 1, 12, 31);
                    static_assert(d2 - d1 == 365, "DifferenceWithHugeYear");
                }
                {
                    // Check the limits of the return value at the end of the year range.
                    constexpr civil_day d1(9223372036854775807, 1, 1);
                    constexpr civil_day d2(9198119301927009252, 6, 6);
                    static_assert(d1 - d2 == 9223372036854775807, "DifferenceWithHugeYear");
                    static_assert((d2 - 1) - d1 == -9223372036854775807 - 1,
                                  "DifferenceWithHugeYear");
                }
                {
                    // Check the limits of the return value at the start of the year range.
                    constexpr civil_day d1(-9223372036854775807 - 1, 1, 1);
                    constexpr civil_day d2(-9198119301927009254, 7, 28);
                    static_assert(d2 - d1 == 9223372036854775807, "DifferenceWithHugeYear");
                    static_assert(d1 - (d2 + 1) == -9223372036854775807 - 1,
                                  "DifferenceWithHugeYear");
                }
                {
                    // Check the limits of the return value from either side of year 0.
                    constexpr civil_day d1(-12626367463883278, 9, 3);
                    constexpr civil_day d2(12626367463883277, 3, 28);
                    static_assert(d2 - d1 == 9223372036854775807, "DifferenceWithHugeYear");
                    static_assert(d1 - (d2 + 1) == -9223372036854775807 - 1,
                                  "DifferenceWithHugeYear");
                }
            }

            // NOTE: Run this with --copt=-ftrapv to detect overflow problems.
            TEST_CASE("CivilTime, DifferenceNoIntermediateOverflow") {
                {
                    // The difference up to the minute field would be below the minimum
                    // diff_t, but the 52 extra seconds brings us back to the minimum.
                    constexpr civil_second s1(-292277022657, 1, 27, 8, 29 - 1, 52);
                    constexpr civil_second s2(1970, 1, 1, 0, 0 - 1, 0);
                    static_assert(s1 - s2 == -9223372036854775807 - 1,
                                  "DifferenceNoIntermediateOverflow");
                }
                {
                    // The difference up to the minute field would be above the maximum
                    // diff_t, but the -53 extra seconds brings us back to the maximum.
                    constexpr civil_second s1(292277026596, 12, 4, 15, 30, 7 - 7);
                    constexpr civil_second s2(1970, 1, 1, 0, 0, 0 - 7);
                    static_assert(s1 - s2 == 9223372036854775807,
                                  "DifferenceNoIntermediateOverflow");
                }
            }

// Helper constexpr tests

            TEST_CASE("CivilTime, WeekDay") {
                constexpr civil_day cd(2016, 1, 28);
                constexpr weekday wd = get_weekday(cd);
                static_assert(wd == weekday::thursday, "Weekday");
            }

            TEST_CASE("CivilTime, NextWeekDay") {
                constexpr civil_day cd(2016, 1, 28);
                constexpr civil_day next = next_weekday(cd, weekday::thursday);
                static_assert(next.year() == 2016, "NextWeekDay.year");
                static_assert(next.month() == 2, "NextWeekDay.month");
                static_assert(next.day() == 4, "NextWeekDay.day");
            }

            TEST_CASE("CivilTime, PrevWeekDay") {
                constexpr civil_day cd(2016, 1, 28);
                constexpr civil_day prev = prev_weekday(cd, weekday::thursday);
                static_assert(prev.year() == 2016, "PrevWeekDay.year");
                static_assert(prev.month() == 1, "PrevWeekDay.month");
                static_assert(prev.day() == 21, "PrevWeekDay.day");
            }

            TEST_CASE("CivilTime, YearDay") {
                constexpr civil_day cd(2016, 1, 28);
                constexpr int yd = get_yearday(cd);
                static_assert(yd == 28, "YearDay");
            }

#endif  // __cpp_constexpr >= 201304 || (defined(_MSC_VER) && _MSC_VER >= 1910)

// The remaining tests do not use constexpr.

            TEST_CASE("CivilTime, DefaultConstruction") {
                civil_second ss;
                REQUIRE_EQ("1970-01-01T00:00:00", format(ss));

                civil_minute mm;
                REQUIRE_EQ("1970-01-01T00:00", format(mm));

                civil_hour hh;
                REQUIRE_EQ("1970-01-01T00", format(hh));

                civil_day d;
                REQUIRE_EQ("1970-01-01", format(d));

                civil_month m;
                REQUIRE_EQ("1970-01", format(m));

                civil_year y;
                REQUIRE_EQ("1970", format(y));
            }

            TEST_CASE("CivilTime, StructMember") {
                struct S {
                    civil_day day;
                };
                S s = {};
                REQUIRE_EQ(civil_day{}, s.day);
            }

            TEST_CASE("CivilTime, FieldsConstruction") {
                REQUIRE_EQ("2015-01-02T03:04:05", format(civil_second(2015, 1, 2, 3, 4, 5)));
                REQUIRE_EQ("2015-01-02T03:04:00", format(civil_second(2015, 1, 2, 3, 4)));
                REQUIRE_EQ("2015-01-02T03:00:00", format(civil_second(2015, 1, 2, 3)));
                REQUIRE_EQ("2015-01-02T00:00:00", format(civil_second(2015, 1, 2)));
                REQUIRE_EQ("2015-01-01T00:00:00", format(civil_second(2015, 1)));
                REQUIRE_EQ("2015-01-01T00:00:00", format(civil_second(2015)));

                REQUIRE_EQ("2015-01-02T03:04", format(civil_minute(2015, 1, 2, 3, 4, 5)));
                REQUIRE_EQ("2015-01-02T03:04", format(civil_minute(2015, 1, 2, 3, 4)));
                REQUIRE_EQ("2015-01-02T03:00", format(civil_minute(2015, 1, 2, 3)));
                REQUIRE_EQ("2015-01-02T00:00", format(civil_minute(2015, 1, 2)));
                REQUIRE_EQ("2015-01-01T00:00", format(civil_minute(2015, 1)));
                REQUIRE_EQ("2015-01-01T00:00", format(civil_minute(2015)));

                REQUIRE_EQ("2015-01-02T03", format(civil_hour(2015, 1, 2, 3, 4, 5)));
                REQUIRE_EQ("2015-01-02T03", format(civil_hour(2015, 1, 2, 3, 4)));
                REQUIRE_EQ("2015-01-02T03", format(civil_hour(2015, 1, 2, 3)));
                REQUIRE_EQ("2015-01-02T00", format(civil_hour(2015, 1, 2)));
                REQUIRE_EQ("2015-01-01T00", format(civil_hour(2015, 1)));
                REQUIRE_EQ("2015-01-01T00", format(civil_hour(2015)));

                REQUIRE_EQ("2015-01-02", format(civil_day(2015, 1, 2, 3, 4, 5)));
                REQUIRE_EQ("2015-01-02", format(civil_day(2015, 1, 2, 3, 4)));
                REQUIRE_EQ("2015-01-02", format(civil_day(2015, 1, 2, 3)));
                REQUIRE_EQ("2015-01-02", format(civil_day(2015, 1, 2)));
                REQUIRE_EQ("2015-01-01", format(civil_day(2015, 1)));
                REQUIRE_EQ("2015-01-01", format(civil_day(2015)));

                REQUIRE_EQ("2015-01", format(civil_month(2015, 1, 2, 3, 4, 5)));
                REQUIRE_EQ("2015-01", format(civil_month(2015, 1, 2, 3, 4)));
                REQUIRE_EQ("2015-01", format(civil_month(2015, 1, 2, 3)));
                REQUIRE_EQ("2015-01", format(civil_month(2015, 1, 2)));
                REQUIRE_EQ("2015-01", format(civil_month(2015, 1)));
                REQUIRE_EQ("2015-01", format(civil_month(2015)));

                REQUIRE_EQ("2015", format(civil_year(2015, 1, 2, 3, 4, 5)));
                REQUIRE_EQ("2015", format(civil_year(2015, 1, 2, 3, 4)));
                REQUIRE_EQ("2015", format(civil_year(2015, 1, 2, 3)));
                REQUIRE_EQ("2015", format(civil_year(2015, 1, 2)));
                REQUIRE_EQ("2015", format(civil_year(2015, 1)));
                REQUIRE_EQ("2015", format(civil_year(2015)));
            }

            TEST_CASE("CivilTime, FieldsConstructionLimits") {
                const int kIntMax = std::numeric_limits<int>::max();
                REQUIRE_EQ("2038-01-19T03:14:07",
                          format(civil_second(1970, 1, 1, 0, 0, kIntMax)));
                REQUIRE_EQ("6121-02-11T05:21:07",
                          format(civil_second(1970, 1, 1, 0, kIntMax, kIntMax)));
                REQUIRE_EQ("251104-11-20T12:21:07",
                          format(civil_second(1970, 1, 1, kIntMax, kIntMax, kIntMax)));
                REQUIRE_EQ("6130715-05-30T12:21:07",
                          format(civil_second(1970, 1, kIntMax, kIntMax, kIntMax, kIntMax)));
                REQUIRE_EQ(
                        "185087685-11-26T12:21:07",
                        format(civil_second(1970, kIntMax, kIntMax, kIntMax, kIntMax, kIntMax)));

                const int kIntMin = std::numeric_limits<int>::min();
                REQUIRE_EQ("1901-12-13T20:45:52",
                          format(civil_second(1970, 1, 1, 0, 0, kIntMin)));
                REQUIRE_EQ("-2182-11-20T18:37:52",
                          format(civil_second(1970, 1, 1, 0, kIntMin, kIntMin)));
                REQUIRE_EQ("-247165-02-11T10:37:52",
                          format(civil_second(1970, 1, 1, kIntMin, kIntMin, kIntMin)));
                REQUIRE_EQ("-6126776-08-01T10:37:52",
                          format(civil_second(1970, 1, kIntMin, kIntMin, kIntMin, kIntMin)));
                REQUIRE_EQ(
                        "-185083747-10-31T10:37:52",
                        format(civil_second(1970, kIntMin, kIntMin, kIntMin, kIntMin, kIntMin)));
            }

            TEST_CASE("CivilTime, ImplicitCrossAlignment") {
                civil_year year(2015);
                civil_month month = year;
                civil_day day = month;
                civil_hour hour = day;
                civil_minute minute = hour;
                civil_second second = minute;

                second = year;
                REQUIRE_EQ(second, year);
                second = month;
                REQUIRE_EQ(second, month);
                second = day;
                REQUIRE_EQ(second, day);
                second = hour;
                REQUIRE_EQ(second, hour);
                second = minute;
                REQUIRE_EQ(second, minute);

                minute = year;
                REQUIRE_EQ(minute, year);
                minute = month;
                REQUIRE_EQ(minute, month);
                minute = day;
                REQUIRE_EQ(minute, day);
                minute = hour;
                REQUIRE_EQ(minute, hour);

                hour = year;
                REQUIRE_EQ(hour, year);
                hour = month;
                REQUIRE_EQ(hour, month);
                hour = day;
                REQUIRE_EQ(hour, day);

                day = year;
                REQUIRE_EQ(day, year);
                day = month;
                REQUIRE_EQ(day, month);

                month = year;
                REQUIRE_EQ(month, year);

                // Ensures unsafe conversions are not allowed.
                REQUIRE_FALSE((std::is_convertible<civil_second, civil_minute>::value));
                REQUIRE_FALSE((std::is_convertible<civil_second, civil_hour>::value));
                REQUIRE_FALSE((std::is_convertible<civil_second, civil_day>::value));
                REQUIRE_FALSE((std::is_convertible<civil_second, civil_month>::value));
                REQUIRE_FALSE((std::is_convertible<civil_second, civil_year>::value));

                REQUIRE_FALSE((std::is_convertible<civil_minute, civil_hour>::value));
                REQUIRE_FALSE((std::is_convertible<civil_minute, civil_day>::value));
                REQUIRE_FALSE((std::is_convertible<civil_minute, civil_month>::value));
                REQUIRE_FALSE((std::is_convertible<civil_minute, civil_year>::value));

                REQUIRE_FALSE((std::is_convertible<civil_hour, civil_day>::value));
                REQUIRE_FALSE((std::is_convertible<civil_hour, civil_month>::value));
                REQUIRE_FALSE((std::is_convertible<civil_hour, civil_year>::value));

                REQUIRE_FALSE((std::is_convertible<civil_day, civil_month>::value));
                REQUIRE_FALSE((std::is_convertible<civil_day, civil_year>::value));

                REQUIRE_FALSE((std::is_convertible<civil_month, civil_year>::value));
            }

            TEST_CASE("CivilTime, ExplicitCrossAlignment") {
                //
                // Assign from smaller units -> larger units
                //

                civil_second second(2015, 1, 2, 3, 4, 5);
                REQUIRE_EQ("2015-01-02T03:04:05", format(second));

                civil_minute minute(second);
                REQUIRE_EQ("2015-01-02T03:04", format(minute));

                civil_hour hour(minute);
                REQUIRE_EQ("2015-01-02T03", format(hour));

                civil_day day(hour);
                REQUIRE_EQ("2015-01-02", format(day));

                civil_month month(day);
                REQUIRE_EQ("2015-01", format(month));

                civil_year year(month);
                REQUIRE_EQ("2015", format(year));

                //
                // Now assign from larger units -> smaller units
                //

                month = civil_month(year);
                REQUIRE_EQ("2015-01", format(month));

                day = civil_day(month);
                REQUIRE_EQ("2015-01-01", format(day));

                hour = civil_hour(day);
                REQUIRE_EQ("2015-01-01T00", format(hour));

                minute = civil_minute(hour);
                REQUIRE_EQ("2015-01-01T00:00", format(minute));

                second = civil_second(minute);
                REQUIRE_EQ("2015-01-01T00:00:00", format(second));
            }

            // Metafunction to test whether difference is allowed between two types.
            template<typename T1, typename T2>
            struct HasDifference {
                template<typename U1, typename U2>
                static std::false_type test(...);

                template<typename U1, typename U2>
                static std::true_type test(decltype(std::declval<U1>() - std::declval<U2>()));

                static constexpr bool value = decltype(test<T1, T2>(0))::value;
            };

            TEST_CASE("CivilTime, DisallowCrossAlignedDifference") {
                // Difference is allowed between types with the same alignment.
                static_assert(HasDifference<civil_second, civil_second>::value, "");
                static_assert(HasDifference<civil_minute, civil_minute>::value, "");
                static_assert(HasDifference<civil_hour, civil_hour>::value, "");
                static_assert(HasDifference<civil_day, civil_day>::value, "");
                static_assert(HasDifference<civil_month, civil_month>::value, "");
                static_assert(HasDifference<civil_year, civil_year>::value, "");

                // Difference is disallowed between types with different alignments.
                static_assert(!HasDifference<civil_second, civil_minute>::value, "");
                static_assert(!HasDifference<civil_second, civil_hour>::value, "");
                static_assert(!HasDifference<civil_second, civil_day>::value, "");
                static_assert(!HasDifference<civil_second, civil_month>::value, "");
                static_assert(!HasDifference<civil_second, civil_year>::value, "");

                static_assert(!HasDifference<civil_minute, civil_hour>::value, "");
                static_assert(!HasDifference<civil_minute, civil_day>::value, "");
                static_assert(!HasDifference<civil_minute, civil_month>::value, "");
                static_assert(!HasDifference<civil_minute, civil_year>::value, "");

                static_assert(!HasDifference<civil_hour, civil_day>::value, "");
                static_assert(!HasDifference<civil_hour, civil_month>::value, "");
                static_assert(!HasDifference<civil_hour, civil_year>::value, "");

                static_assert(!HasDifference<civil_day, civil_month>::value, "");
                static_assert(!HasDifference<civil_day, civil_year>::value, "");

                static_assert(!HasDifference<civil_month, civil_year>::value, "");
            }

            TEST_CASE("CivilTime, ValueSemantics") {
                const civil_hour a(2015, 1, 2, 3);
                const civil_hour b = a;
                const civil_hour c(b);
                civil_hour d;
                d = c;
                REQUIRE_EQ("2015-01-02T03", format(d));
            }

            TEST_CASE("CivilTime, Relational") {
                // Tests that the alignment unit is ignored in comparison.
                const civil_year year(2014);
                const civil_month month(year);
                REQUIRE_EQ(year, month);

#define TEST_RELATIONAL(OLDER, YOUNGER) \
  do {                                  \
    REQUIRE_FALSE(OLDER < OLDER);        \
    REQUIRE_FALSE(OLDER > OLDER);        \
    REQUIRE(OLDER >= OLDER);        \
    REQUIRE(OLDER <= OLDER);        \
    REQUIRE_FALSE(YOUNGER < YOUNGER);    \
    REQUIRE_FALSE(YOUNGER > YOUNGER);    \
    REQUIRE(YOUNGER >= YOUNGER);    \
    REQUIRE(YOUNGER <= YOUNGER);    \
    REQUIRE_EQ(OLDER, OLDER);            \
    REQUIRE_NE(OLDER, YOUNGER);          \
    REQUIRE_LT(OLDER, YOUNGER);          \
    REQUIRE_LE(OLDER, YOUNGER);          \
    REQUIRE_GT(YOUNGER, OLDER);          \
    REQUIRE_GE(YOUNGER, OLDER);          \
  } while (0)

                // Alignment is ignored in comparison (verified above), so kSecond is used
                // to test comparison in all field positions.
                TEST_RELATIONAL(civil_second(2014, 1, 1, 0, 0, 0),
                                civil_second(2015, 1, 1, 0, 0, 0));
                TEST_RELATIONAL(civil_second(2014, 1, 1, 0, 0, 0),
                                civil_second(2014, 2, 1, 0, 0, 0));
                TEST_RELATIONAL(civil_second(2014, 1, 1, 0, 0, 0),
                                civil_second(2014, 1, 2, 0, 0, 0));
                TEST_RELATIONAL(civil_second(2014, 1, 1, 0, 0, 0),
                                civil_second(2014, 1, 1, 1, 0, 0));
                TEST_RELATIONAL(civil_second(2014, 1, 1, 1, 0, 0),
                                civil_second(2014, 1, 1, 1, 1, 0));
                TEST_RELATIONAL(civil_second(2014, 1, 1, 1, 1, 0),
                                civil_second(2014, 1, 1, 1, 1, 1));

                // Tests the relational operators of two different civil-time types.
                TEST_RELATIONAL(civil_day(2014, 1, 1), civil_minute(2014, 1, 1, 1, 1));
                TEST_RELATIONAL(civil_day(2014, 1, 1), civil_month(2014, 2));

#undef TEST_RELATIONAL
            }

            TEST_CASE("CivilTime, Arithmetic") {
                civil_second second(2015, 1, 2, 3, 4, 5);
                REQUIRE_EQ("2015-01-02T03:04:06", format(second += 1));
                REQUIRE_EQ("2015-01-02T03:04:07", format(second + 1));
                REQUIRE_EQ("2015-01-02T03:04:08", format(2 + second));
                REQUIRE_EQ("2015-01-02T03:04:05", format(second - 1));
                REQUIRE_EQ("2015-01-02T03:04:05", format(second -= 1));
                REQUIRE_EQ("2015-01-02T03:04:05", format(second++));
                REQUIRE_EQ("2015-01-02T03:04:07", format(++second));
                REQUIRE_EQ("2015-01-02T03:04:07", format(second--));
                REQUIRE_EQ("2015-01-02T03:04:05", format(--second));

                civil_minute minute(2015, 1, 2, 3, 4);
                REQUIRE_EQ("2015-01-02T03:05", format(minute += 1));
                REQUIRE_EQ("2015-01-02T03:06", format(minute + 1));
                REQUIRE_EQ("2015-01-02T03:07", format(2 + minute));
                REQUIRE_EQ("2015-01-02T03:04", format(minute - 1));
                REQUIRE_EQ("2015-01-02T03:04", format(minute -= 1));
                REQUIRE_EQ("2015-01-02T03:04", format(minute++));
                REQUIRE_EQ("2015-01-02T03:06", format(++minute));
                REQUIRE_EQ("2015-01-02T03:06", format(minute--));
                REQUIRE_EQ("2015-01-02T03:04", format(--minute));

                civil_hour hour(2015, 1, 2, 3);
                REQUIRE_EQ("2015-01-02T04", format(hour += 1));
                REQUIRE_EQ("2015-01-02T05", format(hour + 1));
                REQUIRE_EQ("2015-01-02T06", format(2 + hour));
                REQUIRE_EQ("2015-01-02T03", format(hour - 1));
                REQUIRE_EQ("2015-01-02T03", format(hour -= 1));
                REQUIRE_EQ("2015-01-02T03", format(hour++));
                REQUIRE_EQ("2015-01-02T05", format(++hour));
                REQUIRE_EQ("2015-01-02T05", format(hour--));
                REQUIRE_EQ("2015-01-02T03", format(--hour));

                civil_day day(2015, 1, 2);
                REQUIRE_EQ("2015-01-03", format(day += 1));
                REQUIRE_EQ("2015-01-04", format(day + 1));
                REQUIRE_EQ("2015-01-05", format(2 + day));
                REQUIRE_EQ("2015-01-02", format(day - 1));
                REQUIRE_EQ("2015-01-02", format(day -= 1));
                REQUIRE_EQ("2015-01-02", format(day++));
                REQUIRE_EQ("2015-01-04", format(++day));
                REQUIRE_EQ("2015-01-04", format(day--));
                REQUIRE_EQ("2015-01-02", format(--day));

                civil_month month(2015, 1);
                REQUIRE_EQ("2015-02", format(month += 1));
                REQUIRE_EQ("2015-03", format(month + 1));
                REQUIRE_EQ("2015-04", format(2 + month));
                REQUIRE_EQ("2015-01", format(month - 1));
                REQUIRE_EQ("2015-01", format(month -= 1));
                REQUIRE_EQ("2015-01", format(month++));
                REQUIRE_EQ("2015-03", format(++month));
                REQUIRE_EQ("2015-03", format(month--));
                REQUIRE_EQ("2015-01", format(--month));

                civil_year year(2015);
                REQUIRE_EQ("2016", format(year += 1));
                REQUIRE_EQ("2017", format(year + 1));
                REQUIRE_EQ("2018", format(2 + year));
                REQUIRE_EQ("2015", format(year - 1));
                REQUIRE_EQ("2015", format(year -= 1));
                REQUIRE_EQ("2015", format(year++));
                REQUIRE_EQ("2017", format(++year));
                REQUIRE_EQ("2017", format(year--));
                REQUIRE_EQ("2015", format(--year));
            }

            TEST_CASE("CivilTime, ArithmeticLimits") {
                const int kIntMax = std::numeric_limits<int>::max();
                const int kIntMin = std::numeric_limits<int>::min();

                civil_second second(1970, 1, 1, 0, 0, 0);
                second += kIntMax;
                REQUIRE_EQ("2038-01-19T03:14:07", format(second));
                second -= kIntMax;
                REQUIRE_EQ("1970-01-01T00:00:00", format(second));
                second += kIntMin;
                REQUIRE_EQ("1901-12-13T20:45:52", format(second));
                second -= kIntMin;
                REQUIRE_EQ("1970-01-01T00:00:00", format(second));

                civil_minute minute(1970, 1, 1, 0, 0);
                minute += kIntMax;
                REQUIRE_EQ("6053-01-23T02:07", format(minute));
                minute -= kIntMax;
                REQUIRE_EQ("1970-01-01T00:00", format(minute));
                minute += kIntMin;
                REQUIRE_EQ("-2114-12-08T21:52", format(minute));
                minute -= kIntMin;
                REQUIRE_EQ("1970-01-01T00:00", format(minute));

                civil_hour hour(1970, 1, 1, 0);
                hour += kIntMax;
                REQUIRE_EQ("246953-10-09T07", format(hour));
                hour -= kIntMax;
                REQUIRE_EQ("1970-01-01T00", format(hour));
                hour += kIntMin;
                REQUIRE_EQ("-243014-03-24T16", format(hour));
                hour -= kIntMin;
                REQUIRE_EQ("1970-01-01T00", format(hour));

                civil_day day(1970, 1, 1);
                day += kIntMax;
                REQUIRE_EQ("5881580-07-11", format(day));
                day -= kIntMax;
                REQUIRE_EQ("1970-01-01", format(day));
                day += kIntMin;
                REQUIRE_EQ("-5877641-06-23", format(day));
                day -= kIntMin;
                REQUIRE_EQ("1970-01-01", format(day));

                civil_month month(1970, 1);
                month += kIntMax;
                REQUIRE_EQ("178958940-08", format(month));
                month -= kIntMax;
                REQUIRE_EQ("1970-01", format(month));
                month += kIntMin;
                REQUIRE_EQ("-178955001-05", format(month));
                month -= kIntMin;
                REQUIRE_EQ("1970-01", format(month));

                civil_year year(0);
                year += kIntMax;
                REQUIRE_EQ("2147483647", format(year));
                year -= kIntMax;
                REQUIRE_EQ("0", format(year));
                year += kIntMin;
                REQUIRE_EQ("-2147483648", format(year));
                year -= kIntMin;
                REQUIRE_EQ("0", format(year));
            }

            TEST_CASE("CivilTime, ArithmeticDifference") {
                civil_second second(2015, 1, 2, 3, 4, 5);
                REQUIRE_EQ(0, second - second);
                REQUIRE_EQ(10, (second + 10) - second);
                REQUIRE_EQ(-10, (second - 10) - second);

                civil_minute minute(2015, 1, 2, 3, 4);
                REQUIRE_EQ(0, minute - minute);
                REQUIRE_EQ(10, (minute + 10) - minute);
                REQUIRE_EQ(-10, (minute - 10) - minute);

                civil_hour hour(2015, 1, 2, 3);
                REQUIRE_EQ(0, hour - hour);
                REQUIRE_EQ(10, (hour + 10) - hour);
                REQUIRE_EQ(-10, (hour - 10) - hour);

                civil_day day(2015, 1, 2);
                REQUIRE_EQ(0, day - day);
                REQUIRE_EQ(10, (day + 10) - day);
                REQUIRE_EQ(-10, (day - 10) - day);

                civil_month month(2015, 1);
                REQUIRE_EQ(0, month - month);
                REQUIRE_EQ(10, (month + 10) - month);
                REQUIRE_EQ(-10, (month - 10) - month);

                civil_year year(2015);
                REQUIRE_EQ(0, year - year);
                REQUIRE_EQ(10, (year + 10) - year);
                REQUIRE_EQ(-10, (year - 10) - year);
            }

            TEST_CASE("CivilTime, DifferenceLimits") {
                const int kIntMax = std::numeric_limits<int>::max();
                const int kIntMin = std::numeric_limits<int>::min();

                // Check day arithmetic at the end of the year range.
                const civil_day max_day(kIntMax, 12, 31);
                REQUIRE_EQ(1, max_day - (max_day - 1));
                REQUIRE_EQ(-1, (max_day - 1) - max_day);

                // Check day arithmetic at the end of the year range.
                const civil_day min_day(kIntMin, 1, 1);
                REQUIRE_EQ(1, (min_day + 1) - min_day);
                REQUIRE_EQ(-1, min_day - (min_day + 1));

                // Check the limits of the return value.
                const civil_day d1(1970, 1, 1);
                const civil_day d2(5881580, 7, 11);
                REQUIRE_EQ(kIntMax, d2 - d1);
                REQUIRE_EQ(kIntMin, d1 - (d2 + 1));
            }

            TEST_CASE("CivilTime, Properties") {
                civil_second ss(2015, 2, 3, 4, 5, 6);
                REQUIRE_EQ(2015, ss.year());
                REQUIRE_EQ(2, ss.month());
                REQUIRE_EQ(3, ss.day());
                REQUIRE_EQ(4, ss.hour());
                REQUIRE_EQ(5, ss.minute());
                REQUIRE_EQ(6, ss.second());
                REQUIRE_EQ(weekday::tuesday, get_weekday(ss));
                REQUIRE_EQ(34, get_yearday(ss));

                civil_minute mm(2015, 2, 3, 4, 5, 6);
                REQUIRE_EQ(2015, mm.year());
                REQUIRE_EQ(2, mm.month());
                REQUIRE_EQ(3, mm.day());
                REQUIRE_EQ(4, mm.hour());
                REQUIRE_EQ(5, mm.minute());
                REQUIRE_EQ(0, mm.second());
                REQUIRE_EQ(weekday::tuesday, get_weekday(mm));
                REQUIRE_EQ(34, get_yearday(mm));

                civil_hour hh(2015, 2, 3, 4, 5, 6);
                REQUIRE_EQ(2015, hh.year());
                REQUIRE_EQ(2, hh.month());
                REQUIRE_EQ(3, hh.day());
                REQUIRE_EQ(4, hh.hour());
                REQUIRE_EQ(0, hh.minute());
                REQUIRE_EQ(0, hh.second());
                REQUIRE_EQ(weekday::tuesday, get_weekday(hh));
                REQUIRE_EQ(34, get_yearday(hh));

                civil_day d(2015, 2, 3, 4, 5, 6);
                REQUIRE_EQ(2015, d.year());
                REQUIRE_EQ(2, d.month());
                REQUIRE_EQ(3, d.day());
                REQUIRE_EQ(0, d.hour());
                REQUIRE_EQ(0, d.minute());
                REQUIRE_EQ(0, d.second());
                REQUIRE_EQ(weekday::tuesday, get_weekday(d));
                REQUIRE_EQ(34, get_yearday(d));

                civil_month m(2015, 2, 3, 4, 5, 6);
                REQUIRE_EQ(2015, m.year());
                REQUIRE_EQ(2, m.month());
                REQUIRE_EQ(1, m.day());
                REQUIRE_EQ(0, m.hour());
                REQUIRE_EQ(0, m.minute());
                REQUIRE_EQ(0, m.second());
                REQUIRE_EQ(weekday::sunday, get_weekday(m));
                REQUIRE_EQ(32, get_yearday(m));

                civil_year y(2015, 2, 3, 4, 5, 6);
                REQUIRE_EQ(2015, y.year());
                REQUIRE_EQ(1, y.month());
                REQUIRE_EQ(1, y.day());
                REQUIRE_EQ(0, y.hour());
                REQUIRE_EQ(0, y.minute());
                REQUIRE_EQ(0, y.second());
                REQUIRE_EQ(weekday::thursday, get_weekday(y));
                REQUIRE_EQ(1, get_yearday(y));
            }

            TEST_CASE("CivilTime, OutputStream") {
                // Tests formatting of civil_year, which does not pad.
                REQUIRE_EQ("2016", format(civil_year(2016)));
                REQUIRE_EQ("123", format(civil_year(123)));
                REQUIRE_EQ("0", format(civil_year(0)));
                REQUIRE_EQ("-1", format(civil_year(-1)));

                // Tests formatting of sub-year types, which pad to 2 digits
                REQUIRE_EQ("2016-02", format(civil_month(2016, 2)));
                REQUIRE_EQ("2016-02-03", format(civil_day(2016, 2, 3)));
                REQUIRE_EQ("2016-02-03T04", format(civil_hour(2016, 2, 3, 4)));
                REQUIRE_EQ("2016-02-03T04:05", format(civil_minute(2016, 2, 3, 4, 5)));
                REQUIRE_EQ("2016-02-03T04:05:06", format(civil_second(2016, 2, 3, 4, 5, 6)));

                // Tests formatting of weekday.
                REQUIRE_EQ("Monday", format(weekday::monday));
                REQUIRE_EQ("Tuesday", format(weekday::tuesday));
                REQUIRE_EQ("Wednesday", format(weekday::wednesday));
                REQUIRE_EQ("Thursday", format(weekday::thursday));
                REQUIRE_EQ("Friday", format(weekday::friday));
                REQUIRE_EQ("Saturday", format(weekday::saturday));
                REQUIRE_EQ("Sunday", format(weekday::sunday));
            }

            TEST_CASE("CivilTime, OutputStreamLeftFillWidth") {
                civil_second cs(2016, 2, 3, 4, 5, 6);
                {
                    std::stringstream ss;
                    ss << std::left << std::setfill('.');
                    ss << std::setw(3) << 'X';
                    ss << std::setw(21) << civil_year(cs);
                    ss << std::setw(3) << 'X';
                    REQUIRE_EQ("X..2016.................X..", ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::left << std::setfill('.');
                    ss << std::setw(3) << 'X';
                    ss << std::setw(21) << civil_month(cs);
                    ss << std::setw(3) << 'X';
                    REQUIRE_EQ("X..2016-02..............X..", ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::left << std::setfill('.');
                    ss << std::setw(3) << 'X';
                    ss << std::setw(21) << civil_day(cs);
                    ss << std::setw(3) << 'X';
                    REQUIRE_EQ("X..2016-02-03...........X..", ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::left << std::setfill('.');
                    ss << std::setw(3) << 'X';
                    ss << std::setw(21) << civil_hour(cs);
                    ss << std::setw(3) << 'X';
                    REQUIRE_EQ("X..2016-02-03T04........X..", ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::left << std::setfill('.');
                    ss << std::setw(3) << 'X';
                    ss << std::setw(21) << civil_minute(cs);
                    ss << std::setw(3) << 'X';
                    REQUIRE_EQ("X..2016-02-03T04:05.....X..", ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::left << std::setfill('.');
                    ss << std::setw(3) << 'X';
                    ss << std::setw(21) << civil_second(cs);
                    ss << std::setw(3) << 'X';
                    REQUIRE_EQ("X..2016-02-03T04:05:06..X..", ss.str());
                }
            }

            TEST_CASE("CivilTime, NextPrevWeekday") {
                // Jan 1, 1970 was a Thursday.
                const civil_day thursday(1970, 1, 1);
                REQUIRE_EQ(weekday::thursday, get_weekday(thursday));

                // Thursday -> Thursday
                civil_day d = next_weekday(thursday, weekday::thursday);
                REQUIRE_EQ(7, d - thursday) ;
                REQUIRE_EQ(d - 14, prev_weekday(thursday, weekday::thursday));

                // Thursday -> Friday
                d = next_weekday(thursday, weekday::friday);
                REQUIRE_EQ(1, d - thursday) ;
                REQUIRE_EQ(d - 7, prev_weekday(thursday, weekday::friday));

                // Thursday -> Saturday
                d = next_weekday(thursday, weekday::saturday);
                REQUIRE_EQ(2, d - thursday) ;
                REQUIRE_EQ(d - 7, prev_weekday(thursday, weekday::saturday));

                // Thursday -> Sunday
                d = next_weekday(thursday, weekday::sunday);
                REQUIRE_EQ(3, d - thursday) ;
                REQUIRE_EQ(d - 7, prev_weekday(thursday, weekday::sunday));

                // Thursday -> Monday
                d = next_weekday(thursday, weekday::monday);
                REQUIRE_EQ(4, d - thursday) ;
                REQUIRE_EQ(d - 7, prev_weekday(thursday, weekday::monday));

                // Thursday -> Tuesday
                d = next_weekday(thursday, weekday::tuesday);
                REQUIRE_EQ(5, d - thursday) ;
                REQUIRE_EQ(d - 7, prev_weekday(thursday, weekday::tuesday));

                // Thursday -> Wednesday
                d = next_weekday(thursday, weekday::wednesday);
                REQUIRE_EQ(6, d - thursday) ;
                REQUIRE_EQ(d - 7, prev_weekday(thursday, weekday::wednesday));
            }

            TEST_CASE("CivilTime, NormalizeWithHugeYear") {
                civil_month c(9223372036854775807, 1);
                REQUIRE_EQ("9223372036854775807-01", format(c));
                c = c - 1;  // Causes normalization
                REQUIRE_EQ("9223372036854775806-12", format(c));

                c = civil_month(-9223372036854775807 - 1, 1);
                REQUIRE_EQ("-9223372036854775808-01", format(c));
                c = c + 12;  // Causes normalization
                REQUIRE_EQ("-9223372036854775807-01", format(c));
            }

            TEST_CASE("CivilTime, LeapYears") {
                // Test data for leap years.
                const struct {
                    int year;
                    int days;
                    struct {
                        int month;
                        int day;
                    } leap_day;  // The date of the day after Feb 28.
                } kLeapYearTable[]{
                        {1900, 365, {3, 1}},
                        {1999, 365, {3, 1}},
                        {2000, 366, {2, 29}},  // leap year
                        {2001, 365, {3, 1}},
                        {2002, 365, {3, 1}},
                        {2003, 365, {3, 1}},
                        {2004, 366, {2, 29}},  // leap year
                        {2005, 365, {3, 1}},
                        {2006, 365, {3, 1}},
                        {2007, 365, {3, 1}},
                        {2008, 366, {2, 29}},  // leap year
                        {2009, 365, {3, 1}},
                        {2100, 365, {3, 1}},
                };

                for (const auto &e: kLeapYearTable) {
                    // Tests incrementing through the leap day.
                    const civil_day feb28(e.year, 2, 28);
                    const civil_day next_day = feb28 + 1;
                    REQUIRE_EQ(e.leap_day.month, next_day.month());
                    REQUIRE_EQ(e.leap_day.day, next_day.day());

                    // Tests difference in days of leap years.
                    const civil_year year(feb28);
                    const civil_year next_year = year + 1;
                    REQUIRE_EQ(e.days, civil_day(next_year) -civil_day(year));
                }
            }

            TEST_CASE("CivilTime, FirstThursdayInMonth") {
                const civil_day nov1(2014, 11, 1);
                const civil_day thursday = next_weekday(nov1 - 1, weekday::thursday);
                REQUIRE_EQ("2014-11-06", format(thursday));

                // Bonus: Date of Thanksgiving in the United States
                // Rule: Fourth Thursday of November
                const civil_day thanksgiving = thursday + 7 * 3;
                REQUIRE_EQ("2014-11-27", format(thanksgiving));
            }

        }  // namespace cctz
    }  // namespace time_internal
}  // namespace turbo
