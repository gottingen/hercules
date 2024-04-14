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

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include "turbo/platform/port.h"
#include "turbo/times/cctz/time_zone.h"

#if defined(__linux__)

#include <features.h>

#endif

#include "turbo/testing/test.h"
#include "turbo/times/cctz/civil_time.h"

namespace chrono = std::chrono;

namespace turbo::time_internal {
    namespace cctz {

        namespace {

// This helper is a macro so that failed expectations show up with the
// correct line numbers.
#define ExpectTime(tp, tz, y, m, d, hh, mm, ss, off, isdst, zone) \
  do {                                                            \
    time_zone::absolute_lookup al = tz.lookup(tp);                \
    REQUIRE_EQ(y, al.cs.year());                                   \
    REQUIRE_EQ(m, al.cs.month());                                  \
    REQUIRE_EQ(d, al.cs.day());                                    \
    REQUIRE_EQ(hh, al.cs.hour());                                  \
    REQUIRE_EQ(mm, al.cs.minute());                                \
    REQUIRE_EQ(ss, al.cs.second());                                \
    REQUIRE_EQ(off, al.offset);                                    \
    REQUIRE(isdst == al.is_dst);                              \
    REQUIRE_EQ(zone, std::string(al.abbr));                                  \
  } while (0)

            const char RFC3339_full[] = "%Y-%m-%d%ET%H:%M:%E*S%Ez";
            const char RFC3339_sec[] = "%Y-%m-%d%ET%H:%M:%S%Ez";

            const char RFC1123_full[] = "%a, %d %b %Y %H:%M:%S %z";
            const char RFC1123_no_wday[] = "%d %b %Y %H:%M:%S %z";

            // A helper that tests the given format specifier by itself, and with leading
            // and trailing characters.  For example: TestFormatSpecifier(tp, "%a", "Thu").
            template<typename D>
            void TestFormatSpecifier(time_point<D> tp, time_zone tz, const std::string &fmt,
                                     const std::string &ans) {
                REQUIRE_EQ(ans, format(fmt, tp, tz));
                REQUIRE_EQ("xxx " + ans, format("xxx " + fmt, tp, tz));
                REQUIRE_EQ(ans + " yyy", format(fmt + " yyy", tp, tz));
                REQUIRE_EQ("xxx " + ans + " yyy", format("xxx " + fmt + " yyy", tp, tz));
            }

        }  // namespace

        //
        // Testing format()
        //

        TEST_CASE("format, TimePointResolution") {
            const char kFmt[] = "%H:%M:%E*S";
            const time_zone utc = utc_time_zone();
            const time_point<chrono::nanoseconds> t0 =
                    chrono::system_clock::from_time_t(1420167845) +
                    chrono::milliseconds(123) + chrono::microseconds(456) +
                    chrono::nanoseconds(789);
            REQUIRE_EQ(
                    "03:04:05.123456789",
                    format(kFmt, chrono::time_point_cast<chrono::nanoseconds>(t0), utc));
            REQUIRE_EQ(
                    "03:04:05.123456",
                    format(kFmt, chrono::time_point_cast<chrono::microseconds>(t0), utc));
            REQUIRE_EQ(
                    "03:04:05.123",
                    format(kFmt, chrono::time_point_cast<chrono::milliseconds>(t0), utc));
            REQUIRE_EQ("03:04:05",
                       format(kFmt, chrono::time_point_cast<chrono::seconds>(t0), utc));
            REQUIRE_EQ(
                    "03:04:05",
                    format(kFmt,
                           chrono::time_point_cast<turbo::time_internal::cctz::seconds>(t0),
                           utc));
            REQUIRE_EQ("03:04:00",
                       format(kFmt, chrono::time_point_cast<chrono::minutes>(t0), utc));
            REQUIRE_EQ("03:00:00",
                       format(kFmt, chrono::time_point_cast<chrono::hours>(t0), utc));
        }

        TEST_CASE("format, TimePointExtendedResolution") {
            const char kFmt[] = "%H:%M:%E*S";
            const time_zone utc = utc_time_zone();
            const time_point<turbo::time_internal::cctz::seconds> tp =
                    chrono::time_point_cast<turbo::time_internal::cctz::seconds>(
                            chrono::system_clock::from_time_t(0)) +
                    chrono::hours(12) + chrono::minutes(34) + chrono::seconds(56);

            REQUIRE_EQ(
                    "12:34:56.123456789012345",
                    detail::format(kFmt, tp, detail::femtoseconds(123456789012345), utc));
            REQUIRE_EQ(
                    "12:34:56.012345678901234",
                    detail::format(kFmt, tp, detail::femtoseconds(12345678901234), utc));
            REQUIRE_EQ("12:34:56.001234567890123",
                       detail::format(kFmt, tp, detail::femtoseconds(1234567890123), utc));
            REQUIRE_EQ("12:34:56.000123456789012",
                       detail::format(kFmt, tp, detail::femtoseconds(123456789012), utc));

            REQUIRE_EQ("12:34:56.000000000000123",
                       detail::format(kFmt, tp, detail::femtoseconds(123), utc));
            REQUIRE_EQ("12:34:56.000000000000012",
                       detail::format(kFmt, tp, detail::femtoseconds(12), utc));
            REQUIRE_EQ("12:34:56.000000000000001",
                       detail::format(kFmt, tp, detail::femtoseconds(1), utc));
        }

        TEST_CASE("format, Basics") {
            time_zone tz = utc_time_zone();
            time_point<chrono::nanoseconds> tp = chrono::system_clock::from_time_t(0);

            // Starts with a couple basic edge cases.
            REQUIRE_EQ("", format("", tp, tz));
            REQUIRE_EQ(" ", format(" ", tp, tz));
            REQUIRE_EQ("  ", format("  ", tp, tz));
            REQUIRE_EQ("xxx", format("xxx", tp, tz));
            std::string big(128, 'x');
            REQUIRE_EQ(big, format(big, tp, tz));
            // Cause the 1024-byte buffer to grow.
            std::string bigger(100000, 'x');
            REQUIRE_EQ(bigger, format(bigger, tp, tz));

            tp += chrono::hours(13) + chrono::minutes(4) + chrono::seconds(5);
            tp += chrono::milliseconds(6) + chrono::microseconds(7) +
                  chrono::nanoseconds(8);
            REQUIRE_EQ("1970-01-01", format("%Y-%m-%d", tp, tz));
            REQUIRE_EQ("13:04:05", format("%H:%M:%S", tp, tz));
            REQUIRE_EQ("13:04:05.006", format("%H:%M:%E3S", tp, tz));
            REQUIRE_EQ("13:04:05.006007", format("%H:%M:%E6S", tp, tz));
            REQUIRE_EQ("13:04:05.006007008", format("%H:%M:%E9S", tp, tz));
        }

        TEST_CASE("format, PosixConversions") {
            const time_zone tz = utc_time_zone();
            auto tp = chrono::system_clock::from_time_t(0);

            TestFormatSpecifier(tp, tz, "%d", "01");
            TestFormatSpecifier(tp, tz, "%e", " 1");  // extension but internal support
            TestFormatSpecifier(tp, tz, "%H", "00");
            TestFormatSpecifier(tp, tz, "%I", "12");
            TestFormatSpecifier(tp, tz, "%j", "001");
            TestFormatSpecifier(tp, tz, "%m", "01");
            TestFormatSpecifier(tp, tz, "%M", "00");
            TestFormatSpecifier(tp, tz, "%S", "00");
            TestFormatSpecifier(tp, tz, "%U", "00");
#if !defined(__EMSCRIPTEN__)
            TestFormatSpecifier(tp, tz, "%w", "4");  // 4=Thursday
#endif
            TestFormatSpecifier(tp, tz, "%W", "00");
            TestFormatSpecifier(tp, tz, "%y", "70");
            TestFormatSpecifier(tp, tz, "%Y", "1970");
            TestFormatSpecifier(tp, tz, "%z", "+0000");
            TestFormatSpecifier(tp, tz, "%Z", "UTC");
            TestFormatSpecifier(tp, tz, "%%", "%");

#if defined(__linux__)
            // SU/C99/TZ extensions
            TestFormatSpecifier(tp, tz, "%C", "19");
            TestFormatSpecifier(tp, tz, "%D", "01/01/70");
            TestFormatSpecifier(tp, tz, "%F", "1970-01-01");
            TestFormatSpecifier(tp, tz, "%g", "70");
            TestFormatSpecifier(tp, tz, "%G", "1970");
#if defined(__GLIBC__)
            TestFormatSpecifier(tp, tz, "%k", " 0");
            TestFormatSpecifier(tp, tz, "%l", "12");
#endif
            TestFormatSpecifier(tp, tz, "%n", "\n");
            TestFormatSpecifier(tp, tz, "%R", "00:00");
            TestFormatSpecifier(tp, tz, "%t", "\t");
            TestFormatSpecifier(tp, tz, "%T", "00:00:00");
            TestFormatSpecifier(tp, tz, "%u", "4");  // 4=Thursday
            TestFormatSpecifier(tp, tz, "%V", "01");
            TestFormatSpecifier(tp, tz, "%s", "0");
#endif
        }

        TEST_CASE("format, LocaleSpecific") {
            const time_zone tz = utc_time_zone();
            auto tp = chrono::system_clock::from_time_t(0);

            TestFormatSpecifier(tp, tz, "%a", "Thu");
            TestFormatSpecifier(tp, tz, "%A", "Thursday");
            TestFormatSpecifier(tp, tz, "%b", "Jan");
            TestFormatSpecifier(tp, tz, "%B", "January");

            // %c should at least produce the numeric year and time-of-day.
            const std::string s = format("%c", tp, utc_time_zone());

            TestFormatSpecifier(tp, tz, "%p", "AM");
            TestFormatSpecifier(tp, tz, "%x", "01/01/70");
            TestFormatSpecifier(tp, tz, "%X", "00:00:00");

#if defined(__linux__)
            // SU/C99/TZ extensions
            TestFormatSpecifier(tp, tz, "%h", "Jan");  // Same as %b
#if defined(__GLIBC__)
            TestFormatSpecifier(tp, tz, "%P", "am");
#endif
            TestFormatSpecifier(tp, tz, "%r", "12:00:00 AM");

            // Modified conversion specifiers %E_
            TestFormatSpecifier(tp, tz, "%Ec", "Thu Jan  1 00:00:00 1970");
            TestFormatSpecifier(tp, tz, "%EC", "19");
            TestFormatSpecifier(tp, tz, "%Ex", "01/01/70");
            TestFormatSpecifier(tp, tz, "%EX", "00:00:00");
            TestFormatSpecifier(tp, tz, "%Ey", "70");
            TestFormatSpecifier(tp, tz, "%EY", "1970");

            // Modified conversion specifiers %O_
            TestFormatSpecifier(tp, tz, "%Od", "01");
            TestFormatSpecifier(tp, tz, "%Oe", " 1");
            TestFormatSpecifier(tp, tz, "%OH", "00");
            TestFormatSpecifier(tp, tz, "%OI", "12");
            TestFormatSpecifier(tp, tz, "%Om", "01");
            TestFormatSpecifier(tp, tz, "%OM", "00");
            TestFormatSpecifier(tp, tz, "%OS", "00");
            TestFormatSpecifier(tp, tz, "%Ou", "4");  // 4=Thursday
            TestFormatSpecifier(tp, tz, "%OU", "00");
            TestFormatSpecifier(tp, tz, "%OV", "01");
            TestFormatSpecifier(tp, tz, "%Ow", "4");  // 4=Thursday
            TestFormatSpecifier(tp, tz, "%OW", "00");
            TestFormatSpecifier(tp, tz, "%Oy", "70");
#endif
        }

        TEST_CASE("format, Escaping") {
            const time_zone tz = utc_time_zone();
            auto tp = chrono::system_clock::from_time_t(0);

            TestFormatSpecifier(tp, tz, "%%", "%");
            TestFormatSpecifier(tp, tz, "%%a", "%a");
            TestFormatSpecifier(tp, tz, "%%b", "%b");
            TestFormatSpecifier(tp, tz, "%%Ea", "%Ea");
            TestFormatSpecifier(tp, tz, "%%Es", "%Es");
            TestFormatSpecifier(tp, tz, "%%E3S", "%E3S");
            TestFormatSpecifier(tp, tz, "%%OS", "%OS");
            TestFormatSpecifier(tp, tz, "%%O3S", "%O3S");

            // Multiple levels of escaping.
            TestFormatSpecifier(tp, tz, "%%%Y", "%1970");
            TestFormatSpecifier(tp, tz, "%%%E3S", "%00.000");
            TestFormatSpecifier(tp, tz, "%%%%E3S", "%%E3S");
        }

        TEST_CASE("format, ExtendedSeconds") {
            const time_zone tz = utc_time_zone();

            // No subseconds.
            time_point<chrono::nanoseconds> tp = chrono::system_clock::from_time_t(0);
            tp += chrono::seconds(5);
            REQUIRE_EQ("05", format("%E*S", tp, tz));
            REQUIRE_EQ("05", format("%E0S", tp, tz));
            REQUIRE_EQ("05.0", format("%E1S", tp, tz));
            REQUIRE_EQ("05.00", format("%E2S", tp, tz));
            REQUIRE_EQ("05.000", format("%E3S", tp, tz));
            REQUIRE_EQ("05.0000", format("%E4S", tp, tz));
            REQUIRE_EQ("05.00000", format("%E5S", tp, tz));
            REQUIRE_EQ("05.000000", format("%E6S", tp, tz));
            REQUIRE_EQ("05.0000000", format("%E7S", tp, tz));
            REQUIRE_EQ("05.00000000", format("%E8S", tp, tz));
            REQUIRE_EQ("05.000000000", format("%E9S", tp, tz));
            REQUIRE_EQ("05.0000000000", format("%E10S", tp, tz));
            REQUIRE_EQ("05.00000000000", format("%E11S", tp, tz));
            REQUIRE_EQ("05.000000000000", format("%E12S", tp, tz));
            REQUIRE_EQ("05.0000000000000", format("%E13S", tp, tz));
            REQUIRE_EQ("05.00000000000000", format("%E14S", tp, tz));
            REQUIRE_EQ("05.000000000000000", format("%E15S", tp, tz));

            // With subseconds.
            tp += chrono::milliseconds(6) + chrono::microseconds(7) +
                  chrono::nanoseconds(8);
            REQUIRE_EQ("05.006007008", format("%E*S", tp, tz));
            REQUIRE_EQ("05", format("%E0S", tp, tz));
            REQUIRE_EQ("05.0", format("%E1S", tp, tz));
            REQUIRE_EQ("05.00", format("%E2S", tp, tz));
            REQUIRE_EQ("05.006", format("%E3S", tp, tz));
            REQUIRE_EQ("05.0060", format("%E4S", tp, tz));
            REQUIRE_EQ("05.00600", format("%E5S", tp, tz));
            REQUIRE_EQ("05.006007", format("%E6S", tp, tz));
            REQUIRE_EQ("05.0060070", format("%E7S", tp, tz));
            REQUIRE_EQ("05.00600700", format("%E8S", tp, tz));
            REQUIRE_EQ("05.006007008", format("%E9S", tp, tz));
            REQUIRE_EQ("05.0060070080", format("%E10S", tp, tz));
            REQUIRE_EQ("05.00600700800", format("%E11S", tp, tz));
            REQUIRE_EQ("05.006007008000", format("%E12S", tp, tz));
            REQUIRE_EQ("05.0060070080000", format("%E13S", tp, tz));
            REQUIRE_EQ("05.00600700800000", format("%E14S", tp, tz));
            REQUIRE_EQ("05.006007008000000", format("%E15S", tp, tz));

            // Times before the Unix epoch.
            tp = chrono::system_clock::from_time_t(0) + chrono::microseconds(-1);
            REQUIRE_EQ("1969-12-31 23:59:59.999999",
                       format("%Y-%m-%d %H:%M:%E*S", tp, tz));

            // Here is a "%E*S" case we got wrong for a while.  While the first
            // instant below is correctly rendered as "...:07.333304", the second
            // one used to appear as "...:07.33330499999999999".
            tp = chrono::system_clock::from_time_t(0) +
                 chrono::microseconds(1395024427333304);
            REQUIRE_EQ("2014-03-17 02:47:07.333304",
                       format("%Y-%m-%d %H:%M:%E*S", tp, tz));
            tp += chrono::microseconds(1);
            REQUIRE_EQ("2014-03-17 02:47:07.333305",
                       format("%Y-%m-%d %H:%M:%E*S", tp, tz));
        }

        TEST_CASE("format, ExtendedSubeconds") {
            const time_zone tz = utc_time_zone();

            // No subseconds.
            time_point<chrono::nanoseconds> tp = chrono::system_clock::from_time_t(0);
            tp += chrono::seconds(5);
            REQUIRE_EQ("0", format("%E*f", tp, tz));
            REQUIRE_EQ("", format("%E0f", tp, tz));
            REQUIRE_EQ("0", format("%E1f", tp, tz));
            REQUIRE_EQ("00", format("%E2f", tp, tz));
            REQUIRE_EQ("000", format("%E3f", tp, tz));
            REQUIRE_EQ("0000", format("%E4f", tp, tz));
            REQUIRE_EQ("00000", format("%E5f", tp, tz));
            REQUIRE_EQ("000000", format("%E6f", tp, tz));
            REQUIRE_EQ("0000000", format("%E7f", tp, tz));
            REQUIRE_EQ("00000000", format("%E8f", tp, tz));
            REQUIRE_EQ("000000000", format("%E9f", tp, tz));
            REQUIRE_EQ("0000000000", format("%E10f", tp, tz));
            REQUIRE_EQ("00000000000", format("%E11f", tp, tz));
            REQUIRE_EQ("000000000000", format("%E12f", tp, tz));
            REQUIRE_EQ("0000000000000", format("%E13f", tp, tz));
            REQUIRE_EQ("00000000000000", format("%E14f", tp, tz));
            REQUIRE_EQ("000000000000000", format("%E15f", tp, tz));

            // With subseconds.
            tp += chrono::milliseconds(6) + chrono::microseconds(7) +
                  chrono::nanoseconds(8);
            REQUIRE_EQ("006007008", format("%E*f", tp, tz));
            REQUIRE_EQ("", format("%E0f", tp, tz));
            REQUIRE_EQ("0", format("%E1f", tp, tz));
            REQUIRE_EQ("00", format("%E2f", tp, tz));
            REQUIRE_EQ("006", format("%E3f", tp, tz));
            REQUIRE_EQ("0060", format("%E4f", tp, tz));
            REQUIRE_EQ("00600", format("%E5f", tp, tz));
            REQUIRE_EQ("006007", format("%E6f", tp, tz));
            REQUIRE_EQ("0060070", format("%E7f", tp, tz));
            REQUIRE_EQ("00600700", format("%E8f", tp, tz));
            REQUIRE_EQ("006007008", format("%E9f", tp, tz));
            REQUIRE_EQ("0060070080", format("%E10f", tp, tz));
            REQUIRE_EQ("00600700800", format("%E11f", tp, tz));
            REQUIRE_EQ("006007008000", format("%E12f", tp, tz));
            REQUIRE_EQ("0060070080000", format("%E13f", tp, tz));
            REQUIRE_EQ("00600700800000", format("%E14f", tp, tz));
            REQUIRE_EQ("006007008000000", format("%E15f", tp, tz));

            // Times before the Unix epoch.
            tp = chrono::system_clock::from_time_t(0) + chrono::microseconds(-1);
            REQUIRE_EQ("1969-12-31 23:59:59.999999",
                       format("%Y-%m-%d %H:%M:%S.%E*f", tp, tz));

            // Here is a "%E*S" case we got wrong for a while.  While the first
            // instant below is correctly rendered as "...:07.333304", the second
            // one used to appear as "...:07.33330499999999999".
            tp = chrono::system_clock::from_time_t(0) +
                 chrono::microseconds(1395024427333304);
            REQUIRE_EQ("2014-03-17 02:47:07.333304",
                       format("%Y-%m-%d %H:%M:%S.%E*f", tp, tz));
            tp += chrono::microseconds(1);
            REQUIRE_EQ("2014-03-17 02:47:07.333305",
                       format("%Y-%m-%d %H:%M:%S.%E*f", tp, tz));
        }

        TEST_CASE("format, CompareExtendSecondsVsSubseconds") {
            const time_zone tz = utc_time_zone();

            // This test case illustrates the differences/similarities between:
            //   fmt_A: %E<prec>S
            //   fmt_B: %S.%E<prec>f
            auto fmt_A = [](const std::string &prec) { return "%E" + prec + "S"; };
            auto fmt_B = [](const std::string &prec) { return "%S.%E" + prec + "f"; };

            // No subseconds:
            time_point<chrono::nanoseconds> tp = chrono::system_clock::from_time_t(0);
            tp += chrono::seconds(5);
            // ... %E*S and %S.%E*f are different.
            REQUIRE_EQ("05", format(fmt_A("*"), tp, tz));
            REQUIRE_EQ("05.0", format(fmt_B("*"), tp, tz));
            // ... %E0S and %S.%E0f are different.
            REQUIRE_EQ("05", format(fmt_A("0"), tp, tz));
            REQUIRE_EQ("05.", format(fmt_B("0"), tp, tz));
            // ... %E<prec>S and %S.%E<prec>f are the same for prec in [1:15].
            for (int prec = 1; prec <= 15; ++prec) {
                const std::string a = format(fmt_A(std::to_string(prec)), tp, tz);
                const std::string b = format(fmt_B(std::to_string(prec)), tp, tz);
                REQUIRE_EQ(a, b);
            }

            // With subseconds:
            // ... %E*S and %S.%E*f are the same.
            tp += chrono::milliseconds(6) + chrono::microseconds(7) +
                  chrono::nanoseconds(8);
            REQUIRE_EQ("05.006007008", format(fmt_A("*"), tp, tz));
            REQUIRE_EQ("05.006007008", format(fmt_B("*"), tp, tz));
            // ... %E0S and %S.%E0f are different.
            REQUIRE_EQ("05", format(fmt_A("0"), tp, tz));
            REQUIRE_EQ("05.", format(fmt_B("0"), tp, tz));
            // ... %E<prec>S and %S.%E<prec>f are the same for prec in [1:15].
            for (int prec = 1; prec <= 15; ++prec) {
                const std::string a = format(fmt_A(std::to_string(prec)), tp, tz);
                const std::string b = format(fmt_B(std::to_string(prec)), tp, tz);
                REQUIRE_EQ(a, b);
            }
        }

        TEST_CASE("format, ExtendedOffset") {
            const auto tp = chrono::system_clock::from_time_t(0);

            auto tz = fixed_time_zone(turbo::time_internal::cctz::seconds::zero());
            TestFormatSpecifier(tp, tz, "%z", "+0000");
            TestFormatSpecifier(tp, tz, "%:z", "+00:00");
            TestFormatSpecifier(tp, tz, "%Ez", "+00:00");

            tz = fixed_time_zone(chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%z", "+0000");
            TestFormatSpecifier(tp, tz, "%:z", "+00:00");
            TestFormatSpecifier(tp, tz, "%Ez", "+00:00");

            tz = fixed_time_zone(-chrono::seconds(56));  // NOTE: +00:00
            TestFormatSpecifier(tp, tz, "%z", "+0000");
            TestFormatSpecifier(tp, tz, "%:z", "+00:00");
            TestFormatSpecifier(tp, tz, "%Ez", "+00:00");

            tz = fixed_time_zone(chrono::minutes(34));
            TestFormatSpecifier(tp, tz, "%z", "+0034");
            TestFormatSpecifier(tp, tz, "%:z", "+00:34");
            TestFormatSpecifier(tp, tz, "%Ez", "+00:34");

            tz = fixed_time_zone(-chrono::minutes(34));
            TestFormatSpecifier(tp, tz, "%z", "-0034");
            TestFormatSpecifier(tp, tz, "%:z", "-00:34");
            TestFormatSpecifier(tp, tz, "%Ez", "-00:34");

            tz = fixed_time_zone(chrono::minutes(34) + chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%z", "+0034");
            TestFormatSpecifier(tp, tz, "%:z", "+00:34");
            TestFormatSpecifier(tp, tz, "%Ez", "+00:34");

            tz = fixed_time_zone(-chrono::minutes(34) - chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%z", "-0034");
            TestFormatSpecifier(tp, tz, "%:z", "-00:34");
            TestFormatSpecifier(tp, tz, "%Ez", "-00:34");

            tz = fixed_time_zone(chrono::hours(12));
            TestFormatSpecifier(tp, tz, "%z", "+1200");
            TestFormatSpecifier(tp, tz, "%:z", "+12:00");
            TestFormatSpecifier(tp, tz, "%Ez", "+12:00");

            tz = fixed_time_zone(-chrono::hours(12));
            TestFormatSpecifier(tp, tz, "%z", "-1200");
            TestFormatSpecifier(tp, tz, "%:z", "-12:00");
            TestFormatSpecifier(tp, tz, "%Ez", "-12:00");

            tz = fixed_time_zone(chrono::hours(12) + chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%z", "+1200");
            TestFormatSpecifier(tp, tz, "%:z", "+12:00");
            TestFormatSpecifier(tp, tz, "%Ez", "+12:00");

            tz = fixed_time_zone(-chrono::hours(12) - chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%z", "-1200");
            TestFormatSpecifier(tp, tz, "%:z", "-12:00");
            TestFormatSpecifier(tp, tz, "%Ez", "-12:00");

            tz = fixed_time_zone(chrono::hours(12) + chrono::minutes(34));
            TestFormatSpecifier(tp, tz, "%z", "+1234");
            TestFormatSpecifier(tp, tz, "%:z", "+12:34");
            TestFormatSpecifier(tp, tz, "%Ez", "+12:34");

            tz = fixed_time_zone(-chrono::hours(12) - chrono::minutes(34));
            TestFormatSpecifier(tp, tz, "%z", "-1234");
            TestFormatSpecifier(tp, tz, "%:z", "-12:34");
            TestFormatSpecifier(tp, tz, "%Ez", "-12:34");

            tz = fixed_time_zone(chrono::hours(12) + chrono::minutes(34) +
                                 chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%z", "+1234");
            TestFormatSpecifier(tp, tz, "%:z", "+12:34");
            TestFormatSpecifier(tp, tz, "%Ez", "+12:34");

            tz = fixed_time_zone(-chrono::hours(12) - chrono::minutes(34) -
                                 chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%z", "-1234");
            TestFormatSpecifier(tp, tz, "%:z", "-12:34");
            TestFormatSpecifier(tp, tz, "%Ez", "-12:34");
        }

        TEST_CASE("format, ExtendedSecondOffset") {
            const auto tp = chrono::system_clock::from_time_t(0);

            auto tz = fixed_time_zone(turbo::time_internal::cctz::seconds::zero());
            TestFormatSpecifier(tp, tz, "%E*z", "+00:00:00");
            TestFormatSpecifier(tp, tz, "%::z", "+00:00:00");
            TestFormatSpecifier(tp, tz, "%:::z", "+00");

            tz = fixed_time_zone(chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%E*z", "+00:00:56");
            TestFormatSpecifier(tp, tz, "%::z", "+00:00:56");
            TestFormatSpecifier(tp, tz, "%:::z", "+00:00:56");

            tz = fixed_time_zone(-chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%E*z", "-00:00:56");
            TestFormatSpecifier(tp, tz, "%::z", "-00:00:56");
            TestFormatSpecifier(tp, tz, "%:::z", "-00:00:56");

            tz = fixed_time_zone(chrono::minutes(34));
            TestFormatSpecifier(tp, tz, "%E*z", "+00:34:00");
            TestFormatSpecifier(tp, tz, "%::z", "+00:34:00");
            TestFormatSpecifier(tp, tz, "%:::z", "+00:34");

            tz = fixed_time_zone(-chrono::minutes(34));
            TestFormatSpecifier(tp, tz, "%E*z", "-00:34:00");
            TestFormatSpecifier(tp, tz, "%::z", "-00:34:00");
            TestFormatSpecifier(tp, tz, "%:::z", "-00:34");

            tz = fixed_time_zone(chrono::minutes(34) + chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%E*z", "+00:34:56");
            TestFormatSpecifier(tp, tz, "%::z", "+00:34:56");
            TestFormatSpecifier(tp, tz, "%:::z", "+00:34:56");

            tz = fixed_time_zone(-chrono::minutes(34) - chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%E*z", "-00:34:56");
            TestFormatSpecifier(tp, tz, "%::z", "-00:34:56");
            TestFormatSpecifier(tp, tz, "%:::z", "-00:34:56");

            tz = fixed_time_zone(chrono::hours(12));
            TestFormatSpecifier(tp, tz, "%E*z", "+12:00:00");
            TestFormatSpecifier(tp, tz, "%::z", "+12:00:00");
            TestFormatSpecifier(tp, tz, "%:::z", "+12");

            tz = fixed_time_zone(-chrono::hours(12));
            TestFormatSpecifier(tp, tz, "%E*z", "-12:00:00");
            TestFormatSpecifier(tp, tz, "%::z", "-12:00:00");
            TestFormatSpecifier(tp, tz, "%:::z", "-12");

            tz = fixed_time_zone(chrono::hours(12) + chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%E*z", "+12:00:56");
            TestFormatSpecifier(tp, tz, "%::z", "+12:00:56");
            TestFormatSpecifier(tp, tz, "%:::z", "+12:00:56");

            tz = fixed_time_zone(-chrono::hours(12) - chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%E*z", "-12:00:56");
            TestFormatSpecifier(tp, tz, "%::z", "-12:00:56");
            TestFormatSpecifier(tp, tz, "%:::z", "-12:00:56");

            tz = fixed_time_zone(chrono::hours(12) + chrono::minutes(34));
            TestFormatSpecifier(tp, tz, "%E*z", "+12:34:00");
            TestFormatSpecifier(tp, tz, "%::z", "+12:34:00");
            TestFormatSpecifier(tp, tz, "%:::z", "+12:34");

            tz = fixed_time_zone(-chrono::hours(12) - chrono::minutes(34));
            TestFormatSpecifier(tp, tz, "%E*z", "-12:34:00");
            TestFormatSpecifier(tp, tz, "%::z", "-12:34:00");
            TestFormatSpecifier(tp, tz, "%:::z", "-12:34");

            tz = fixed_time_zone(chrono::hours(12) + chrono::minutes(34) +
                                 chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%E*z", "+12:34:56");
            TestFormatSpecifier(tp, tz, "%::z", "+12:34:56");
            TestFormatSpecifier(tp, tz, "%:::z", "+12:34:56");

            tz = fixed_time_zone(-chrono::hours(12) - chrono::minutes(34) -
                                 chrono::seconds(56));
            TestFormatSpecifier(tp, tz, "%E*z", "-12:34:56");
            TestFormatSpecifier(tp, tz, "%::z", "-12:34:56");
            TestFormatSpecifier(tp, tz, "%:::z", "-12:34:56");
        }

        TEST_CASE("format, ExtendedYears") {
            const time_zone utc = utc_time_zone();
            const char e4y_fmt[] = "%E4Y%m%d";  // no separators

            // %E4Y zero-pads the year to produce at least 4 chars, including the sign.
            auto tp = convert(civil_second(-999, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("-9991127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(-99, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("-0991127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(-9, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("-0091127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(-1, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("-0011127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(0, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("00001127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(1, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("00011127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(9, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("00091127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(99, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("00991127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(999, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("09991127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(9999, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("99991127", format(e4y_fmt, tp, utc));

            // When the year is outside [-999:9999], more than 4 chars are produced.
            tp = convert(civil_second(-1000, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("-10001127", format(e4y_fmt, tp, utc));
            tp = convert(civil_second(10000, 11, 27, 0, 0, 0), utc);
            REQUIRE_EQ("100001127", format(e4y_fmt, tp, utc));
        }

        TEST_CASE("format, RFC3339Format") {
            time_zone tz;
            REQUIRE(load_time_zone("America/Los_Angeles", &tz));

            time_point<chrono::nanoseconds> tp =
                    convert(civil_second(1977, 6, 28, 9, 8, 7), tz);
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::milliseconds(100);
            REQUIRE_EQ("1977-06-28T09:08:07.1-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::milliseconds(20);
            REQUIRE_EQ("1977-06-28T09:08:07.12-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::milliseconds(3);
            REQUIRE_EQ("1977-06-28T09:08:07.123-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::microseconds(400);
            REQUIRE_EQ("1977-06-28T09:08:07.1234-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::microseconds(50);
            REQUIRE_EQ("1977-06-28T09:08:07.12345-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::microseconds(6);
            REQUIRE_EQ("1977-06-28T09:08:07.123456-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::nanoseconds(700);
            REQUIRE_EQ("1977-06-28T09:08:07.1234567-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::nanoseconds(80);
            REQUIRE_EQ("1977-06-28T09:08:07.12345678-07:00", format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));

            tp += chrono::nanoseconds(9);
            REQUIRE_EQ("1977-06-28T09:08:07.123456789-07:00",
                       format(RFC3339_full, tp, tz));
            REQUIRE_EQ("1977-06-28T09:08:07-07:00", format(RFC3339_sec, tp, tz));
        }

        TEST_CASE("format, RFC1123Format") {  // locale specific
            time_zone tz;
            REQUIRE(load_time_zone("America/Los_Angeles", &tz));

            auto tp = convert(civil_second(1977, 6, 28, 9, 8, 7), tz);
            REQUIRE_EQ("Tue, 28 Jun 1977 09:08:07 -0700", format(RFC1123_full, tp, tz));
            REQUIRE_EQ("28 Jun 1977 09:08:07 -0700", format(RFC1123_no_wday, tp, tz));
        }

        TEST_CASE("format, Week") {
            const time_zone utc = utc_time_zone();

            auto tp = convert(civil_second(2017, 1, 1, 0, 0, 0), utc);
            REQUIRE_EQ("2017-01-7", format("%Y-%U-%u", tp, utc));
            REQUIRE_EQ("2017-00-0", format("%Y-%W-%w", tp, utc));

            tp = convert(civil_second(2017, 12, 31, 0, 0, 0), utc);
            REQUIRE_EQ("2017-53-7", format("%Y-%U-%u", tp, utc));
            REQUIRE_EQ("2017-52-0", format("%Y-%W-%w", tp, utc));

            tp = convert(civil_second(2018, 1, 1, 0, 0, 0), utc);
            REQUIRE_EQ("2018-00-1", format("%Y-%U-%u", tp, utc));
            REQUIRE_EQ("2018-01-1", format("%Y-%W-%w", tp, utc));

            tp = convert(civil_second(2018, 12, 31, 0, 0, 0), utc);
            REQUIRE_EQ("2018-52-1", format("%Y-%U-%u", tp, utc));
            REQUIRE_EQ("2018-53-1", format("%Y-%W-%w", tp, utc));

            tp = convert(civil_second(2019, 1, 1, 0, 0, 0), utc);
            REQUIRE_EQ("2019-00-2", format("%Y-%U-%u", tp, utc));
            REQUIRE_EQ("2019-00-2", format("%Y-%W-%w", tp, utc));

            tp = convert(civil_second(2019, 12, 31, 0, 0, 0), utc);
            REQUIRE_EQ("2019-52-2", format("%Y-%U-%u", tp, utc));
            REQUIRE_EQ("2019-52-2", format("%Y-%W-%w", tp, utc));
        }

//
// Testing parse()
//

        TEST_CASE("Parse, TimePointResolution") {
            const char kFmt[] = "%H:%M:%E*S";
            const time_zone utc = utc_time_zone();

            time_point<chrono::nanoseconds> tp_ns;
            REQUIRE(parse(kFmt, "03:04:05.123456789", utc, &tp_ns));
            REQUIRE_EQ("03:04:05.123456789", format(kFmt, tp_ns, utc));
            REQUIRE(parse(kFmt, "03:04:05.123456", utc, &tp_ns));
            REQUIRE_EQ("03:04:05.123456", format(kFmt, tp_ns, utc));

            time_point<chrono::microseconds> tp_us;
            REQUIRE(parse(kFmt, "03:04:05.123456789", utc, &tp_us));
            REQUIRE_EQ("03:04:05.123456", format(kFmt, tp_us, utc));
            REQUIRE(parse(kFmt, "03:04:05.123456", utc, &tp_us));
            REQUIRE_EQ("03:04:05.123456", format(kFmt, tp_us, utc));
            REQUIRE(parse(kFmt, "03:04:05.123", utc, &tp_us));
            REQUIRE_EQ("03:04:05.123", format(kFmt, tp_us, utc));

            time_point<chrono::milliseconds> tp_ms;
            REQUIRE(parse(kFmt, "03:04:05.123456", utc, &tp_ms));
            REQUIRE_EQ("03:04:05.123", format(kFmt, tp_ms, utc));
            REQUIRE(parse(kFmt, "03:04:05.123", utc, &tp_ms));
            REQUIRE_EQ("03:04:05.123", format(kFmt, tp_ms, utc));
            REQUIRE(parse(kFmt, "03:04:05", utc, &tp_ms));
            REQUIRE_EQ("03:04:05", format(kFmt, tp_ms, utc));

            time_point<chrono::seconds> tp_s;
            REQUIRE(parse(kFmt, "03:04:05.123", utc, &tp_s));
            REQUIRE_EQ("03:04:05", format(kFmt, tp_s, utc));
            REQUIRE(parse(kFmt, "03:04:05", utc, &tp_s));
            REQUIRE_EQ("03:04:05", format(kFmt, tp_s, utc));

            time_point<chrono::minutes> tp_m;
            REQUIRE(parse(kFmt, "03:04:05", utc, &tp_m));
            REQUIRE_EQ("03:04:00", format(kFmt, tp_m, utc));

            time_point<chrono::hours> tp_h;
            REQUIRE(parse(kFmt, "03:04:05", utc, &tp_h));
            REQUIRE_EQ("03:00:00", format(kFmt, tp_h, utc));
        }

        TEST_CASE("Parse, TimePointExtendedResolution") {
            const char kFmt[] = "%H:%M:%E*S";
            const time_zone utc = utc_time_zone();

            time_point<turbo::time_internal::cctz::seconds> tp;
            detail::femtoseconds fs;
            REQUIRE(detail::parse(kFmt, "12:34:56.123456789012345", utc, &tp, &fs));
            REQUIRE_EQ("12:34:56.123456789012345", detail::format(kFmt, tp, fs, utc));
            REQUIRE(detail::parse(kFmt, "12:34:56.012345678901234", utc, &tp, &fs));
            REQUIRE_EQ("12:34:56.012345678901234", detail::format(kFmt, tp, fs, utc));
            REQUIRE(detail::parse(kFmt, "12:34:56.001234567890123", utc, &tp, &fs));
            REQUIRE_EQ("12:34:56.001234567890123", detail::format(kFmt, tp, fs, utc));
            REQUIRE(detail::parse(kFmt, "12:34:56.000000000000123", utc, &tp, &fs));
            REQUIRE_EQ("12:34:56.000000000000123", detail::format(kFmt, tp, fs, utc));
            REQUIRE(detail::parse(kFmt, "12:34:56.000000000000012", utc, &tp, &fs));
            REQUIRE_EQ("12:34:56.000000000000012", detail::format(kFmt, tp, fs, utc));
            REQUIRE(detail::parse(kFmt, "12:34:56.000000000000001", utc, &tp, &fs));
            REQUIRE_EQ("12:34:56.000000000000001", detail::format(kFmt, tp, fs, utc));
        }

        TEST_CASE("Parse, Basics") {
            time_zone tz = utc_time_zone();
            time_point<chrono::nanoseconds> tp =
                    chrono::system_clock::from_time_t(1234567890);

            // Simple edge cases.
            REQUIRE(parse("", "", tz, &tp));
            REQUIRE_EQ(chrono::system_clock::from_time_t(0), tp);  // everything defaulted
            REQUIRE(parse(" ", " ", tz, &tp));
            REQUIRE(parse("  ", "  ", tz, &tp));
            REQUIRE(parse("x", "x", tz, &tp));
            REQUIRE(parse("xxx", "xxx", tz, &tp));

            REQUIRE(
                    parse("%Y-%m-%d %H:%M:%S %z", "2013-06-28 19:08:09 -0800", tz, &tp));
            ExpectTime(tp, tz, 2013, 6, 29, 3, 8, 9, 0, false, "UTC");
        }

        TEST_CASE("Parse, WithTimeZone") {
            time_zone tz;
            REQUIRE(load_time_zone("America/Los_Angeles", &tz));
            time_point<chrono::nanoseconds> tp;

            // We can parse a string without a UTC offset if we supply a timezone.
            REQUIRE(parse("%Y-%m-%d %H:%M:%S", "2013-06-28 19:08:09", tz, &tp));
            ExpectTime(tp, tz, 2013, 6, 28, 19, 8, 9, -7 * 60 * 60, true, "PDT");

            // But the timezone is ignored when a UTC offset is present.
            REQUIRE(parse("%Y-%m-%d %H:%M:%S %z", "2013-06-28 19:08:09 +0800",
                          utc_time_zone(), &tp));
            ExpectTime(tp, tz, 2013, 6, 28, 19 - 8 - 7, 8, 9, -7 * 60 * 60, true, "PDT");

            // Check a skipped time (a Spring DST transition). parse() uses the
            // pre-transition offset.
            REQUIRE(parse("%Y-%m-%d %H:%M:%S", "2011-03-13 02:15:00", tz, &tp));
            ExpectTime(tp, tz, 2011, 3, 13, 3, 15, 0, -7 * 60 * 60, true, "PDT");

            // Check a repeated time (a Fall DST transition).  parse() uses the
            // pre-transition offset.
            REQUIRE(parse("%Y-%m-%d %H:%M:%S", "2011-11-06 01:15:00", tz, &tp));
            ExpectTime(tp, tz, 2011, 11, 6, 1, 15, 0, -7 * 60 * 60, true, "PDT");
        }

        TEST_CASE("Parse, LeapSecond") {
            time_zone tz;
            REQUIRE(load_time_zone("America/Los_Angeles", &tz));
            time_point<chrono::nanoseconds> tp;

            // ":59" -> ":59"
            REQUIRE(parse(RFC3339_full, "2013-06-28T07:08:59-08:00", tz, &tp));
            ExpectTime(tp, tz, 2013, 6, 28, 8, 8, 59, -7 * 60 * 60, true, "PDT");

            // ":59.5" -> ":59.5"
            REQUIRE(parse(RFC3339_full, "2013-06-28T07:08:59.5-08:00", tz, &tp));
            ExpectTime(tp, tz, 2013, 6, 28, 8, 8, 59, -7 * 60 * 60, true, "PDT");

            // ":60" -> ":00"
            REQUIRE(parse(RFC3339_full, "2013-06-28T07:08:60-08:00", tz, &tp));
            ExpectTime(tp, tz, 2013, 6, 28, 8, 9, 0, -7 * 60 * 60, true, "PDT");

            // ":60.5" -> ":00.0"
            REQUIRE(parse(RFC3339_full, "2013-06-28T07:08:60.5-08:00", tz, &tp));
            ExpectTime(tp, tz, 2013, 6, 28, 8, 9, 0, -7 * 60 * 60, true, "PDT");

            // ":61" -> error
            REQUIRE_FALSE(parse(RFC3339_full, "2013-06-28T07:08:61-08:00", tz, &tp));
        }

        TEST_CASE("Parse, ErrorCases") {
            const time_zone tz = utc_time_zone();
            auto tp = chrono::system_clock::from_time_t(0);

            // Illegal trailing data.
            REQUIRE_FALSE(parse("%S", "123", tz, &tp));

            // Can't parse an illegal format specifier.
            REQUIRE_FALSE(parse("%Q", "x", tz, &tp));

            // Fails because of trailing, unparsed data "blah".
            REQUIRE_FALSE(parse("%m-%d", "2-3 blah", tz, &tp));

            // Trailing whitespace is allowed.
            REQUIRE(parse("%m-%d", "2-3  ", tz, &tp));
            REQUIRE_EQ(2, convert(tp, utc_time_zone()).month());
            REQUIRE_EQ(3, convert(tp, utc_time_zone()).day());

            // Feb 31 requires normalization.
            REQUIRE_FALSE(parse("%m-%d", "2-31", tz, &tp));

            // Check that we cannot have spaces in UTC offsets.
            REQUIRE(parse("%z", "-0203", tz, &tp));
            REQUIRE_FALSE(parse("%z", "- 2 3", tz, &tp));
            REQUIRE(parse("%Ez", "-02:03", tz, &tp));
            REQUIRE_FALSE(parse("%Ez", "- 2: 3", tz, &tp));

            // Check that we reject other malformed UTC offsets.
            REQUIRE_FALSE(parse("%Ez", "+-08:00", tz, &tp));
            REQUIRE_FALSE(parse("%Ez", "-+08:00", tz, &tp));

            // Check that we do not accept "-0" in fields that allow zero.
            REQUIRE_FALSE(parse("%Y", "-0", tz, &tp));
            REQUIRE_FALSE(parse("%E4Y", "-0", tz, &tp));
            REQUIRE_FALSE(parse("%H", "-0", tz, &tp));
            REQUIRE_FALSE(parse("%M", "-0", tz, &tp));
            REQUIRE_FALSE(parse("%S", "-0", tz, &tp));
            REQUIRE_FALSE(parse("%z", "+-000", tz, &tp));
            REQUIRE_FALSE(parse("%Ez", "+-0:00", tz, &tp));
            REQUIRE_FALSE(parse("%z", "-00-0", tz, &tp));
            REQUIRE_FALSE(parse("%Ez", "-00:-0", tz, &tp));
        }

        TEST_CASE("Parse, PosixConversions") {
            time_zone tz = utc_time_zone();
            auto tp = chrono::system_clock::from_time_t(0);
            const auto reset = convert(civil_second(1977, 6, 28, 9, 8, 7), tz);

            tp = reset;
            REQUIRE(parse("%d", "15", tz, &tp));
            REQUIRE_EQ(15, convert(tp, tz).day());

            // %e is an extension, but is supported internally.
            tp = reset;
            REQUIRE(parse("%e", "15", tz, &tp));
            REQUIRE_EQ(15, convert(tp, tz).day());  // Equivalent to %d

            tp = reset;
            REQUIRE(parse("%H", "17", tz, &tp));
            REQUIRE_EQ(17, convert(tp, tz).hour());

            tp = reset;
            REQUIRE(parse("%I", "5", tz, &tp));
            REQUIRE_EQ(5, convert(tp, tz).hour());

            // %j is parsed but ignored.
            REQUIRE(parse("%j", "32", tz, &tp));

            tp = reset;
            REQUIRE(parse("%m", "11", tz, &tp));
            REQUIRE_EQ(11, convert(tp, tz).month());

            tp = reset;
            REQUIRE(parse("%M", "33", tz, &tp));
            REQUIRE_EQ(33, convert(tp, tz).minute());

            tp = reset;
            REQUIRE(parse("%S", "55", tz, &tp));
            REQUIRE_EQ(55, convert(tp, tz).second());

            // %U is parsed but ignored.
            REQUIRE(parse("%U", "15", tz, &tp));

            // %w is parsed but ignored.
            REQUIRE(parse("%w", "2", tz, &tp));

            // %W is parsed but ignored.
            REQUIRE(parse("%W", "22", tz, &tp));

            tp = reset;
            REQUIRE(parse("%y", "04", tz, &tp));
            REQUIRE_EQ(2004, convert(tp, tz).year());

            tp = reset;
            REQUIRE(parse("%Y", "2004", tz, &tp));
            REQUIRE_EQ(2004, convert(tp, tz).year());

            REQUIRE(parse("%%", "%", tz, &tp));

#if defined(__linux__)
            // SU/C99/TZ extensions

            // Because we handle each (non-internal) specifier in a separate call
            // to strptime(), there is no way to group %C and %y together.  So we
            // just skip the %C/%y case.
#if 0
            tp = reset;
            REQUIRE(parse("%C %y", "20 04", tz, &tp));
            REQUIRE_EQ(2004, convert(tp, tz).year());
#endif

            tp = reset;
            REQUIRE(parse("%D", "02/03/04", tz, &tp));
            REQUIRE_EQ(2, convert(tp, tz).month());
            REQUIRE_EQ(3, convert(tp, tz).day());
            REQUIRE_EQ(2004, convert(tp, tz).year());

            REQUIRE(parse("%n", "\n", tz, &tp));

            tp = reset;
            REQUIRE(parse("%R", "03:44", tz, &tp));
            REQUIRE_EQ(3, convert(tp, tz).hour());
            REQUIRE_EQ(44, convert(tp, tz).minute());

            REQUIRE(parse("%t", "\t\v\f\n\r ", tz, &tp));

            tp = reset;
            REQUIRE(parse("%T", "03:44:55", tz, &tp));
            REQUIRE_EQ(3, convert(tp, tz).hour());
            REQUIRE_EQ(44, convert(tp, tz).minute());
            REQUIRE_EQ(55, convert(tp, tz).second());

            tp = reset;
            REQUIRE(parse("%s", "1234567890", tz, &tp));
            REQUIRE_EQ(chrono::system_clock::from_time_t(1234567890), tp);

            // %s conversion, like %z/%Ez, pays no heed to the optional zone.
            time_zone lax;
            REQUIRE(load_time_zone("America/Los_Angeles", &lax));
            tp = reset;
            REQUIRE(parse("%s", "1234567890", lax, &tp));
            REQUIRE_EQ(chrono::system_clock::from_time_t(1234567890), tp);

            // This is most important when the time has the same YMDhms
            // breakdown in the zone as some other time.  For example, ...
            //  1414917000 in US/Pacific -> Sun Nov 2 01:30:00 2014 (PDT)
            //  1414920600 in US/Pacific -> Sun Nov 2 01:30:00 2014 (PST)
            tp = reset;
            REQUIRE(parse("%s", "1414917000", lax, &tp));
            REQUIRE_EQ(chrono::system_clock::from_time_t(1414917000), tp);
            tp = reset;
            REQUIRE(parse("%s", "1414920600", lax, &tp));
            REQUIRE_EQ(chrono::system_clock::from_time_t(1414920600), tp);
#endif
        }

        TEST_CASE("Parse, LocaleSpecific") {
            time_zone tz = utc_time_zone();
            auto tp = chrono::system_clock::from_time_t(0);
            const auto reset = convert(civil_second(1977, 6, 28, 9, 8, 7), tz);

            // %a is parsed but ignored.
            REQUIRE(parse("%a", "Mon", tz, &tp));

            // %A is parsed but ignored.
            REQUIRE(parse("%A", "Monday", tz, &tp));

            tp = reset;
            REQUIRE(parse("%b", "Feb", tz, &tp));
            REQUIRE_EQ(2, convert(tp, tz).month());

            tp = reset;
            REQUIRE(parse("%B", "February", tz, &tp));
            REQUIRE_EQ(2, convert(tp, tz).month());

            // %p is parsed but ignored if it's alone.  But it's used with %I.
            REQUIRE(parse("%p", "AM", tz, &tp));
            tp = reset;
            REQUIRE(parse("%I %p", "5 PM", tz, &tp));
            REQUIRE_EQ(17, convert(tp, tz).hour());

            tp = reset;
            REQUIRE(parse("%x", "02/03/04", tz, &tp));
            if (convert(tp, tz).month() == 2) {
                REQUIRE_EQ(3, convert(tp, tz).day());
            } else {
                REQUIRE_EQ(2, convert(tp, tz).day());
                REQUIRE_EQ(3, convert(tp, tz).month());
            }
            REQUIRE_EQ(2004, convert(tp, tz).year());

            tp = reset;
            REQUIRE(parse("%X", "15:44:55", tz, &tp));
            REQUIRE_EQ(15, convert(tp, tz).hour());
            REQUIRE_EQ(44, convert(tp, tz).minute());
            REQUIRE_EQ(55, convert(tp, tz).second());

#if defined(__linux__)
            // SU/C99/TZ extensions

            tp = reset;
            REQUIRE(parse("%h", "Feb", tz, &tp));
            REQUIRE_EQ(2, convert(tp, tz).month());  // Equivalent to %b

#if defined(__GLIBC__)
            tp = reset;
            REQUIRE(parse("%l %p", "5 PM", tz, &tp));
            REQUIRE_EQ(17, convert(tp, tz).hour());
#endif

            tp = reset;
            REQUIRE(parse("%r", "03:44:55 PM", tz, &tp));
            REQUIRE_EQ(15, convert(tp, tz).hour());
            REQUIRE_EQ(44, convert(tp, tz).minute());
            REQUIRE_EQ(55, convert(tp, tz).second());

#if defined(__GLIBC__)
            tp = reset;
            REQUIRE(parse("%Ec", "Tue Nov 19 05:06:07 2013", tz, &tp));
            REQUIRE_EQ(convert(civil_second(2013, 11, 19, 5, 6, 7), tz), tp);

            // Modified conversion specifiers %E_

            tp = reset;
            REQUIRE(parse("%Ex", "02/03/04", tz, &tp));
            REQUIRE_EQ(2, convert(tp, tz).month());
            REQUIRE_EQ(3, convert(tp, tz).day());
            REQUIRE_EQ(2004, convert(tp, tz).year());

            tp = reset;
            REQUIRE(parse("%EX", "15:44:55", tz, &tp));
            REQUIRE_EQ(15, convert(tp, tz).hour());
            REQUIRE_EQ(44, convert(tp, tz).minute());
            REQUIRE_EQ(55, convert(tp, tz).second());

            // %Ey, the year offset from %EC, doesn't really make sense alone as there
            // is no way to represent it in tm_year (%EC is not simply the century).
            // Yet, because we handle each (non-internal) specifier in a separate call
            // to strptime(), there is no way to group %EC and %Ey either.  So we just
            // skip the %EC and %Ey cases.

            tp = reset;
            REQUIRE(parse("%EY", "2004", tz, &tp));
            REQUIRE_EQ(2004, convert(tp, tz).year());

            // Modified conversion specifiers %O_

            tp = reset;
            REQUIRE(parse("%Od", "15", tz, &tp));
            REQUIRE_EQ(15, convert(tp, tz).day());

            tp = reset;
            REQUIRE(parse("%Oe", "15", tz, &tp));
            REQUIRE_EQ(15, convert(tp, tz).day());  // Equivalent to %d

            tp = reset;
            REQUIRE(parse("%OH", "17", tz, &tp));
            REQUIRE_EQ(17, convert(tp, tz).hour());

            tp = reset;
            REQUIRE(parse("%OI", "5", tz, &tp));
            REQUIRE_EQ(5, convert(tp, tz).hour());

            tp = reset;
            REQUIRE(parse("%Om", "11", tz, &tp));
            REQUIRE_EQ(11, convert(tp, tz).month());

            tp = reset;
            REQUIRE(parse("%OM", "33", tz, &tp));
            REQUIRE_EQ(33, convert(tp, tz).minute());

            tp = reset;
            REQUIRE(parse("%OS", "55", tz, &tp));
            REQUIRE_EQ(55, convert(tp, tz).second());

            // %OU is parsed but ignored.
            REQUIRE(parse("%OU", "15", tz, &tp));

            // %Ow is parsed but ignored.
            REQUIRE(parse("%Ow", "2", tz, &tp));

            // %OW is parsed but ignored.
            REQUIRE(parse("%OW", "22", tz, &tp));

            tp = reset;
            REQUIRE(parse("%Oy", "04", tz, &tp));
            REQUIRE_EQ(2004, convert(tp, tz).year());
#endif
#endif
        }

        TEST_CASE("Parse, ExtendedSeconds") {
            const time_zone tz = utc_time_zone();
            const time_point<chrono::nanoseconds> unix_epoch =
                    chrono::system_clock::from_time_t(0);

            // All %E<prec>S cases are treated the same as %E*S on input.
            auto precisions = {"*", "0", "1", "2", "3", "4", "5", "6", "7",
                               "8", "9", "10", "11", "12", "13", "14", "15"};
            for (const std::string prec: precisions) {
                const std::string fmt = "%E" + prec + "S";
                time_point<chrono::nanoseconds> tp = unix_epoch;
                REQUIRE(parse(fmt, "5", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05.0", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05.00", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05.6", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5) + chrono::milliseconds(600), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05.60", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5) + chrono::milliseconds(600), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05.600", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5) + chrono::milliseconds(600), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05.67", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5) + chrono::milliseconds(670), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05.670", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5) + chrono::milliseconds(670), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "05.678", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::seconds(5) + chrono::milliseconds(678), tp);
            }

            // Here is a "%E*S" case we got wrong for a while.  The fractional
            // part of the first instant is less than 2^31 and was correctly
            // parsed, while the second (and any subsecond field >=2^31) failed.
            time_point<chrono::nanoseconds> tp = unix_epoch;
            REQUIRE(parse("%E*S", "0.2147483647", tz, &tp));
            REQUIRE_EQ(unix_epoch + chrono::nanoseconds(214748364), tp);
            tp = unix_epoch;
            REQUIRE(parse("%E*S", "0.2147483648", tz, &tp));
            REQUIRE_EQ(unix_epoch + chrono::nanoseconds(214748364), tp);

            // We should also be able to specify long strings of digits far
            // beyond the current resolution and have them convert the same way.
            tp = unix_epoch;
            REQUIRE(parse(
                    "%E*S", "0.214748364801234567890123456789012345678901234567890123456789",
                    tz, &tp));
            REQUIRE_EQ(unix_epoch + chrono::nanoseconds(214748364), tp);
        }

        TEST_CASE("Parse, ExtendedSecondsScan") {
            const time_zone tz = utc_time_zone();
            time_point<chrono::nanoseconds> tp;
            for (int ms = 0; ms < 1000; ms += 111) {
                for (int us = 0; us < 1000; us += 27) {
                    const int micros = ms * 1000 + us;
                    for (int ns = 0; ns < 1000; ns += 9) {
                        const auto expected = chrono::system_clock::from_time_t(0) +
                                              chrono::nanoseconds(micros * 1000 + ns);
                        std::ostringstream oss;
                        oss << "0." << std::setfill('0') << std::setw(3);
                        oss << ms << std::setw(3) << us << std::setw(3) << ns;
                        const std::string input = oss.str();
                        REQUIRE(parse("%E*S", input, tz, &tp));
                        REQUIRE_EQ(expected, tp);
                    }
                }
            }
        }

        TEST_CASE("Parse, ExtendedSubeconds") {
            const time_zone tz = utc_time_zone();
            const time_point<chrono::nanoseconds> unix_epoch =
                    chrono::system_clock::from_time_t(0);

            // All %E<prec>f cases are treated the same as %E*f on input.
            auto precisions = {"*", "0", "1", "2", "3", "4", "5", "6", "7",
                               "8", "9", "10", "11", "12", "13", "14", "15"};
            for (const std::string prec: precisions) {
                const std::string fmt = "%E" + prec + "f";
                time_point<chrono::nanoseconds> tp = unix_epoch - chrono::seconds(1);
                REQUIRE(parse(fmt, "", tz, &tp));
                REQUIRE_EQ(unix_epoch, tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "6", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::milliseconds(600), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "60", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::milliseconds(600), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "600", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::milliseconds(600), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "67", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::milliseconds(670), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "670", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::milliseconds(670), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "678", tz, &tp));
                REQUIRE_EQ(unix_epoch + chrono::milliseconds(678), tp);
                tp = unix_epoch;
                REQUIRE(parse(fmt, "6789", tz, &tp));
                REQUIRE_EQ(
                        unix_epoch + chrono::milliseconds(678) + chrono::microseconds(900), tp);
            }

            // Here is a "%E*f" case we got wrong for a while.  The fractional
            // part of the first instant is less than 2^31 and was correctly
            // parsed, while the second (and any subsecond field >=2^31) failed.
            time_point<chrono::nanoseconds> tp = unix_epoch;
            REQUIRE(parse("%E*f", "2147483647", tz, &tp));
            REQUIRE_EQ(unix_epoch + chrono::nanoseconds(214748364), tp);
            tp = unix_epoch;
            REQUIRE(parse("%E*f", "2147483648", tz, &tp));
            REQUIRE_EQ(unix_epoch + chrono::nanoseconds(214748364), tp);

            // We should also be able to specify long strings of digits far
            // beyond the current resolution and have them convert the same way.
            tp = unix_epoch;
            REQUIRE(parse(
                    "%E*f", "214748364801234567890123456789012345678901234567890123456789",
                    tz, &tp));
            REQUIRE_EQ(unix_epoch + chrono::nanoseconds(214748364), tp);
        }

        TEST_CASE("Parse, ExtendedSubecondsScan") {
            time_point<chrono::nanoseconds> tp;
            const time_zone tz = utc_time_zone();
            for (int ms = 0; ms < 1000; ms += 111) {
                for (int us = 0; us < 1000; us += 27) {
                    const int micros = ms * 1000 + us;
                    for (int ns = 0; ns < 1000; ns += 9) {
                        std::ostringstream oss;
                        oss << std::setfill('0') << std::setw(3) << ms;
                        oss << std::setw(3) << us << std::setw(3) << ns;
                        const std::string nanos = oss.str();
                        const auto expected = chrono::system_clock::from_time_t(0) +
                                              chrono::nanoseconds(micros * 1000 + ns);
                        for (int ps = 0; ps < 1000; ps += 250) {
                            std::ostringstream ps_oss;
                            oss << std::setfill('0') << std::setw(3) << ps;
                            const std::string input = nanos + ps_oss.str() + "999";
                            REQUIRE(parse("%E*f", input, tz, &tp));
                            REQUIRE_EQ(expected + chrono::nanoseconds(ps) / 1000, tp);
                        }
                    }
                }
            }
        }

        TEST_CASE("Parse, ExtendedOffset") {
            const time_zone utc = utc_time_zone();
            time_point<turbo::time_internal::cctz::seconds> tp;

            REQUIRE(parse("%Ez", "+00:00", utc, &tp));
            REQUIRE_EQ(convert(civil_second(1970, 1, 1, 0, 0, 0), utc), tp);
            REQUIRE(parse("%Ez", "-12:34", utc, &tp));
            REQUIRE_EQ(convert(civil_second(1970, 1, 1, 12, 34, 0), utc), tp);
            REQUIRE(parse("%Ez", "+12:34", utc, &tp));
            REQUIRE_EQ(convert(civil_second(1969, 12, 31, 11, 26, 0), utc), tp);
            REQUIRE_FALSE(parse("%Ez", "-12:3", utc, &tp));

            for (auto fmt: {"%Ez", "%z"}) {
                REQUIRE(parse(fmt, "+0000", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 0, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "-1234", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 12, 34, 0), utc), tp);
                REQUIRE(parse(fmt, "+1234", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1969, 12, 31, 11, 26, 0), utc), tp);
                REQUIRE_FALSE(parse(fmt, "-123", utc, &tp));

                REQUIRE(parse(fmt, "+00", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 0, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "-12", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 12, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "+12", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1969, 12, 31, 12, 0, 0), utc), tp);
                REQUIRE_FALSE(parse(fmt, "-1", utc, &tp));
            }
        }

        TEST_CASE("Parse, ExtendedSecondOffset") {
            const time_zone utc = utc_time_zone();
            time_point<turbo::time_internal::cctz::seconds> tp;

            for (auto fmt: {"%Ez", "%E*z", "%:z", "%::z", "%:::z"}) {
                REQUIRE(parse(fmt, "+00:00:00", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 0, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "-12:34:56", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 12, 34, 56), utc), tp);
                REQUIRE(parse(fmt, "+12:34:56", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1969, 12, 31, 11, 25, 4), utc), tp);
                REQUIRE_FALSE(parse(fmt, "-12:34:5", utc, &tp));

                REQUIRE(parse(fmt, "+000000", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 0, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "-123456", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 12, 34, 56), utc), tp);
                REQUIRE(parse(fmt, "+123456", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1969, 12, 31, 11, 25, 4), utc), tp);
                REQUIRE_FALSE(parse(fmt, "-12345", utc, &tp));

                REQUIRE(parse(fmt, "+00:00", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 0, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "-12:34", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 12, 34, 0), utc), tp);
                REQUIRE(parse(fmt, "+12:34", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1969, 12, 31, 11, 26, 0), utc), tp);
                REQUIRE_FALSE(parse(fmt, "-12:3", utc, &tp));

                REQUIRE(parse(fmt, "+0000", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 0, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "-1234", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 12, 34, 0), utc), tp);
                REQUIRE(parse(fmt, "+1234", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1969, 12, 31, 11, 26, 0), utc), tp);
                REQUIRE_FALSE(parse(fmt, "-123", utc, &tp));

                REQUIRE(parse(fmt, "+00", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 0, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "-12", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1970, 1, 1, 12, 0, 0), utc), tp);
                REQUIRE(parse(fmt, "+12", utc, &tp));
                REQUIRE_EQ(convert(civil_second(1969, 12, 31, 12, 0, 0), utc), tp);
                REQUIRE_FALSE(parse(fmt, "-1", utc, &tp));
            }
        }

        TEST_CASE("Parse, ExtendedYears") {
            const time_zone utc = utc_time_zone();
            const char e4y_fmt[] = "%E4Y%m%d";  // no separators
            time_point<turbo::time_internal::cctz::seconds> tp;

            // %E4Y consumes exactly four chars, including any sign.
            REQUIRE(parse(e4y_fmt, "-9991127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(-999, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "-0991127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(-99, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "-0091127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(-9, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "-0011127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(-1, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "00001127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(0, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "00011127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(1, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "00091127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(9, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "00991127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(99, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "09991127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(999, 11, 27, 0, 0, 0), utc), tp);
            REQUIRE(parse(e4y_fmt, "99991127", utc, &tp));
            REQUIRE_EQ(convert(civil_second(9999, 11, 27, 0, 0, 0), utc), tp);

            // When the year is outside [-999:9999], the parse fails.
            REQUIRE_FALSE(parse(e4y_fmt, "-10001127", utc, &tp));
            REQUIRE_FALSE(parse(e4y_fmt, "100001127", utc, &tp));
        }

        TEST_CASE("Parse, RFC3339Format") {
            const time_zone tz = utc_time_zone();
            time_point<chrono::nanoseconds> tp;
            REQUIRE(parse(RFC3339_sec, "2014-02-12T20:21:00+00:00", tz, &tp));
            ExpectTime(tp, tz, 2014, 2, 12, 20, 21, 0, 0, false, "UTC");

            // Check that %ET also accepts "t".
            time_point<chrono::nanoseconds> tp2;
            REQUIRE(parse(RFC3339_sec, "2014-02-12t20:21:00+00:00", tz, &tp2));
            REQUIRE_EQ(tp, tp2);

            // Check that %Ez also accepts "Z" as a synonym for "+00:00".
            time_point<chrono::nanoseconds> tp3;
            REQUIRE(parse(RFC3339_sec, "2014-02-12T20:21:00Z", tz, &tp3));
            REQUIRE_EQ(tp, tp3);

            // Check that %Ez also accepts "z" as a synonym for "+00:00".
            time_point<chrono::nanoseconds> tp4;
            REQUIRE(parse(RFC3339_sec, "2014-02-12T20:21:00z", tz, &tp4));
            REQUIRE_EQ(tp, tp4);
        }

        TEST_CASE("Parse, Week") {
            const time_zone utc = utc_time_zone();
            time_point<turbo::time_internal::cctz::seconds> tp;

            auto exp = convert(civil_second(2017, 1, 1, 0, 0, 0), utc);
            REQUIRE(parse("%Y-%U-%u", "2017-01-7", utc, &tp));
            REQUIRE_EQ(exp, tp);
            REQUIRE(parse("%Y-%W-%w", "2017-00-0", utc, &tp));
            REQUIRE_EQ(exp, tp);

            exp = convert(civil_second(2017, 12, 31, 0, 0, 0), utc);
            REQUIRE(parse("%Y-%U-%u", "2017-53-7", utc, &tp));
            REQUIRE_EQ(exp, tp);
            REQUIRE(parse("%Y-%W-%w", "2017-52-0", utc, &tp));
            REQUIRE_EQ(exp, tp);

            exp = convert(civil_second(2018, 1, 1, 0, 0, 0), utc);
            REQUIRE(parse("%Y-%U-%u", "2018-00-1", utc, &tp));
            REQUIRE_EQ(exp, tp);
            REQUIRE(parse("%Y-%W-%w", "2018-01-1", utc, &tp));
            REQUIRE_EQ(exp, tp);

            exp = convert(civil_second(2018, 12, 31, 0, 0, 0), utc);
            REQUIRE(parse("%Y-%U-%u", "2018-52-1", utc, &tp));
            REQUIRE_EQ(exp, tp);
            REQUIRE(parse("%Y-%W-%w", "2018-53-1", utc, &tp));
            REQUIRE_EQ(exp, tp);

            exp = convert(civil_second(2019, 1, 1, 0, 0, 0), utc);
            REQUIRE(parse("%Y-%U-%u", "2019-00-2", utc, &tp));
            REQUIRE_EQ(exp, tp);
            REQUIRE(parse("%Y-%W-%w", "2019-00-2", utc, &tp));
            REQUIRE_EQ(exp, tp);

            exp = convert(civil_second(2019, 12, 31, 0, 0, 0), utc);
            REQUIRE(parse("%Y-%U-%u", "2019-52-2", utc, &tp));
            REQUIRE_EQ(exp, tp);
            REQUIRE(parse("%Y-%W-%w", "2019-52-2", utc, &tp));
            REQUIRE_EQ(exp, tp);
        }

        TEST_CASE("Parse, WeekYearShift") {
            // %U/%W conversions with week values in {0, 52, 53} can slip
            // into the previous/following calendar years.
            const time_zone utc = utc_time_zone();
            time_point<turbo::time_internal::cctz::seconds> tp;

            auto exp = convert(civil_second(2019, 12, 31, 0, 0, 0), utc);
            REQUIRE(parse("%Y-%U-%u", "2020-00-2", utc, &tp));
            REQUIRE_EQ(exp, tp);
            REQUIRE(parse("%Y-%W-%w", "2020-00-2", utc, &tp));
            REQUIRE_EQ(exp, tp);

            exp = convert(civil_second(2021, 1, 1, 0, 0, 0), utc);
            REQUIRE(parse("%Y-%U-%u", "2020-52-5", utc, &tp));
            REQUIRE_EQ(exp, tp);
            REQUIRE(parse("%Y-%W-%w", "2020-52-5", utc, &tp));
            REQUIRE_EQ(exp, tp);

            // Slipping into the previous/following calendar years should fail when
            // we're already at the extremes.
            REQUIRE_FALSE(parse("%Y-%U-%u", "-9223372036854775808-0-7", utc, &tp));
            REQUIRE_FALSE(parse("%Y-%U-%u", "9223372036854775807-53-7", utc, &tp));
        }

        TEST_CASE("Parse, MaxRange") {
            const time_zone utc = utc_time_zone();
            time_point<turbo::time_internal::cctz::seconds> tp;

            // tests the upper limit using +00:00 offset
            REQUIRE(
                    parse(RFC3339_sec, "292277026596-12-04T15:30:07+00:00", utc, &tp));
            REQUIRE_EQ(tp, time_point<turbo::time_internal::cctz::seconds>::max());
            REQUIRE_FALSE(
                    parse(RFC3339_sec, "292277026596-12-04T15:30:08+00:00", utc, &tp));

            // tests the upper limit using -01:00 offset
            REQUIRE(
                    parse(RFC3339_sec, "292277026596-12-04T14:30:07-01:00", utc, &tp));
            REQUIRE_EQ(tp, time_point<turbo::time_internal::cctz::seconds>::max());
            REQUIRE_FALSE(
                    parse(RFC3339_sec, "292277026596-12-04T14:30:08-01:00", utc, &tp));

            // tests the lower limit using +00:00 offset
            REQUIRE(
                    parse(RFC3339_sec, "-292277022657-01-27T08:29:52+00:00", utc, &tp));
            REQUIRE_EQ(tp, time_point<turbo::time_internal::cctz::seconds>::min());
            REQUIRE_FALSE(
                    parse(RFC3339_sec, "-292277022657-01-27T08:29:51+00:00", utc, &tp));

            // tests the lower limit using +01:00 offset
            REQUIRE(
                    parse(RFC3339_sec, "-292277022657-01-27T09:29:52+01:00", utc, &tp));
            REQUIRE_EQ(tp, time_point<turbo::time_internal::cctz::seconds>::min());
            REQUIRE_FALSE(
                    parse(RFC3339_sec, "-292277022657-01-27T08:29:51+01:00", utc, &tp));

            // tests max/min civil-second overflow
            REQUIRE_FALSE(
                    parse(RFC3339_sec, "9223372036854775807-12-31T23:59:59-00:01", utc, &tp));
            REQUIRE_FALSE(parse(RFC3339_sec, "-9223372036854775808-01-01T00:00:00+00:01",
                                utc, &tp));
        }

        TEST_CASE("Parse, TimePointOverflow") {
            const time_zone utc = utc_time_zone();

            using D = chrono::duration<std::int64_t, std::nano>;
            time_point<D> tp;

            REQUIRE(
                    parse(RFC3339_full, "2262-04-11T23:47:16.8547758079+00:00", utc, &tp));
            REQUIRE_EQ(tp, time_point<D>::max());
            REQUIRE_EQ("2262-04-11T23:47:16.854775807+00:00",
                       format(RFC3339_full, tp, utc));
#if 0
            // TODO(#199): Will fail until cctz::parse() properly detects overflow.
            REQUIRE_FALSE(
                parse(RFC3339_full, "2262-04-11T23:47:16.8547758080+00:00", utc, &tp));
            REQUIRE(
                parse(RFC3339_full, "1677-09-21T00:12:43.1452241920+00:00", utc, &tp));
            REQUIRE_EQ(tp, time_point<D>::min());
            REQUIRE_EQ("1677-09-21T00:12:43.145224192+00:00",
                      format(RFC3339_full, tp, utc));
            REQUIRE_FALSE(
                parse(RFC3339_full, "1677-09-21T00:12:43.1452241919+00:00", utc, &tp));
#endif

            using DS = chrono::duration<std::int8_t, chrono::seconds::period>;
            time_point<DS> stp;

            REQUIRE(parse(RFC3339_full, "1970-01-01T00:02:07.9+00:00", utc, &stp));
            REQUIRE_EQ(stp, time_point<DS>::max());
            REQUIRE_EQ("1970-01-01T00:02:07+00:00", format(RFC3339_full, stp, utc));
            REQUIRE_FALSE(parse(RFC3339_full, "1970-01-01T00:02:08+00:00", utc, &stp));

            REQUIRE(parse(RFC3339_full, "1969-12-31T23:57:52+00:00", utc, &stp));
            REQUIRE_EQ(stp, time_point<DS>::min());
            REQUIRE_EQ("1969-12-31T23:57:52+00:00", format(RFC3339_full, stp, utc));
            REQUIRE_FALSE(parse(RFC3339_full, "1969-12-31T23:57:51.9+00:00", utc, &stp));

            using DM = chrono::duration<std::int8_t, chrono::minutes::period>;
            time_point<DM> mtp;

            REQUIRE(parse(RFC3339_full, "1970-01-01T02:07:59+00:00", utc, &mtp));
            REQUIRE_EQ(mtp, time_point<DM>::max());
            REQUIRE_EQ("1970-01-01T02:07:00+00:00", format(RFC3339_full, mtp, utc));
            REQUIRE_FALSE(parse(RFC3339_full, "1970-01-01T02:08:00+00:00", utc, &mtp));

            REQUIRE(parse(RFC3339_full, "1969-12-31T21:52:00+00:00", utc, &mtp));
            REQUIRE_EQ(mtp, time_point<DM>::min());
            REQUIRE_EQ("1969-12-31T21:52:00+00:00", format(RFC3339_full, mtp, utc));
            REQUIRE_FALSE(parse(RFC3339_full, "1969-12-31T21:51:59+00:00", utc, &mtp));
        }

        TEST_CASE("Parse, TimePointOverflowFloor") {
            const time_zone utc = utc_time_zone();

            using D = chrono::duration<std::int64_t, std::micro>;
            time_point<D> tp;

            REQUIRE(
                    parse(RFC3339_full, "294247-01-10T04:00:54.7758079+00:00", utc, &tp));
            REQUIRE_EQ(tp, time_point<D>::max());
            REQUIRE_EQ("294247-01-10T04:00:54.775807+00:00",
                       format(RFC3339_full, tp, utc));
#if 0
            // TODO(#199): Will fail until cctz::parse() properly detects overflow.
            REQUIRE_FALSE(
                parse(RFC3339_full, "294247-01-10T04:00:54.7758080+00:00", utc, &tp));
            REQUIRE(
                parse(RFC3339_full, "-290308-12-21T19:59:05.2241920+00:00", utc, &tp));
            REQUIRE_EQ(tp, time_point<D>::min());
            REQUIRE_EQ("-290308-12-21T19:59:05.224192+00:00",
                      format(RFC3339_full, tp, utc));
            REQUIRE_FALSE(
                parse(RFC3339_full, "-290308-12-21T19:59:05.2241919+00:00", utc, &tp));
#endif
        }

//
// Roundtrip test for format()/parse().
//

        TEST_CASE("FormatParse, RoundTrip") {
            time_zone lax;
            REQUIRE(load_time_zone("America/Los_Angeles", &lax));
            const auto in = convert(civil_second(1977, 6, 28, 9, 8, 7), lax);
            const auto subseconds = chrono::nanoseconds(654321);

            // RFC3339, which renders subseconds.
            {
                time_point<chrono::nanoseconds> out;
                const std::string s = format(RFC3339_full, in + subseconds, lax);
                REQUIRE(parse(RFC3339_full, s, lax, &out));
                REQUIRE_EQ(in + subseconds, out);  // RFC3339_full includes %Ez
            }

            // RFC1123, which only does whole seconds.
            {
                time_point<chrono::nanoseconds> out;
                const std::string s = format(RFC1123_full, in, lax);
                REQUIRE(parse(RFC1123_full, s, lax, &out));
                REQUIRE_EQ(in, out);  // RFC1123_full includes %z
            }

#if defined(_WIN32) || defined(_WIN64)
            // Initial investigations indicate the %c does not roundtrip on Windows.
            // TODO: Figure out what is going on here (perhaps a locale problem).
#elif defined(__EMSCRIPTEN__)
            // strftime() and strptime() use different defintions for "%c" under
            // emscripten (see https://github.com/kripken/emscripten/pull/7491),
            // causing its round-trip test to fail.
#else
            // Even though we don't know what %c will produce, it should roundtrip,
            // but only in the 0-offset timezone.
            {
                time_point<chrono::nanoseconds> out;
                time_zone utc = utc_time_zone();
                const std::string s = format("%c", in, utc);
                REQUIRE(parse("%c", s, utc, &out));
                REQUIRE_EQ(in, out);
            }
#endif
        }

        TEST_CASE("FormatParse, RoundTripDistantFuture") {
            const time_zone utc = utc_time_zone();
            const time_point<turbo::time_internal::cctz::seconds> in =
                    time_point<turbo::time_internal::cctz::seconds>::max();
            const std::string s = format(RFC3339_full, in, utc);
            time_point<turbo::time_internal::cctz::seconds> out;
            REQUIRE(parse(RFC3339_full, s, utc, &out));
            REQUIRE_EQ(in, out);
        }

        TEST_CASE("FormatParse, RoundTripDistantPast") {
            const time_zone utc = utc_time_zone();
            const time_point<turbo::time_internal::cctz::seconds> in =
                    time_point<turbo::time_internal::cctz::seconds>::min();
            const std::string s = format(RFC3339_full, in, utc);
            time_point<turbo::time_internal::cctz::seconds> out;
            REQUIRE(parse(RFC3339_full, s, utc, &out));
            REQUIRE_EQ(in, out);
        }

    }  // namespace cctz
}  // namespace turbo::time_internal
