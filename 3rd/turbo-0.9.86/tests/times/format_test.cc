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

#include <cstdint>
#include <limits>
#include <string>

#include "turbo/testing/test.h"
#include "tests/times/test_util.h"
#include "turbo/times/time.h"
#include "turbo/strings/match.h"
namespace {

    // A helper that tests the given format specifier by itself, and with leading
    // and trailing characters.  For example: TestFormatSpecifier(t, "%a", "Thu").
    void TestFormatSpecifier(turbo::Time t, turbo::TimeZone tz,
                             const std::string &fmt, const std::string &ans) {
        REQUIRE_EQ(ans, t.to_string(fmt,  tz));
        REQUIRE_EQ("xxx " + ans, t.to_string("xxx " + fmt, tz));
        REQUIRE_EQ(ans + " yyy", t.to_string(fmt + " yyy",  tz));
        REQUIRE_EQ("xxx " + ans + " yyy",
                   t.to_string("xxx " + fmt + " yyy",  tz));
    }

    //
    // Testing format_time()
    //

    TEST_CASE("format_time, Basics") {
        turbo::TimeZone tz = turbo::utc_time_zone();
        turbo::Time t = turbo::Time::from_time_t(0);

        // Starts with a couple basic edge cases.
        REQUIRE_EQ("", t.to_string("", tz));
        REQUIRE_EQ(" ", t.to_string(" ", tz));
        REQUIRE_EQ("  ", t.to_string("  ",  tz));
        REQUIRE_EQ("xxx", t.to_string("xxx", tz));
        std::string big(128, 'x');
        REQUIRE_EQ(big, t.to_string(big,  tz));
        // Cause the 1024-byte buffer to grow.
        std::string bigger(100000, 'x');
        REQUIRE_EQ(bigger, t.to_string(bigger,  tz));

        t += turbo::Duration::hours(13) + turbo::Duration::minutes(4) + turbo::Duration::seconds(5);
        t += turbo::Duration::milliseconds(6) + turbo::Duration::microseconds(7) + turbo::Duration::nanoseconds(8);
        REQUIRE_EQ("1970-01-01", t.to_string("%Y-%m-%d", tz));
        REQUIRE_EQ("13:04:05", t.to_string("%H:%M:%S",  tz));
        REQUIRE_EQ("13:04:05.006", t.to_string("%H:%M:%E3S",  tz));
        REQUIRE_EQ("13:04:05.006007", t.to_string("%H:%M:%E6S",  tz));
        REQUIRE_EQ("13:04:05.006007008", t.to_string("%H:%M:%E9S",  tz));
    }

    TEST_CASE("format_time, LocaleSpecific") {
        const turbo::TimeZone tz = turbo::utc_time_zone();
        turbo::Time t = turbo::Time::from_time_t(0);

        TestFormatSpecifier(t, tz, "%a", "Thu");
        TestFormatSpecifier(t, tz, "%A", "Thursday");
        TestFormatSpecifier(t, tz, "%b", "Jan");
        TestFormatSpecifier(t, tz, "%B", "January");

        // %c should at least produce the numeric year and time-of-day.
        const std::string s =
        turbo::Time::from_time_t(0).to_string("%c",  turbo::utc_time_zone());
        REQUIRE(turbo::str_contains(s, "1970"));
        REQUIRE(turbo::str_contains(s, "00:00:00"));

        TestFormatSpecifier(t, tz, "%p", "AM");
        TestFormatSpecifier(t, tz, "%x", "01/01/70");
        TestFormatSpecifier(t, tz, "%X", "00:00:00");
    }

    TEST_CASE("format_time, ExtendedSeconds") {
        const turbo::TimeZone tz = turbo::utc_time_zone();

        // No subseconds.
        turbo::Time t = turbo::Time::from_time_t(0) + turbo::Duration::seconds(5);
        REQUIRE_EQ("05", t.to_string("%E*S",  tz));
        REQUIRE_EQ("05.000000000000000", t.to_string("%E15S",  tz));

        // With subseconds.
        t += turbo::Duration::milliseconds(6) + turbo::Duration::microseconds(7) + turbo::Duration::nanoseconds(8);
        REQUIRE_EQ("05.006007008", t.to_string("%E*S",  tz));
        REQUIRE_EQ("05", t.to_string("%E0S", tz));
        REQUIRE_EQ("05.006007008000000", t.to_string("%E15S",  tz));

        // Times before the Unix epoch.
        t = turbo::Time::from_microseconds(-1);
        REQUIRE_EQ("1969-12-31 23:59:59.999999",
                  t.to_string("%Y-%m-%d %H:%M:%E*S",  tz));

        // Here is a "%E*S" case we got wrong for a while.  While the first
        // instant below is correctly rendered as "...:07.333304", the second
        // one used to appear as "...:07.33330499999999999".
        t = turbo::Time::from_microseconds(1395024427333304);
        REQUIRE_EQ("2014-03-17 02:47:07.333304",
                  t.to_string("%Y-%m-%d %H:%M:%E*S",  tz));
        t += turbo::Duration::microseconds(1);
        REQUIRE_EQ("2014-03-17 02:47:07.333305",
                  t.to_string("%Y-%m-%d %H:%M:%E*S",  tz));
    }

    TEST_CASE("format_time, RFC1123FormatPadsYear") {  // locale specific
        turbo::TimeZone tz = turbo::utc_time_zone();

        // A year of 77 should be padded to 0077.
        turbo::Time t = turbo::Time::from_civil(turbo::CivilSecond(77, 6, 28, 9, 8, 7), tz);
        REQUIRE_EQ("Mon, 28 Jun 0077 09:08:07 +0000",
                  t.to_string(turbo::RFC1123_full,  tz));
        REQUIRE_EQ("28 Jun 0077 09:08:07 +0000",
                  t.to_string(turbo::RFC1123_no_wday,  tz));
    }

    TEST_CASE("format_time, InfiniteTime") {
        turbo::TimeZone tz = turbo::time_internal::load_time_zone("America/Los_Angeles");

        // The format and timezone are ignored.
        REQUIRE_EQ("infinite-future",
                   turbo::Time::infinite_future().to_string("%H:%M blah", tz));
        REQUIRE_EQ("infinite-past",
                   turbo::Time::infinite_past().to_string("%H:%M blah", tz));
    }

    //
    // Testing parse_time()
    //

    TEST_CASE("parse_time, Basics") {
        turbo::Time t = turbo::Time::from_time_t(1234567890);
        std::string err;

        // Simple edge cases.
        REQUIRE(t.parse_time("", "",  &err)) ;
        REQUIRE_EQ(turbo::Time::unix_epoch(), t);  // everything defaulted
        REQUIRE(t.parse_time(" ", " ",  &err)) ;
        REQUIRE(t.parse_time("  ", "  ",  &err)) ;
        REQUIRE(t.parse_time("x", "x",  &err)) ;
        REQUIRE(t.parse_time("xxx", "xxx",  &err)) ;

        REQUIRE(t.parse_time("%Y-%m-%d %H:%M:%S %z",
                                     "2013-06-28 19:08:09 -0800",  &err));
        const auto ci = turbo::fixed_time_zone(-8 * 60 * 60).at(t);
        REQUIRE_EQ(turbo::CivilSecond(2013, 6, 28, 19, 8, 9), ci.cs);
        REQUIRE_EQ(turbo::Duration::zero(), ci.subsecond);
    }

    TEST_CASE("parse_time, NullErrorString") {
        turbo::Time t;
        REQUIRE_FALSE(t.parse_time("%Q", "invalid format",  nullptr));
        REQUIRE_FALSE(t.parse_time("%H", "12 trailing data",  nullptr));
        REQUIRE_FALSE(
                t.parse_time("%H out of range", "42 out of range",  nullptr));
    }

    TEST_CASE("parse_time, WithTimeZone") {
        const turbo::TimeZone tz =
                turbo::time_internal::load_time_zone("America/Los_Angeles");
        turbo::Time t;
        std::string e;

        // We can parse a string without a UTC offset if we supply a timezone.
        REQUIRE(
                t.parse_time("%Y-%m-%d %H:%M:%S", "2013-06-28 19:08:09", tz, &e))
                            ;
        auto ci = tz.at(t);
        REQUIRE_EQ(turbo::CivilSecond(2013, 6, 28, 19, 8, 9), ci.cs);
        REQUIRE_EQ(turbo::Duration::zero(), ci.subsecond);

        // But the timezone is ignored when a UTC offset is present.
        REQUIRE(t.parse_time("%Y-%m-%d %H:%M:%S %z",
                                     "2013-06-28 19:08:09 +0800", tz, &e))
                            ;
        ci = turbo::fixed_time_zone(8 * 60 * 60).at(t);
        REQUIRE_EQ(turbo::CivilSecond(2013, 6, 28, 19, 8, 9), ci.cs);
        REQUIRE_EQ(turbo::Duration::zero(), ci.subsecond);
    }

    TEST_CASE("parse_time, ErrorCases") {
        turbo::Time t = turbo::Time::from_time_t(0);
        std::string err;

        REQUIRE_FALSE(t.parse_time("%S", "123",  &err)) ;
        REQUIRE(turbo::str_contains(err, "Illegal trailing data"));

        // Can't parse an illegal format specifier.
        err.clear();
        REQUIRE_FALSE(t.parse_time("%Q", "x",  &err)) ;
        // Exact contents of "err" are platform-dependent because of
        // differences in the strptime implementation between macOS and Linux.
        REQUIRE_FALSE(err.empty());

        // Fails because of trailing, unparsed data "blah".
        REQUIRE_FALSE(t.parse_time("%m-%d", "2-3 blah",  &err)) ;
        REQUIRE(turbo::str_contains(err, "Illegal trailing data"));

        // Feb 31 requires normalization.
        REQUIRE_FALSE(t.parse_time("%m-%d", "2-31",  &err)) ;
        REQUIRE(turbo::str_contains(err, "Out-of-range"));

        // Check that we cannot have spaces in UTC offsets.
        REQUIRE(t.parse_time("%z", "-0203",  &err)) ;
        REQUIRE_FALSE(t.parse_time("%z", "- 2 3",  &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE(t.parse_time("%Ez", "-02:03",  &err)) ;
        REQUIRE_FALSE(t.parse_time("%Ez", "- 2: 3",  &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));

        // Check that we reject other malformed UTC offsets.
        REQUIRE_FALSE(t.parse_time("%Ez", "+-08:00", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE_FALSE(t.parse_time("%Ez", "-+08:00", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));

        // Check that we do not accept "-0" in fields that allow zero.
        REQUIRE_FALSE(t.parse_time("%Y", "-0", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE_FALSE(t.parse_time("%E4Y", "-0", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE_FALSE(t.parse_time("%H", "-0", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE_FALSE(t.parse_time("%M", "-0", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE_FALSE(t.parse_time("%S", "-0", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE_FALSE(t.parse_time("%z", "+-000", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE_FALSE(t.parse_time("%Ez", "+-0:00", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
        REQUIRE_FALSE(t.parse_time("%z", "-00-0", &err)) ;
        REQUIRE(turbo::str_contains(err, "Illegal trailing data"));
        REQUIRE_FALSE(t.parse_time("%Ez", "-00:-0", &err)) ;
        REQUIRE(turbo::str_contains(err, "Illegal trailing data"));
    }

    TEST_CASE("parse_time, ExtendedSeconds") {
        std::string err;
        turbo::Time t;

        // Here is a "%E*S" case we got wrong for a while.  The fractional
        // part of the first instant is less than 2^31 and was correctly
        // parsed, while the second (and any subsecond field >=2^31) failed.
        t = turbo::Time::unix_epoch();
        REQUIRE(t.parse_time("%E*S", "0.2147483647", &err)) ;
        REQUIRE_EQ(turbo::Time::unix_epoch() + turbo::Duration::nanoseconds(214748364) +
                  turbo::Duration::nanoseconds(1) / 2,
                  t);
        t = turbo::Time::unix_epoch();
        REQUIRE(t.parse_time("%E*S", "0.2147483648", &err)) ;
        REQUIRE_EQ(turbo::Time::unix_epoch() + turbo::Duration::nanoseconds(214748364) +
                  turbo::Duration::nanoseconds(3) / 4,
                  t);

        // We should also be able to specify long strings of digits far
        // beyond the current resolution and have them convert the same way.
        t = turbo::Time::unix_epoch();
        REQUIRE(t.parse_time(
                "%E*S", "0.214748364801234567890123456789012345678901234567890123456789",
                &err))
                            ;
        REQUIRE_EQ(turbo::Time::unix_epoch() + turbo::Duration::nanoseconds(214748364) +
                  turbo::Duration::nanoseconds(3) / 4,
                  t);
    }

    TEST_CASE("parse_time, ExtendedOffsetErrors") {
        std::string err;
        turbo::Time t;

        // %z against +-HHMM.
        REQUIRE_FALSE(t.parse_time("%z", "-123", &err)) ;
        REQUIRE(turbo::str_contains(err, "Illegal trailing data"));

        // %z against +-HH.
        REQUIRE_FALSE(t.parse_time("%z", "-1", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));

        // %Ez against +-HH:MM.
        REQUIRE_FALSE(t.parse_time("%Ez", "-12:3", &err)) ;
        REQUIRE(turbo::str_contains(err, "Illegal trailing data"));

        // %Ez against +-HHMM.
        REQUIRE_FALSE(t.parse_time("%Ez", "-123", &err)) ;
        REQUIRE(turbo::str_contains(err, "Illegal trailing data"));

        // %Ez against +-HH.
        REQUIRE_FALSE(t.parse_time("%Ez", "-1", &err)) ;
        REQUIRE(turbo::str_contains(err, "Failed to parse"));
    }

    TEST_CASE("parse_time, InfiniteTime") {
        turbo::Time t;
        std::string err;
        REQUIRE(t.parse_time("%H:%M blah", "infinite-future", &err));
        REQUIRE_EQ(turbo::Time::infinite_future(), t);

        // Surrounding whitespace.
        REQUIRE(t.parse_time("%H:%M blah", "  infinite-future", &err));
        REQUIRE_EQ(turbo::Time::infinite_future(), t);
        REQUIRE(t.parse_time("%H:%M blah", "infinite-future  ", &err));
        REQUIRE_EQ(turbo::Time::infinite_future(), t);
        REQUIRE(t.parse_time("%H:%M blah", "  infinite-future  ", &err));
        REQUIRE_EQ(turbo::Time::infinite_future(), t);

        REQUIRE(t.parse_time("%H:%M blah", "infinite-past", &err));
        REQUIRE_EQ(turbo::Time::infinite_past(), t);

        // Surrounding whitespace.
        REQUIRE(t.parse_time("%H:%M blah", "  infinite-past", &err));
        REQUIRE_EQ(turbo::Time::infinite_past(), t);
        REQUIRE(t.parse_time("%H:%M blah", "infinite-past  ", &err));
        REQUIRE_EQ(turbo::Time::infinite_past(), t);
        REQUIRE(t.parse_time("%H:%M blah", "  infinite-past  ", &err));
        REQUIRE_EQ(turbo::Time::infinite_past(), t);

        // "infinite-future" as literal string
        turbo::TimeZone tz = turbo::utc_time_zone();
        REQUIRE(t.parse_time("infinite-future %H:%M", "infinite-future 03:04",
                                     &err));
        REQUIRE_NE(turbo::Time::infinite_future(), t);
        REQUIRE_EQ(3, tz.at(t).cs.hour());
        REQUIRE_EQ(4, tz.at(t).cs.minute());

        // "infinite-past" as literal string
        REQUIRE(
                t.parse_time("infinite-past %H:%M", "infinite-past 03:04", &err));
        REQUIRE_NE(turbo::Time::infinite_past(), t);
        REQUIRE_EQ(3, tz.at(t).cs.hour());
        REQUIRE_EQ(4, tz.at(t).cs.minute());

        // The input doesn't match the format.
        REQUIRE_FALSE(t.parse_time("infinite-future %H:%M", "03:04", &err));
        REQUIRE_FALSE(t.parse_time("infinite-past %H:%M", "03:04", &err));
    }

    TEST_CASE("parse_time, FailsOnUnrepresentableTime") {
        const turbo::TimeZone utc = turbo::utc_time_zone();
        turbo::Time t;
        REQUIRE_FALSE(
                t.parse_time("%Y-%m-%d", "-292277022657-01-27", utc, nullptr));
        REQUIRE(
                t.parse_time("%Y-%m-%d", "-292277022657-01-28", utc, nullptr));
        REQUIRE(
                t.parse_time("%Y-%m-%d", "292277026596-12-04", utc, nullptr));
        REQUIRE_FALSE(
                t.parse_time("%Y-%m-%d", "292277026596-12-05", utc, nullptr));
    }

//
// Roundtrip test for format_time()/parse_time().
//

    TEST_CASE("FormatParse, RoundTrip") {
        const turbo::TimeZone lax =
                turbo::time_internal::load_time_zone("America/Los_Angeles");
        const turbo::Time in =
                turbo::Time::from_civil(turbo::CivilSecond(1977, 6, 28, 9, 8, 7), lax);
        const turbo::Duration subseconds = turbo::Duration::nanoseconds(654321);
        std::string err;

        // RFC3339, which renders subseconds.
        {
            turbo::Time out;
            const std::string s =
            (in + subseconds).to_string(turbo::RFC3339_full, lax);
            REQUIRE(out.parse_time(turbo::RFC3339_full, s,  &err));
            REQUIRE_EQ(in + subseconds, out);  // RFC3339_full includes %Ez
        }

        // RFC1123, which only does whole seconds.
        {
            turbo::Time out;
            const std::string s = in.to_string(turbo::RFC1123_full, lax);
            REQUIRE(out.parse_time(turbo::RFC1123_full, s, &err));
            REQUIRE_EQ(in, out);  // RFC1123_full includes %z
        }

        // `t.to_string()` falls back to strftime() for "%c", which appears to
        // work. On Windows, `t.parse_time()` falls back to std::get_time() which
        // appears to fail on "%c" (or at least on the "%c" text produced by
        // `strftime()`). This makes it fail the round-trip test.
        //
        // Under the emscripten compiler `t.parse_time() falls back to
        // `strptime()`, but that ends up using a different definition for "%c"
        // compared to `strftime()`, also causing the round-trip test to fail
        // (see https://github.com/kripken/emscripten/pull/7491).
#if !defined(_MSC_VER) && !defined(__EMSCRIPTEN__)
        // Even though we don't know what %c will produce, it should roundtrip,
        // but only in the 0-offset timezone.
        {
            turbo::Time out;
            const std::string s = in.to_string("%c", turbo::utc_time_zone());
            REQUIRE(out.parse_time("%c", s, &err));
            REQUIRE_EQ(in, out);
        }
#endif  // !_MSC_VER && !__EMSCRIPTEN__
    }

    TEST_CASE("FormatParse, RoundTripDistantFuture") {
        const turbo::TimeZone tz = turbo::utc_time_zone();
        const turbo::Time in =
                turbo::Time::from_seconds(std::numeric_limits<int64_t>::max());
        std::string err;

        turbo::Time out;
        const std::string s = in.to_string(turbo::RFC3339_full, tz);
        REQUIRE(out.parse_time(turbo::RFC3339_full, s, &err));
        REQUIRE_EQ(in, out);
    }

    TEST_CASE("FormatParse, RoundTripDistantPast") {
        const turbo::TimeZone tz = turbo::utc_time_zone();
        const turbo::Time in =
                turbo::Time::from_seconds(std::numeric_limits<int64_t>::min());
        std::string err;

        turbo::Time out;
        const std::string s = in.to_string(turbo::RFC3339_full, tz);
        REQUIRE(out.parse_time(turbo::RFC3339_full, s, &err));
        REQUIRE_EQ(in, out);
    }

}  // namespace
