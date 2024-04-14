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

#include "turbo/times/cctz/time_zone.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "tests/times/test_util.h"
#include "turbo/times/time.h"

namespace cctz = turbo::time_internal::cctz;

namespace {

    TEST_CASE("TimeZone, ValueSemantics") {
        turbo::TimeZone tz;
        turbo::TimeZone tz2 = tz;  // Copy-construct
        REQUIRE_EQ(tz, tz2);
        tz2 = tz;  // Copy-assign
        REQUIRE_EQ(tz, tz2);
    }

    TEST_CASE("TimeZone, Equality") {
        turbo::TimeZone a, b;
        REQUIRE_EQ(a, b);
        REQUIRE_EQ(a.name(), b.name());

        turbo::TimeZone implicit_utc;
        turbo::TimeZone explicit_utc = turbo::utc_time_zone();
        REQUIRE_EQ(implicit_utc, explicit_utc);
        REQUIRE_EQ(implicit_utc.name(), explicit_utc.name());

        turbo::TimeZone la = turbo::time_internal::load_time_zone("America/Los_Angeles");
        turbo::TimeZone nyc = turbo::time_internal::load_time_zone("America/New_York");
        REQUIRE_NE(la, nyc);
    }

    TEST_CASE("TimeZone, CCTZConversion") {
        const cctz::time_zone cz = cctz::utc_time_zone();
        const turbo::TimeZone tz(cz);
        REQUIRE_EQ(cz, cctz::time_zone(tz));
    }

    TEST_CASE("TimeZone, DefaultTimeZones") {
        turbo::TimeZone tz;
        REQUIRE_EQ("UTC", turbo::TimeZone().name());
        REQUIRE_EQ("UTC", turbo::utc_time_zone().name());
    }

    TEST_CASE("TimeZone, fixed_time_zone") {
        const turbo::TimeZone tz = turbo::fixed_time_zone(123);
        const cctz::time_zone cz = cctz::fixed_time_zone(cctz::seconds(123));
        REQUIRE_EQ(tz, turbo::TimeZone(cz));
    }

    TEST_CASE("TimeZone, local_time_zone") {
        const turbo::TimeZone local_tz = turbo::local_time_zone();
        turbo::TimeZone tz = turbo::time_internal::load_time_zone("localtime");
        REQUIRE_EQ(tz, local_tz);
    }

    TEST_CASE("TimeZone, NamedTimeZones") {
        turbo::TimeZone nyc = turbo::time_internal::load_time_zone("America/New_York");
        REQUIRE_EQ("America/New_York", nyc.name());
        turbo::TimeZone syd = turbo::time_internal::load_time_zone("Australia/Sydney");
        REQUIRE_EQ("Australia/Sydney", syd.name());
        turbo::TimeZone fixed = turbo::fixed_time_zone((((3 * 60) + 25) * 60) + 45);
        REQUIRE_EQ("Fixed/UTC+03:25:45", fixed.name());
    }

    TEST_CASE("TimeZone, Failures") {
        turbo::TimeZone tz = turbo::time_internal::load_time_zone("America/Los_Angeles");
        REQUIRE_FALSE(load_time_zone("Invalid/TimeZone", &tz));
        REQUIRE_EQ(turbo::utc_time_zone(), tz);  // guaranteed fallback to UTC

        // Ensures that the load still fails on a subsequent attempt.
        tz = turbo::time_internal::load_time_zone("America/Los_Angeles");
        REQUIRE_FALSE(load_time_zone("Invalid/TimeZone", &tz));
        REQUIRE_EQ(turbo::utc_time_zone(), tz);  // guaranteed fallback to UTC

        // Loading an empty string timezone should fail.
        tz = turbo::time_internal::load_time_zone("America/Los_Angeles");
        REQUIRE_FALSE(load_time_zone("", &tz));
        REQUIRE_EQ(turbo::utc_time_zone(), tz);  // guaranteed fallback to UTC
    }

}  // namespace
