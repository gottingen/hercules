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

#include "turbo/strings/match.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

namespace {

    TEST_CASE("MatchTest, starts_with") {
        const std::string s1("123\0abc", 7);
        const std::string_view a("foobar");
        const std::string_view b(s1);
        const std::string_view e;
        CHECK(turbo::starts_with(a, a));
        CHECK(turbo::starts_with(a, "foo"));
        CHECK(turbo::starts_with(a, e));
        CHECK(turbo::starts_with(b, s1));
        CHECK(turbo::starts_with(b, b));
        CHECK(turbo::starts_with(b, e));
        CHECK(turbo::starts_with(e, ""));
        CHECK_FALSE(turbo::starts_with(a, b));
        CHECK_FALSE(turbo::starts_with(b, a));
        CHECK_FALSE(turbo::starts_with(e, a));
    }

    TEST_CASE("MatchTest, ends_with") {
        const std::string s1("123\0abc", 7);
        const std::string_view a("foobar");
        const std::string_view b(s1);
        const std::string_view e;
        CHECK(turbo::ends_with(a, a));
        CHECK(turbo::ends_with(a, "bar"));
        CHECK(turbo::ends_with(a, e));
        CHECK(turbo::ends_with(b, s1));
        CHECK(turbo::ends_with(b, b));
        CHECK(turbo::ends_with(b, e));
        CHECK(turbo::ends_with(e, ""));
        CHECK_FALSE(turbo::ends_with(a, b));
        CHECK_FALSE(turbo::ends_with(b, a));
        CHECK_FALSE(turbo::ends_with(e, a));
    }

    TEST_CASE("MatchTest, Contains") {
        std::string_view a("abcdefg");
        std::string_view b("abcd");
        std::string_view c("efg");
        std::string_view d("gh");
        CHECK(turbo::str_contains(a, a));
        CHECK(turbo::str_contains(a, b));
        CHECK(turbo::str_contains(a, c));
        CHECK_FALSE(turbo::str_contains(a, d));
        CHECK(turbo::str_contains("", ""));
        CHECK(turbo::str_contains("abc", ""));
        CHECK_FALSE(turbo::str_contains("", "a"));
    }

    TEST_CASE("MatchTest, ContainsChar") {
        std::string_view a("abcdefg");
        std::string_view b("abcd");
        CHECK(turbo::str_contains(a, 'a'));
        CHECK(turbo::str_contains(a, 'b'));
        CHECK(turbo::str_contains(a, 'e'));
        CHECK_FALSE(turbo::str_contains(a, 'h'));

        CHECK(turbo::str_contains(b, 'a'));
        CHECK(turbo::str_contains(b, 'b'));
        CHECK_FALSE(turbo::str_contains(b, 'e'));
        CHECK_FALSE(turbo::str_contains(b, 'h'));

        CHECK_FALSE(turbo::str_contains("", 'a'));
        CHECK_FALSE(turbo::str_contains("", 'a'));
    }

    TEST_CASE("MatchTest, ContainsIgnoreCaseChar") {
        std::string_view a("abcdefg");
        std::string_view b("ABCD");
        CHECK(turbo::str_ignore_case_contains(a, 'a'));
        CHECK(turbo::str_ignore_case_contains(a, 'A'));
        CHECK(turbo::str_ignore_case_contains(a, 'b'));
        CHECK(turbo::str_ignore_case_contains(a, 'B'));
        CHECK(turbo::str_ignore_case_contains(a, 'e'));
        CHECK(turbo::str_ignore_case_contains(a, 'E'));
        CHECK_FALSE(turbo::str_ignore_case_contains(a, 'h'));

        CHECK(turbo::str_ignore_case_contains(b, 'a'));
        CHECK(turbo::str_ignore_case_contains(b, 'A'));
        CHECK(turbo::str_ignore_case_contains(b, 'b'));
        CHECK(turbo::str_ignore_case_contains(b, 'B'));
        CHECK_FALSE(turbo::str_ignore_case_contains(b, 'e'));
        CHECK_FALSE(turbo::str_ignore_case_contains(b, 'E'));
        CHECK_FALSE(turbo::str_ignore_case_contains(b, 'h'));
        CHECK_FALSE(turbo::str_ignore_case_contains(b, 'H'));

        CHECK_FALSE(turbo::str_ignore_case_contains("", 'a'));
        CHECK_FALSE(turbo::str_ignore_case_contains("", 'A'));
        CHECK_FALSE(turbo::str_ignore_case_contains("", 'a'));
        CHECK_FALSE(turbo::str_ignore_case_contains("", 'A'));
    }

    TEST_CASE("MatchTest, ContainsNull") {
        const std::string s = "foo";
        const char *cs = "foo";
        const std::string_view sv("foo");
        const std::string_view sv2("foo\0bar", 4);
        CHECK_EQ(s, "foo");
        CHECK_EQ(sv, "foo");
        CHECK_NE(sv2, "foo");
        CHECK(turbo::ends_with(s, sv));
        CHECK(turbo::starts_with(cs, sv));
        CHECK(turbo::str_contains(cs, sv));
        CHECK_FALSE(turbo::str_contains(cs, sv2));
    }

    TEST_CASE("MatchTest, str_equals_ignore_case") {
        std::string text = "the";
        std::string_view data(text);

        CHECK(turbo::str_equals_ignore_case(data, "The"));
        CHECK(turbo::str_equals_ignore_case(data, "THE"));
        CHECK(turbo::str_equals_ignore_case(data, "the"));
        CHECK_FALSE(turbo::str_equals_ignore_case(data, "Quick"));
        CHECK_FALSE(turbo::str_equals_ignore_case(data, "then"));
    }

    TEST_CASE("MatchTest, starts_with_ignore_case") {
        CHECK(turbo::starts_with_ignore_case("foo", "foo"));
        CHECK(turbo::starts_with_ignore_case("foo", "Fo"));
        CHECK(turbo::starts_with_ignore_case("foo", ""));
        CHECK_FALSE(turbo::starts_with_ignore_case("foo", "fooo"));
        CHECK_FALSE(turbo::starts_with_ignore_case("", "fo"));
    }

    TEST_CASE("MatchTest, ends_with_ignore_case") {
        CHECK(turbo::ends_with_ignore_case("foo", "foo"));
        CHECK(turbo::ends_with_ignore_case("foo", "Oo"));
        CHECK(turbo::ends_with_ignore_case("foo", ""));
        CHECK_FALSE(turbo::ends_with_ignore_case("foo", "fooo"));
        CHECK_FALSE(turbo::ends_with_ignore_case("", "fo"));
    }

}  // namespace
