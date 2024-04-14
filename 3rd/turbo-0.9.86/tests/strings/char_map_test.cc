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

#include "turbo/strings/internal/char_map.h"

#include <cctype>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

namespace {

    constexpr turbo::strings_internal::Charmap everything_map =
            ~turbo::strings_internal::Charmap();
    constexpr turbo::strings_internal::Charmap nothing_map{};

    TEST_CASE("Charmap, AllTests") {
        const turbo::strings_internal::Charmap also_nothing_map("", 0);
        REQUIRE(everything_map.contains('\0'));
        REQUIRE(!nothing_map.contains('\0'));
        REQUIRE(!also_nothing_map.contains('\0'));
        for (unsigned char ch = 1; ch != 0; ++ch) {
            REQUIRE(everything_map.contains(ch));
            REQUIRE(!nothing_map.contains(ch));
            REQUIRE(!also_nothing_map.contains(ch));
        }

        const turbo::strings_internal::Charmap symbols("&@#@^!@?", 5);
        REQUIRE(symbols.contains('&'));
        REQUIRE(symbols.contains('@'));
        REQUIRE(symbols.contains('#'));
        REQUIRE(symbols.contains('^'));
        REQUIRE(!symbols.contains('!'));
        REQUIRE(!symbols.contains('?'));
        int cnt = 0;
        for (unsigned char ch = 1; ch != 0; ++ch)
            cnt += symbols.contains(ch);
        REQUIRE_EQ(cnt, 4);

        const turbo::strings_internal::Charmap lets("^abcde", 3);
        const turbo::strings_internal::Charmap lets2("fghij\0klmnop", 10);
        const turbo::strings_internal::Charmap lets3("fghij\0klmnop");
        REQUIRE(lets2.contains('k'));
        REQUIRE(!lets3.contains('k'));

        REQUIRE(symbols.IntersectsWith(lets));
        REQUIRE(!lets2.IntersectsWith(lets));
        REQUIRE(lets.IntersectsWith(symbols));
        REQUIRE(!lets.IntersectsWith(lets2));

        REQUIRE(nothing_map.IsZero());
        REQUIRE(!lets.IsZero());
    }

    namespace {
        std::string Members(const turbo::strings_internal::Charmap &m) {
            std::string r;
            for (size_t i = 0; i < 256; ++i)
                if (m.contains(i)) r.push_back(i);
            return r;
        }

        std::string ClosedRangeString(unsigned char lo, unsigned char hi) {
            // Don't depend on lo<hi. Just increment until lo==hi.
            std::string s;
            while (true) {
                s.push_back(lo);
                if (lo == hi) break;
                ++lo;
            }
            return s;
        }

    }  // namespace

    TEST_CASE("Charmap, Constexpr") {
        constexpr turbo::strings_internal::Charmap kEmpty = nothing_map;
        CHECK_EQ(Members(kEmpty), "");
        constexpr turbo::strings_internal::Charmap kA =
                turbo::strings_internal::Charmap::Char('A');
        CHECK_EQ(Members(kA), "A");
        constexpr turbo::strings_internal::Charmap kAZ =
                turbo::strings_internal::Charmap::Range('A', 'Z');
        CHECK_EQ(Members(kAZ), "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        constexpr turbo::strings_internal::Charmap kIdentifier =
                turbo::strings_internal::Charmap::Range('0', '9') |
                turbo::strings_internal::Charmap::Range('A', 'Z') |
                turbo::strings_internal::Charmap::Range('a', 'z') |
                turbo::strings_internal::Charmap::Char('_');
        CHECK_EQ(Members(kIdentifier),
                 "0123456789"
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "_"
                 "abcdefghijklmnopqrstuvwxyz");
        constexpr turbo::strings_internal::Charmap kAll = everything_map;
        for (size_t i = 0; i < 256; ++i) {
            CHECK(kAll.contains(i));
        }
        constexpr turbo::strings_internal::Charmap kHello =
                turbo::strings_internal::Charmap::FromString("Hello, world!");
        CHECK_EQ(Members(kHello), " !,Hdelorw");

        // test negation and intersection
        constexpr turbo::strings_internal::Charmap kABC =
                turbo::strings_internal::Charmap::Range('A', 'Z') &
                ~turbo::strings_internal::Charmap::Range('D', 'Z');
        CHECK_EQ(Members(kABC), "ABC");
    }

    TEST_CASE("Charmap, Range") {
        // Exhaustive testing takes too long, so test some of the boundaries that
        // are perhaps going to cause trouble.
        std::vector<size_t> poi = {0, 1, 2, 3, 4, 7, 8, 9, 15,
                                   16, 17, 30, 31, 32, 33, 63, 64, 65,
                                   127, 128, 129, 223, 224, 225, 254, 255};
        for (auto lo = poi.begin(); lo != poi.end(); ++lo) {
            CAPTURE(*lo);
            for (auto hi = lo; hi != poi.end(); ++hi) {
                CAPTURE(*hi);
                CHECK_EQ(Members(turbo::strings_internal::Charmap::Range(*lo, *hi)),
                            ClosedRangeString(*lo, *hi));
            }
        }
    }

    bool AsBool(int x) { return static_cast<bool>(x); }

    TEST_CASE("CharmapCtype, Match") {
        for (int c = 0; c < 256; ++c) {
            CAPTURE(c);
            CAPTURE(static_cast<char>(c));
            CHECK_EQ(AsBool(std::isupper(c)),
                     turbo::strings_internal::UpperCharmap().contains(c));
            CHECK_EQ(AsBool(std::islower(c)),
                     turbo::strings_internal::LowerCharmap().contains(c));
            CHECK_EQ(AsBool(std::isdigit(c)),
                     turbo::strings_internal::DigitCharmap().contains(c));
            CHECK_EQ(AsBool(std::isalpha(c)),
                     turbo::strings_internal::AlphaCharmap().contains(c));
            CHECK_EQ(AsBool(std::isalnum(c)),
                     turbo::strings_internal::AlnumCharmap().contains(c));
            CHECK_EQ(AsBool(std::isxdigit(c)),
                     turbo::strings_internal::XDigitCharmap().contains(c));
            CHECK_EQ(AsBool(std::isprint(c)),
                     turbo::strings_internal::PrintCharmap().contains(c));
            CHECK_EQ(AsBool(std::isspace(c)),
                     turbo::strings_internal::SpaceCharmap().contains(c));
            CHECK_EQ(AsBool(std::iscntrl(c)),
                     turbo::strings_internal::CntrlCharmap().contains(c));
            CHECK_EQ(AsBool(std::isblank(c)),
                     turbo::strings_internal::BlankCharmap().contains(c));
            CHECK_EQ(AsBool(std::isgraph(c)),
                     turbo::strings_internal::GraphCharmap().contains(c));
            CHECK_EQ(AsBool(std::ispunct(c)),
                     turbo::strings_internal::PunctCharmap().contains(c));
        }
    }

}  // namespace
