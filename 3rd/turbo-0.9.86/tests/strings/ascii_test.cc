// Copyright 2022 The Turbo Authors.
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

#include "turbo/strings/ascii.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include <cctype>
#include <clocale>
#include <cstring>
#include <string>

#include "turbo/platform/port.h"
#include "turbo/strings/inlined_string.h"

namespace {

    TEST_CASE("AsciiIsFoo, All") {
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
                CHECK(turbo::ascii_is_alpha(c));
            else
                CHECK(!turbo::ascii_is_alpha(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if ((c >= '0' && c <= '9'))
                CHECK(turbo::ascii_is_digit(c));
            else
                CHECK(!turbo::ascii_is_digit(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (turbo::ascii_is_alpha(c) || turbo::ascii_is_digit(c))
                CHECK(turbo::ascii_is_alnum(c));
            else
                CHECK(!turbo::ascii_is_alnum(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (i != '\0' && strchr(" \r\n\t\v\f", i))
                CHECK(turbo::ascii_is_space(c));
            else
                CHECK(!turbo::ascii_is_space(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (i >= 32 && i < 127)
                CHECK(turbo::ascii_is_print(c));
            else
                CHECK(!turbo::ascii_is_print(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (turbo::ascii_is_print(c) && !turbo::ascii_is_space(c) &&
                !turbo::ascii_is_alnum(c)) {
                CHECK(turbo::ascii_is_punct(c));
            } else {
                CHECK(!turbo::ascii_is_punct(c));
            }
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (i == ' ' || i == '\t')
                CHECK(turbo::ascii_is_blank(c));
            else
                CHECK(!turbo::ascii_is_blank(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (i < 32 || i == 127)
                CHECK(turbo::ascii_is_cntrl(c));
            else
                CHECK(!turbo::ascii_is_cntrl(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (turbo::ascii_is_digit(c) || (i >= 'A' && i <= 'F') ||
                (i >= 'a' && i <= 'f')) {
                CHECK(turbo::ascii_is_xdigit(c));
            } else {
                CHECK(!turbo::ascii_is_xdigit(c));
            }
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (i > 32 && i < 127)
                CHECK(turbo::ascii_is_graph(c));
            else
                CHECK(!turbo::ascii_is_graph(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (i >= 'A' && i <= 'Z')
                CHECK(turbo::ascii_is_upper(c));
            else
                CHECK(!turbo::ascii_is_upper(c));
        }
        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (i >= 'a' && i <= 'z')
                CHECK(turbo::ascii_is_lower(c));
            else
                CHECK(!turbo::ascii_is_lower(c));
        }
        for (unsigned char c = 0; c < 128; c++) {
            CHECK(turbo::ascii_is_ascii(c));
        }
        for (int i = 128; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            CHECK(!turbo::ascii_is_ascii(c));
        }
    }

// Checks that turbo::ascii_isfoo returns the same value as isfoo in the C
// locale.
    TEST_CASE("AsciiIsFoo, SameAsIsFoo") {
#ifndef __ANDROID__
        // temporarily change locale to C. It should already be C, but just for safety
        const char *old_locale = setlocale(LC_CTYPE, "C");
        REQUIRE(old_locale != nullptr);
#endif

        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            CHECK_EQ(isalpha(c) != 0, turbo::ascii_is_alpha(c));
            CHECK_EQ(isdigit(c) != 0, turbo::ascii_is_digit(c));
            CHECK_EQ(isalnum(c) != 0, turbo::ascii_is_alnum(c));
            CHECK_EQ(isspace(c) != 0, turbo::ascii_is_space(c));
            CHECK_EQ(ispunct(c) != 0, turbo::ascii_is_punct(c));
            CHECK_EQ(isblank(c) != 0, turbo::ascii_is_blank(c));
            CHECK_EQ(iscntrl(c) != 0, turbo::ascii_is_cntrl(c));
            CHECK_EQ(isxdigit(c) != 0, turbo::ascii_is_xdigit(c));
            CHECK_EQ(isprint(c) != 0, turbo::ascii_is_print(c));
            CHECK_EQ(isgraph(c) != 0, turbo::ascii_is_graph(c));
            CHECK_EQ(isupper(c) != 0, turbo::ascii_is_upper(c));
            CHECK_EQ(islower(c) != 0, turbo::ascii_is_lower(c));
            CHECK_EQ(isascii(c) != 0, turbo::ascii_is_ascii(c));
        }

#ifndef __ANDROID__
        // restore the old locale.
        REQUIRE(setlocale(LC_CTYPE, old_locale));
#endif
    }

    TEST_CASE("AsciiToFoo, All") {

#ifndef __ANDROID__
        // temporarily change locale to C. It should already be C, but just for safety
        const char *old_locale = setlocale(LC_CTYPE, "C");
        REQUIRE(old_locale != nullptr);
#endif

        for (int i = 0; i < 256; i++) {
            const auto c = static_cast<unsigned char>(i);
            if (turbo::ascii_is_lower(c))
                CHECK_EQ(turbo::ascii_to_upper(c), 'A' + (i - 'a'));
            else
                CHECK_EQ(turbo::ascii_to_upper(c), static_cast<char>(i));

            if (turbo::ascii_is_upper(c))
                CHECK_EQ(turbo::ascii_to_lower(c), 'a' + (i - 'A'));
            else
                CHECK_EQ(turbo::ascii_to_lower(c), static_cast<char>(i));

            // These CHECKs only hold in a C locale.
            CHECK_EQ(static_cast<char>(tolower(i)), turbo::ascii_to_lower(c));
            CHECK_EQ(static_cast<char>(toupper(i)), turbo::ascii_to_upper(c));
        }
#ifndef __ANDROID__
        // restore the old locale.
        REQUIRE(setlocale(LC_CTYPE, old_locale));
#endif
    }

}  // namespace
