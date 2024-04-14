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

// Unit test for memutil.cc

#include "turbo/strings/internal/memutil.h"

#include <cstdlib>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/strings/ascii.h"

namespace {

    static char *memcasechr(const char *s, int c, size_t slen) {
        c = turbo::ascii_to_lower(c);
        for (; slen; ++s, --slen) {
            if (turbo::ascii_to_lower(*s) == c) return const_cast<char *>(s);
        }
        return nullptr;
    }

    static const char *memcasematch(const char *phaystack, size_t haylen,
                                    const char *pneedle, size_t neelen) {
        if (0 == neelen) {
            return phaystack;  // even if haylen is 0
        }
        if (haylen < neelen) return nullptr;

        const char *match;
        const char *hayend = phaystack + haylen - neelen + 1;
        while ((match = static_cast<char *>(
                memcasechr(phaystack, pneedle[0], hayend - phaystack)))) {
            if (turbo::strings_internal::memcasecmp(match, pneedle, neelen) == 0)
                return match;
            else
                phaystack = match + 1;
        }
        return nullptr;
    }

    TEST_CASE("MemUtilTest, AllTests") {
        // check memutil functions
        char a[1000];
        turbo::strings_internal::memcat(a, 0, "hello", sizeof("hello") - 1);
        turbo::strings_internal::memcat(a, 5, " there", sizeof(" there") - 1);

        CHECK_EQ(turbo::strings_internal::memcasecmp(a, "heLLO there",
                                                     sizeof("hello there") - 1),
                 0);
        CHECK_EQ(turbo::strings_internal::memcasecmp(a, "heLLO therf",
                                                     sizeof("hello there") - 1),
                 -1);
        CHECK_EQ(turbo::strings_internal::memcasecmp(a, "heLLO therf",
                                                     sizeof("hello there") - 2),
                 0);
        CHECK_EQ(turbo::strings_internal::memcasecmp(a, "whatever", 0), 0);

        char *p = turbo::strings_internal::memdup("hello", 5);
        free(p);

        p = turbo::strings_internal::memrchr("hello there", 'e',
                                             sizeof("hello there") - 1);
        CHECK((p && p[-1] == 'r'));
        p = turbo::strings_internal::memrchr("hello there", 'e',
                                             sizeof("hello there") - 2);
        CHECK((p && p[-1] == 'h'));
        p = turbo::strings_internal::memrchr("hello there", 'u',
                                             sizeof("hello there") - 1);
        CHECK((p == nullptr));

        auto len = turbo::strings_internal::memspn("hello there",
                                                  sizeof("hello there") - 1, "hole");
        CHECK_EQ(len, sizeof("hello") - 1);
        len = turbo::strings_internal::memspn("hello there", sizeof("hello there") - 1,
                                              "u");
        CHECK_EQ(len, 0);
        len = turbo::strings_internal::memspn("hello there", sizeof("hello there") - 1,
                                              "");
        CHECK_EQ(len, 0);
        len = turbo::strings_internal::memspn("hello there", sizeof("hello there") - 1,
                                              "trole h");
        CHECK_EQ(len, sizeof("hello there") - 1);
        len = turbo::strings_internal::memspn("hello there!",
                                              sizeof("hello there!") - 1, "trole h");
        CHECK_EQ(len, sizeof("hello there") - 1);
        len = turbo::strings_internal::memspn("hello there!",
                                              sizeof("hello there!") - 2, "trole h!");
        CHECK_EQ(len, sizeof("hello there!") - 2);

        len = turbo::strings_internal::memcspn("hello there",
                                               sizeof("hello there") - 1, "leho");
        CHECK_EQ(len, 0);
        len = turbo::strings_internal::memcspn("hello there",
                                               sizeof("hello there") - 1, "u");
        CHECK_EQ(len, sizeof("hello there") - 1);
        len = turbo::strings_internal::memcspn("hello there",
                                               sizeof("hello there") - 1, "");
        CHECK_EQ(len, sizeof("hello there") - 1);
        len = turbo::strings_internal::memcspn("hello there",
                                               sizeof("hello there") - 1, " ");
        CHECK_EQ(len, 5);

        p = turbo::strings_internal::mempbrk("hello there", sizeof("hello there") - 1,
                                             "leho");
        CHECK((p && p[1] == 'e' && p[2] == 'l'));
        p = turbo::strings_internal::mempbrk("hello there", sizeof("hello there") - 1,
                                             "nu");
        CHECK_EQ(p , nullptr);
        p = turbo::strings_internal::mempbrk("hello there!",
                                             sizeof("hello there!") - 2, "!");
        CHECK_EQ(p , nullptr);
        p = turbo::strings_internal::mempbrk("hello there", sizeof("hello there") - 1,
                                             " t ");
        CHECK((p && p[-1] == 'o' && p[1] == 't'));

        {
            const char kHaystack[] = "0123456789";
            CHECK_EQ(turbo::strings_internal::memmem(kHaystack, 0, "", 0), kHaystack);
            CHECK_EQ(turbo::strings_internal::memmem(kHaystack, 10, "012", 3),
                     kHaystack);
            CHECK_EQ(turbo::strings_internal::memmem(kHaystack, 10, "0xx", 1),
                     kHaystack);
            CHECK_EQ(turbo::strings_internal::memmem(kHaystack, 10, "789", 3),
                     kHaystack + 7);
            CHECK_EQ(turbo::strings_internal::memmem(kHaystack, 10, "9xx", 1),
                     kHaystack + 9);
            CHECK_EQ(turbo::strings_internal::memmem(kHaystack, 10, "9xx", 3) ,
                  nullptr);
            CHECK_EQ(turbo::strings_internal::memmem(kHaystack, 10, "xxx", 1) ,
                  nullptr);
        }
        {
            const char kHaystack[] = "aBcDeFgHiJ";
            CHECK_EQ(turbo::strings_internal::memcasemem(kHaystack, 0, "", 0),
                     kHaystack);
            CHECK_EQ(turbo::strings_internal::memcasemem(kHaystack, 10, "Abc", 3),
                     kHaystack);
            CHECK_EQ(turbo::strings_internal::memcasemem(kHaystack, 10, "Axx", 1),
                     kHaystack);
            CHECK_EQ(turbo::strings_internal::memcasemem(kHaystack, 10, "hIj", 3),
                     kHaystack + 7);
            CHECK_EQ(turbo::strings_internal::memcasemem(kHaystack, 10, "jxx", 1),
                     kHaystack + 9);
            CHECK_EQ(turbo::strings_internal::memcasemem(kHaystack, 10, "jxx", 3),
                     nullptr);
            CHECK_EQ(turbo::strings_internal::memcasemem(kHaystack, 10, "xxx", 1),
                     nullptr);
        }
        {
            const char kHaystack[] = "0123456789";
            CHECK_EQ(turbo::strings_internal::memmatch(kHaystack, 0, "", 0), kHaystack);
            CHECK_EQ(turbo::strings_internal::memmatch(kHaystack, 10, "012", 3),
                     kHaystack);
            CHECK_EQ(turbo::strings_internal::memmatch(kHaystack, 10, "0xx", 1),
                     kHaystack);
            CHECK_EQ(turbo::strings_internal::memmatch(kHaystack, 10, "789", 3),
                     kHaystack + 7);
            CHECK_EQ(turbo::strings_internal::memmatch(kHaystack, 10, "9xx", 1),
                     kHaystack + 9);
            CHECK_EQ(turbo::strings_internal::memmatch(kHaystack, 10, "9xx", 3),
                     nullptr);
            CHECK_EQ(turbo::strings_internal::memmatch(kHaystack, 10, "xxx", 1),
                     nullptr);
        }
    }

}  // namespace
