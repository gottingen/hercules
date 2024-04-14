// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/format/print.h"
#include "turbo/strings/unicode_view.h"
#include "turbo/strings/unicode.h"

using namespace turbo;

TEST_CASE("UnicodeView, ConstChars") {
    unicode_view a = U"hello";
    REQUIRE_EQ(a, U"hello");
    REQUIRE_NE(a, U"hh");
    unicode_view b(U"hello");
    REQUIRE_EQ(a, b);
    unicode_view c(U"");
    REQUIRE_EQ(c, U"");
    REQUIRE_NE(c, U"hh");
    REQUIRE_NE(c, a);
    std::cout << a << std::endl;
}

TEST_CASE("UnicodeView, std_string") {
    std::basic_string<char32_t> std_s(U"hello");
    unicode_view a = U"hello";
    REQUIRE_EQ(a, std_s);
    a = std_s;
    REQUIRE_EQ(a, std_s);
    std::basic_string<char32_t> new_std_s((const char32_t *) a.data(), a.size());
    REQUIRE_EQ(new_std_s, std_s);
    std::cout << a << std::endl;
    turbo::println("{}", a);
    turbo::println("{}", turbo::Hash<unicode_view>()(a));
}

TEST_CASE("UnicodeView, LargeConverter") {
    Unicode raw;
    raw.resize(4096, 'a');
    unicode_view raw_view = raw;
    Unicode copy1 = raw;
    Unicode copy2 = Unicode(raw_view);
    REQUIRE_EQ(copy1, copy2);
}

TEST_CASE("UnicodeView, SmallConverter") {
    Unicode raw;
    raw.resize(1, 'a');
    unicode_view raw_view = raw;
    Unicode copy1 = raw;
    Unicode copy2 = Unicode(raw_view);
    REQUIRE(copy1.data() != copy2.data());
    REQUIRE(copy1 == copy2);

}