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
//

#include "turbo/strings/fixed_string.h"
#include <array>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

using namespace std::string_literals;

/* define types */
using string8 = turbo::fixed_string<7>;
using string64 = turbo::fixed_string<63>;

TEST_CASE("TestFixture, default_construction")
{
    constexpr string8 a{};

    REQUIRE_EQ(0, a.length());
    REQUIRE(a.empty());
    REQUIRE_EQ(7, a.max_size());
}

TEST_CASE("TestFixture, construction_from_constexpr")
{
    constexpr const char *str = "1234";
    string8 a = str;

    REQUIRE_EQ(4, a.length());
    REQUIRE_FALSE(a.empty());
    REQUIRE_EQ(7, a.max_size());
}

TEST_CASE("TestFixture, construction_from_const_buffer")
{
    using string24 = turbo::fixed_string<23>;
    constexpr std::array<char, 24> buffer{'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!', '\0'};
    constexpr size_t buffer_str_size = 12;

    string24 a(buffer.data(), buffer_str_size);

    REQUIRE_EQ("Hello World!"s, a.c_str());
    REQUIRE_EQ(12, a.length());
    REQUIRE_FALSE(a.empty());
    REQUIRE_EQ(23, a.max_size());
}

TEST_CASE("TestFixture, construction_from_buffer")
{
    using string24 = turbo::fixed_string<23>;
    std::array<char, 24> buffer = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!', '\0'};
    size_t buffer_str_size = 12;

    string24 a(buffer.data(), buffer_str_size);

    REQUIRE_EQ("Hello World!"s, a.c_str());
    REQUIRE_EQ(12, a.length());
    REQUIRE_FALSE(a.empty());
    REQUIRE_EQ(23, a.max_size());
}

TEST_CASE("TestFixture, copy_construction")
{
    string8 a{"abc"};
    auto a_copy = a;

    REQUIRE_EQ(a_copy.length(), a.length());
    REQUIRE_FALSE(a.empty());
    REQUIRE_FALSE(a_copy.empty());
    REQUIRE_EQ(a_copy.max_size(), a.max_size());
}

TEST_CASE("TestFixture, move_construction")
{
    string8 a{"abc"};
    auto a_copy = a;
    auto a_move = std::move(a_copy);

    REQUIRE_EQ(a_move.length(), a.length());
    REQUIRE_FALSE(a.empty());
    REQUIRE_FALSE(a_move.empty());
    REQUIRE_EQ(a_move.max_size(), a.max_size());
}

TEST_CASE("TestFixture, copy_assignment")
{
    string8 b{"1234", 4};
    string8 c{"lmnopqrstuvxyz"};

    REQUIRE_EQ("1234"s, b.c_str());
    REQUIRE_EQ("lmnopqr"s, c.c_str()); // c is "lmnopqr"

    c = b;

    REQUIRE_EQ("1234"s, c.c_str());
}

TEST_CASE("TestFixture, move_assignment")
{
    string8 a{"1234", 4};
    string8 b{"lmnopqrstuvxyz"};
    b = std::move(a);

    REQUIRE_EQ("1234"s, b.c_str());
}

TEST_CASE("TestFixture, using_with_string_view")
{
    string8 e{"abcdefghij", 10};
    auto e_sub_str = e.str().substr(0, 2);
    auto e_str = e.str();
    auto e_length = e.length();

    REQUIRE_EQ("abcdefg"s, e.c_str()); // truncated. e is "abcdefg";
    REQUIRE_EQ("ab"s, e_sub_str);
    REQUIRE_EQ("abcdefg"s, e_str);
    REQUIRE_EQ(7, e.length());
}

TEST_CASE("TestFixture, comparison")
{
    string8 f{"abcd"};
    string8 g{"abcd"};
    string8 h{"abcf"};

    REQUIRE((f == g));
    REQUIRE_FALSE((g == h));
}

TEST_CASE("TestFixture, append")
{
    string8 k{"abc"}; // k is "abc"
    REQUIRE_EQ("abc"s, k.c_str());

    k.append("d");
    REQUIRE_EQ("abcd"s, k.c_str());

    k.append("efghi", 5);
    REQUIRE_EQ("abcdefg"s, k.c_str()); // k is "abcdefg". rest is truncated
    auto str = turbo::format("{}", k);
    REQUIRE_EQ("abcdefg"s, str);
}

TEST_CASE("TestFixture, clear")
{
    string8 k{"abcdefg"};
    REQUIRE_EQ("abcdefg"s, k.c_str());
    REQUIRE_FALSE(k.empty());

    k.clear();
    REQUIRE_EQ(""s, k.c_str());
    REQUIRE(k.empty());
}

TEST_CASE("TestFixture, reset")
{
    string8 k{"abcdefg"};
    REQUIRE_EQ("abcdefg"s, k.c_str());
    REQUIRE_EQ(7, k.length());

    k.reset("1234");
    REQUIRE_EQ("1234"s, k.c_str());
    REQUIRE_EQ(4, k.length());

    k.reset("xyz", 3);
    REQUIRE_EQ("xyz"s, k.c_str());
    REQUIRE_EQ(3, k.length());
}

TEST_CASE("TestFixture, remove_suffix")
{
    string8 l{"1234567"};
    REQUIRE_EQ("1234567"s, l.c_str());

    l.remove_suffix(3);
    REQUIRE_EQ("1234"s, l.c_str());
}

TEST_CASE("TestFixture, remove_prefix")
{
    string8 l{"1234"};
    REQUIRE_EQ("1234"s, l.c_str());

    l.remove_prefix(2);
    REQUIRE_EQ("34"s, l.c_str());
}

TEST_CASE("TestFixture, stream_operator")
{
    string8 l{"1234"};
    std::cout << l << std::endl;
    REQUIRE(1);
}

TEST_CASE("TestFixture, swap")
{
    string8 k{"xyz"};
    string8 l{"34"};

    l.swap(k);
    // l is "xyz" and k is "34"
    REQUIRE_EQ("xyz"s, l.c_str());
    REQUIRE_EQ("34"s, k.c_str());

    std::swap(l, k);
    // l is "34" and k is "xyz"
    REQUIRE_EQ("34"s, l.c_str());
    REQUIRE_EQ("xyz"s, k.c_str());
}

struct test_struct {
    std::uint32_t a_{};
    std::uint64_t b_{};
    string8 c_ = "abcd";

    constexpr auto get_c() const noexcept { return c_.c_str(); }

    constexpr auto get_c_str() const noexcept { return c_.str(); }

    constexpr void set_c(const char *str) { c_.reset(str); }
};

TEST_CASE("TestFixture, using_member_variables")
{
    constexpr auto test_struct_size = sizeof(test_struct);
    // uses only 8 + 4 bytes in stack
    REQUIRE_EQ(32, sizeof(test_struct));

    test_struct t{};
    auto t_c = t.get_c();
    auto t_c_str = t.get_c_str();

    REQUIRE_EQ("abcd"s, t_c);
    REQUIRE_EQ("abcd"s, t_c_str);

    test_struct t2{};
    t2.set_c("123456");
    const auto t2_c = t2.get_c();
    const auto t2_c_str = t2.get_c();
    REQUIRE_EQ(32, sizeof(t2));

    REQUIRE_EQ("123456"s, t2_c);
    REQUIRE_EQ("123456"s, t2_c_str);
}