// Copyright 2023 The Turbo Authors.
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
//

#include "includes.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "collie/testing/test.h"

using clog::memory_buf_t;
using clog::details::fmt_helper::to_string_view;

void test_pad2(int n, const char *expected) {
    memory_buf_t buf;
    clog::details::fmt_helper::pad2(n, buf);

    REQUIRE_EQ(to_string_view(buf) , expected);
}

void test_pad3(uint32_t n, const char *expected) {
    memory_buf_t buf;
    clog::details::fmt_helper::pad3(n, buf);

    REQUIRE_EQ(to_string_view(buf) , expected);
}

void test_pad6(std::size_t n, const char *expected) {
    memory_buf_t buf;
    clog::details::fmt_helper::pad6(n, buf);

    REQUIRE_EQ(to_string_view(buf) , expected);
}

void test_pad9(std::size_t n, const char *expected) {
    memory_buf_t buf;
    clog::details::fmt_helper::pad9(n, buf);

    REQUIRE_EQ(to_string_view(buf) , expected);
}

TEST_CASE("pad2 [fmt_helper]")
{
    test_pad2(0, "00");
    test_pad2(3, "03");
    test_pad2(10, "10");
    test_pad2(23, "23");
    test_pad2(99, "99");
    test_pad2(100, "100");
    test_pad2(123, "123");
    test_pad2(1234, "1234");
    test_pad2(-5, "-5");
}

TEST_CASE("pad3 [fmt_helper]")
{
    test_pad3(0, "000");
    test_pad3(3, "003");
    test_pad3(10, "010");
    test_pad3(23, "023");
    test_pad3(99, "099");
    test_pad3(100, "100");
    test_pad3(123, "123");
    test_pad3(999, "999");
    test_pad3(1000, "1000");
    test_pad3(1234, "1234");
}

TEST_CASE("pad6 [fmt_helper]")
{
    test_pad6(0, "000000");
    test_pad6(3, "000003");
    test_pad6(23, "000023");
    test_pad6(123, "000123");
    test_pad6(1234, "001234");
    test_pad6(12345, "012345");
    test_pad6(123456, "123456");
}

TEST_CASE("pad9 [fmt_helper]")
{
    test_pad9(0, "000000000");
    test_pad9(3, "000000003");
    test_pad9(23, "000000023");
    test_pad9(123, "000000123");
    test_pad9(1234, "000001234");
    test_pad9(12345, "000012345");
    test_pad9(123456, "000123456");
    test_pad9(1234567, "001234567");
    test_pad9(12345678, "012345678");
    test_pad9(123456789, "123456789");
    test_pad9(1234567891, "1234567891");
}
