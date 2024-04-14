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

#include <string>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/strings/inlined_string.h"
#include "turbo/format/format.h"
#include "turbo/format/print.h"
#include "turbo/format/fmt/std.h"
#include <memory>

TEST_CASE("format -InlinedString") {

    turbo::inlined_string s = turbo::format(42);
    CHECK_EQ(s, "42");
    std::string stdstr  = turbo::format(42);
    CHECK_EQ(stdstr, "42");
}

TEST_CASE("format -InlinedString") {
    std::vector<int> array = {1, 2, 4};
    auto s = turbo::format_range("{}",array, ", ");
    CHECK_EQ(s, "1, 2, 4");
    turbo::format_range_append(&s,"{}",array, ", ");
    CHECK_EQ(s, "1, 2, 41, 2, 4");
}

enum TestE {
    A, B, C
};
enum TestE1 {
    AA, BB, CC
};
enum class TestE2 {
    AAA, BBB, CCC
};
TEST_CASE("format -enum") {
    auto e = TestE::A;
    auto e1 = TestE1::AA;
    auto s = turbo::format("{}", e);
    auto s1 = turbo::format("{}", e1);
    turbo::println("{}", s);
    turbo::println("{}", s1);

    auto e2 = TestE2::AAA;
    auto s2 = turbo::format("{}", e2);
    turbo::println("{:>30}", s2);
    std::vector<int> array = {1, 2, 4};
    turbo::println("{}", array);
    std::set<int> set = {1, 2, 4};
    turbo::println("{}", set);
    std::map<int, int> map = {{1, 2},
                              {2, 3},
                              {4, 5}};
    turbo::println("{}", map);
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    turbo::println("{}", ptr);

    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
    turbo::println("{}", ptr1);

   std::vector<std::vector<int>> vv = {
           {1, 2, 3},
           {4, 5, 6}
   };
   turbo::println("{}", vv);
}

TEST_CASE("format container smart pointer") {
   std::shared_ptr<double> ptr;
   turbo::println("{}", ptr);
   turbo::println("{}", turbo::underlying(TestE::B));
}