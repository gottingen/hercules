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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/strings/str_case_conv.h"
#include "turbo/strings/ascii.h"

TEST_CASE("AsciiStrTo, Lower") {
    const char buf[] = "ABCDEF";
    const std::string str("GHIJKL");
    const std::string str2("MNOPQR");
    const std::string_view sp(str2);
    std::string mutable_str("STUVWX");

    CHECK_EQ("abcdef", turbo::str_to_lower(buf));
    CHECK_EQ("ghijkl", turbo::str_to_lower(str));
    CHECK_EQ("mnopqr", turbo::str_to_lower(sp));

    turbo::str_to_lower(&mutable_str);
    CHECK_EQ("stuvwx", mutable_str);

    char mutable_buf[] = "Mutable";
    std::transform(mutable_buf, mutable_buf + strlen(mutable_buf),
                   mutable_buf, turbo::ascii_to_lower);
    CHECK_EQ("mutable", std::string_view(mutable_buf));
}

TEST_CASE("AsciiStrTo, Upper") {
    const char buf[] = "abcdef";
    const std::string str("ghijkl");
    const std::string str2("mnopqr");
    const std::string_view sp(str2);

    CHECK_EQ("ABCDEF", turbo::str_to_upper(buf));
    CHECK_EQ("GHIJKL", turbo::str_to_upper(str));
    CHECK_EQ("MNOPQR", turbo::str_to_upper(sp));

    char mutable_buf[] = "Mutable";
    std::transform(mutable_buf, mutable_buf + strlen(mutable_buf),
                   mutable_buf, turbo::ascii_to_upper);
    CHECK_EQ("MUTABLE", std::string_view(mutable_buf));
}

