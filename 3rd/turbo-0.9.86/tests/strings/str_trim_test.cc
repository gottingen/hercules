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

#include "turbo/strings/str_trim.h"
#include "turbo/strings/inlined_string.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

TEST_CASE("trim_left, FromStringView") {
    CHECK_EQ(std::string_view{},
             turbo::trim_left(std::string_view{}));
    CHECK_EQ("foo", turbo::trim_left({"foo"}));
    CHECK_EQ("foo", turbo::trim_left({"\t  \n\f\r\n\vfoo"}));
    CHECK_EQ("foo foo\n ",
             turbo::trim_left({"\t  \n\f\r\n\vfoo foo\n "}));
    CHECK_EQ(std::string_view{}, turbo::trim_left(
            {"\t  \n\f\r\v\n\t  \n\f\r\v\n"}));
}


template<typename Str>
void TestInPlace() {
    Str str;

    turbo::trim_left(&str);
    CHECK_EQ("", str);

    str = "foo";
    turbo::trim_left(&str);
    CHECK_EQ("foo", str);

    str = "\t  \n\f\r\n\vfoo";
    turbo::trim_left(&str);
    CHECK_EQ("foo", str);

    str = "\t  \n\f\r\n\vfoo foo\n ";
    turbo::trim_left(&str);
    CHECK_EQ("foo foo\n ", str);

    str = "\t  \n\f\r\v\n\t  \n\f\r\v\n";
    turbo::trim_left(&str);
    CHECK_EQ(std::string_view{}, str);
}

TEST_CASE("trim_left, InPlace") {
    TestInPlace<std::string>();

    TestInPlace<turbo::inlined_string>();

}

TEST_CASE("trim_right, FromStringView") {
    CHECK_EQ(std::string_view{}, turbo::trim_right(std::string_view{}));
    CHECK_EQ("foo", turbo::trim_right({"foo"}));
    CHECK_EQ("foo", turbo::trim_right({"foo\t  \n\f\r\n\v"}));
    CHECK_EQ(" \nfoo foo", turbo::trim_right({" \nfoo foo\t  \n\f\r\n\v"}));
    CHECK_EQ(std::string_view{}, turbo::trim_right({"\t  \n\f\r\v\n\t  \n\f\r\v\n"}));
}

template<typename String>
void StripTrailingAsciiWhitespaceinplace() {
    String str;

    turbo::trim_right(&str);
    CHECK_EQ("", str);

    str = "foo";
    turbo::trim_right(&str);
    CHECK_EQ("foo", str);

    str = "foo\t  \n\f\r\n\v";
    turbo::trim_right(&str);
    CHECK_EQ("foo", str);

    str = " \nfoo foo\t  \n\f\r\n\v";
    turbo::trim_right(&str);
    CHECK_EQ(" \nfoo foo", str);

    str = "\t  \n\f\r\v\n\t  \n\f\r\v\n";
    turbo::trim_right(&str);
    CHECK_EQ(std::string_view{}, str);
}

TEST_CASE("trim_right, InPlace") {
    StripTrailingAsciiWhitespaceinplace<std::string>();

    StripTrailingAsciiWhitespaceinplace<turbo::inlined_string>();

}

TEST_CASE("trim_all, FromStringView") {
    CHECK_EQ(std::string_view{},
             turbo::trim_all(std::string_view{}));
    CHECK_EQ("foo", turbo::trim_all({"foo"}));
    CHECK_EQ("foo", turbo::trim_all({"\t  \n\f\r\n\vfoo\t  \n\f\r\n\v"}));
    CHECK_EQ("foo foo", turbo::trim_all({"\t  \n\f\r\n\vfoo foo\t  \n\f\r\n\v"}));
    CHECK_EQ(std::string_view{}, turbo::trim_all({"\t  \n\f\r\v\n\t  \n\f\r\v\n"}));
}

template<typename Str>
void StripAsciiWhitespaceInPlace() {
    Str str;

    turbo::trim_all(&str);
    CHECK_EQ("", str);

    str = "foo";
    turbo::trim_all(&str);
    CHECK_EQ("foo", str);

    str = "\t  \n\f\r\n\vfoo\t  \n\f\r\n\v";
    turbo::trim_all(&str);
    CHECK_EQ("foo", str);

    str = "\t  \n\f\r\n\vfoo foo\t  \n\f\r\n\v";
    turbo::trim_all(&str);
    CHECK_EQ("foo foo", str);

    str = "\t  \n\f\r\v\n\t  \n\f\r\v\n";
    turbo::trim_all(&str);
    CHECK_EQ(std::string_view{}, str);
}

TEST_CASE("trim_all, InPlace") {
    StripAsciiWhitespaceInPlace<std::string>();

    StripAsciiWhitespaceInPlace<turbo::inlined_string>();

}


template<typename String>
void RemoveExtraAsciiWhitespaceInplace() {
    const char *inputs[] = {"No extra space",
                            "  Leading whitespace",
                            "Trailing whitespace  ",
                            "  Leading and trailing  ",
                            " Whitespace \t  in\v   middle  ",
                            "'Eeeeep!  \n Newlines!\n",
                            "nospaces",
                            "",
                            "\n\t a\t\n\nb \t\n"};

    const char *outputs[] = {
            "No extra space",
            "Leading whitespace",
            "Trailing whitespace",
            "Leading and trailing",
            "Whitespace in middle",
            "'Eeeeep! Newlines!",
            "nospaces",
            "",
            "a\nb",
    };
    const int NUM_TESTS = TURBO_ARRAY_SIZE(inputs);

    for (int i = 0; i < NUM_TESTS; i++) {
        String s(inputs[i]);
        turbo::trim_complete(&s);
        CHECK_EQ(outputs[i], s);
    }
}

TEST_CASE("trim_complete, InPlace") {
    RemoveExtraAsciiWhitespaceInplace<std::string>();
    RemoveExtraAsciiWhitespaceInplace<turbo::inlined_string>();
}

TEST_CASE("pred") {
    std::string a = "abc ; ";
    turbo::by_any_of ba(" ;\t");
    auto trimed = turbo::trim_right(a, ba);
    CHECK_EQ(trimed, "abc");

    std::string b = " ; \tabc ; ";
    trimed = turbo::trim_all(b, ba);
    CHECK_EQ(trimed, "abc");
}