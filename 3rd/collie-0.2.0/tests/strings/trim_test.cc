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

#include <collie/strings/trim.h>
#include <collie/strings/inlined_string.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <collie/testing/doctest.h>

TEST_CASE("trim_left, FromStringView") {
    CHECK_EQ(std::string_view{},
             collie::trim_left(std::string_view{}));
    CHECK_EQ("foo", collie::trim_left({"foo"}));
    CHECK_EQ("foo", collie::trim_left({"\t  \n\f\r\n\vfoo"}));
    CHECK_EQ("foo foo\n ",
             collie::trim_left({"\t  \n\f\r\n\vfoo foo\n "}));
    CHECK_EQ(std::string_view{}, collie::trim_left(
            {"\t  \n\f\r\v\n\t  \n\f\r\v\n"}));
}


void TestInPlace() {
    std::string str;

    collie::trim_left(&str);
    CHECK_EQ("", str);

    str = "foo";
    collie::trim_left(&str);
    CHECK_EQ("foo", str);

    str = "\t  \n\f\r\n\vfoo";
    collie::trim_left(&str);
    CHECK_EQ("foo", str);

    str = "\t  \n\f\r\n\vfoo foo\n ";
    collie::trim_left(&str);
    CHECK_EQ("foo foo\n ", str);

    str = "\t  \n\f\r\v\n\t  \n\f\r\v\n";
    collie::trim_left(&str);
    CHECK_EQ(std::string_view{}, str);
}

TEST_CASE("trim_left, InPlace") {
    TestInPlace();
}

TEST_CASE("trim_right, FromStringView") {
    CHECK_EQ(std::string_view{}, collie::trim_right(std::string_view{}));
    CHECK_EQ("foo", collie::trim_right({"foo"}));
    CHECK_EQ("foo", collie::trim_right({"foo\t  \n\f\r\n\v"}));
    CHECK_EQ(" \nfoo foo", collie::trim_right({" \nfoo foo\t  \n\f\r\n\v"}));
    CHECK_EQ(std::string_view{}, collie::trim_right({"\t  \n\f\r\v\n\t  \n\f\r\v\n"}));
}

template<typename String>
void StripTrailingAsciiWhitespaceinplace() {
    String str;

    collie::trim_right(&str);
    CHECK_EQ("", str);

    str = "foo";
    collie::trim_right(&str);
    CHECK_EQ("foo", str);

    str = "foo\t  \n\f\r\n\v";
    collie::trim_right(&str);
    CHECK_EQ("foo", str);

    str = " \nfoo foo\t  \n\f\r\n\v";
    collie::trim_right(&str);
    CHECK_EQ(" \nfoo foo", str);

    str = "\t  \n\f\r\v\n\t  \n\f\r\v\n";
    collie::trim_right(&str);
    CHECK_EQ(std::string_view{}, str);
}

TEST_CASE("trim_right, InPlace") {
    StripTrailingAsciiWhitespaceinplace<std::string>();
}

TEST_CASE("trim_all, FromStringView") {
    CHECK_EQ(std::string_view{},
             collie::trim_all(std::string_view{}));
    CHECK_EQ("foo", collie::trim_all({"foo"}));
    CHECK_EQ("foo", collie::trim_all({"\t  \n\f\r\n\vfoo\t  \n\f\r\n\v"}));
    CHECK_EQ("foo foo", collie::trim_all({"\t  \n\f\r\n\vfoo foo\t  \n\f\r\n\v"}));
    CHECK_EQ(std::string_view{}, collie::trim_all({"\t  \n\f\r\v\n\t  \n\f\r\v\n"}));
}

void StripAsciiWhitespaceInPlace() {
    std::string str;

    collie::trim_all(&str);
    CHECK_EQ("", str);

    str = "foo";
    collie::trim_all(&str);
    CHECK_EQ("foo", str);

    str = "\t  \n\f\r\n\vfoo\t  \n\f\r\n\v";
    collie::trim_all(&str);
    CHECK_EQ("foo", str);

    str = "\t  \n\f\r\n\vfoo foo\t  \n\f\r\n\v";
    collie::trim_all(&str);
    CHECK_EQ("foo foo", str);

    str = "\t  \n\f\r\v\n\t  \n\f\r\v\n";
    collie::trim_all(&str);
    CHECK_EQ(std::string_view{}, str);
}

TEST_CASE("trim_all, InPlace") {
    StripAsciiWhitespaceInPlace();
}


void RemoveExtraAsciiWhitespaceInplace() {
    std::vector<const char *> inputs = {"No extra space",
                            "  Leading whitespace",
                            "Trailing whitespace  ",
                            "  Leading and trailing  ",
                            " Whitespace \t  in\v   middle  ",
                            "'Eeeeep!  \n Newlines!\n",
                            "nospaces",
                            "",
                            "\n\t a\t\n\nb \t\n"};

    std::vector<const char *> outputs = {
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
    const int NUM_TESTS = inputs.size();

    for (int i = 0; i < NUM_TESTS; i++) {
        std::string s(inputs[i]);
        collie::trim_complete(&s);
        CHECK_EQ(outputs[i], s);
    }
}

TEST_CASE("trim_complete, InPlace") {
    RemoveExtraAsciiWhitespaceInplace();
}

TEST_CASE("pred") {
    std::string a = "abc ; ";
    collie::by_any_of ba(" ;\t");
    auto trimed = collie::trim_right(a, ba);
    CHECK_EQ(trimed, "abc");

    std::string b = " ; \tabc ; ";
    trimed = collie::trim_all(b, ba);
    CHECK_EQ(trimed, "abc");
}