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

#include "turbo/strings/str_replace.h"

#include <list>
#include <map>
#include <tuple>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

#include "turbo/format/format.h"
#include "turbo/strings/str_split.h"

struct Cont {
    Cont() {}

    explicit Cont(std::string_view src) : data(src) {}

    std::string_view data;
};

template<int index>
std::string_view get(const Cont &c) {
    auto splitter = turbo::str_split(c.data, ':');
    auto it = splitter.begin();
    for (int i = 0; i < index; ++i) ++it;

    return *it;
}

TEST_CASE("str_replace_all") {
    SUBCASE("OneReplacement") {
        std::string s;

        // Empty string.
        s = turbo::str_replace_all(s, {{"", ""}});
        REQUIRE_EQ(s, "");
        s = turbo::str_replace_all(s, {{"x", ""}});
        REQUIRE_EQ(s, "");
        s = turbo::str_replace_all(s, {{"", "y"}});
        REQUIRE_EQ(s, "");
        s = turbo::str_replace_all(s, {{"x", "y"}});
        REQUIRE_EQ(s, "");

        // Empty substring.
        s = turbo::str_replace_all("abc", {{"", ""}});
        REQUIRE_EQ(s, "abc");
        s = turbo::str_replace_all("abc", {{"", "y"}});
        REQUIRE_EQ(s, "abc");
        s = turbo::str_replace_all("abc", {{"x", ""}});
        REQUIRE_EQ(s, "abc");

        // Substring not found.
        s = turbo::str_replace_all("abc", {{"xyz", "123"}});
        REQUIRE_EQ(s, "abc");

        // Replace entire string.
        s = turbo::str_replace_all("abc", {{"abc", "xyz"}});
        REQUIRE_EQ(s, "xyz");

        // Replace once at the start.
        s = turbo::str_replace_all("abc", {{"a", "x"}});
        REQUIRE_EQ(s, "xbc");

        // Replace once in the middle.
        s = turbo::str_replace_all("abc", {{"b", "x"}});
        REQUIRE_EQ(s, "axc");

        // Replace once at the end.
        s = turbo::str_replace_all("abc", {{"c", "x"}});
        REQUIRE_EQ(s, "abx");

        // Replace multiple times with varying lengths of original/replacement.
        s = turbo::str_replace_all("ababa", {{"a", "xxx"}});
        REQUIRE_EQ(s, "xxxbxxxbxxx");

        s = turbo::str_replace_all("ababa", {{"b", "xxx"}});
        REQUIRE_EQ(s, "axxxaxxxa");

        s = turbo::str_replace_all("aaabaaabaaa", {{"aaa", "x"}});
        REQUIRE_EQ(s, "xbxbx");

        s = turbo::str_replace_all("abbbabbba", {{"bbb", "x"}});
        REQUIRE_EQ(s, "axaxa");

        // Overlapping matches are replaced greedily.
        s = turbo::str_replace_all("aaa", {{"aa", "x"}});
        REQUIRE_EQ(s, "xa");

        // The replacements are not recursive.
        s = turbo::str_replace_all("aaa", {{"aa", "a"}});
        REQUIRE_EQ(s, "aa");
    }

    SUBCASE("ManyReplacements") {
        std::string s;

        // Empty string.
        s = turbo::str_replace_all("", {{"",  ""},
                                        {"x", ""},
                                        {"",  "y"},
                                        {"x", "y"}});
        REQUIRE_EQ(s, "");

        // Empty substring.
        s = turbo::str_replace_all("abc", {{"",  ""},
                                           {"",  "y"},
                                           {"x", ""}});
        REQUIRE_EQ(s, "abc");

        // Replace entire string, one char at a time
        s = turbo::str_replace_all("abc", {{"a", "x"},
                                           {"b", "y"},
                                           {"c", "z"}});
        REQUIRE_EQ(s, "xyz");
        s = turbo::str_replace_all("zxy", {{"z", "x"},
                                           {"x", "y"},
                                           {"y", "z"}});
        REQUIRE_EQ(s, "xyz");

        // Replace once at the start (longer matches take precedence)
        s = turbo::str_replace_all("abc", {{"a",   "x"},
                                           {"ab",  "xy"},
                                           {"abc", "xyz"}});
        REQUIRE_EQ(s, "xyz");

        // Replace once in the middle.
        s = turbo::str_replace_all(
                "Abc!", {{"a",  "x"},
                         {"ab", "xy"},
                         {"b",  "y"},
                         {"bc", "yz"},
                         {"c",  "z"}});
        REQUIRE_EQ(s, "Ayz!");

        // Replace once at the end.
        s = turbo::str_replace_all(
                "Abc!",
                {{"a",   "x"},
                 {"ab",  "xy"},
                 {"b",   "y"},
                 {"bc!", "yz?"},
                 {"c!",  "z;"}});
        REQUIRE_EQ(s, "Ayz?");

        // Replace multiple times with varying lengths of original/replacement.
        s = turbo::str_replace_all("ababa", {{"a", "xxx"},
                                             {"b", "XXXX"}});
        REQUIRE_EQ(s, "xxxXXXXxxxXXXXxxx");

        // Overlapping matches are replaced greedily.
        s = turbo::str_replace_all("aaa", {{"aa", "x"},
                                           {"a",  "X"}});
        REQUIRE_EQ(s, "xX");
        s = turbo::str_replace_all("aaa", {{"a",  "X"},
                                           {"aa", "x"}});
        REQUIRE_EQ(s, "xX");

        // Two well-known sentences
        s = turbo::str_replace_all("the quick brown fox jumped over the lazy dogs",
                                   {
                                           {"brown",    "box"},
                                           {"dogs",     "jugs"},
                                           {"fox",      "with"},
                                           {"jumped",   "five"},
                                           {"over",     "dozen"},
                                           {"quick",    "my"},
                                           {"the",      "pack"},
                                           {"the lazy", "liquor"},
                                   });
        REQUIRE_EQ(s, "pack my box with five dozen liquor jugs");
    }

    SUBCASE("ManyReplacementsInMap") {
        std::map<const char *, const char *> replacements;
        replacements["$who"] = "Bob";
        replacements["$count"] = "5";
        replacements["#Noun"] = "Apples";
        std::string s = turbo::str_replace_all("$who bought $count #Noun. Thanks $who!",
                                               replacements);
        REQUIRE_EQ("Bob bought 5 Apples. Thanks Bob!", s);
    }

    SUBCASE("ReplacementsInPlace") {
        std::string s = std::string("$who bought $count #Noun. Thanks $who!");
        int count;
        count = turbo::str_replace_all({{"$count", turbo::format(5)},
                                        {"$who",   "Bob"},
                                        {"#Noun",  "Apples"}}, &s);
        REQUIRE_EQ(count, 4);
        REQUIRE_EQ("Bob bought 5 Apples. Thanks Bob!", s);
    }

    SUBCASE("ReplacementsInPlaceInMap") {
        std::string s = std::string("$who bought $count #Noun. Thanks $who!");
        std::map<std::string_view, std::string_view> replacements;
        replacements["$who"] = "Bob";
        replacements["$count"] = "5";
        replacements["#Noun"] = "Apples";
        int count;
        count = turbo::str_replace_all(replacements, &s);
        REQUIRE_EQ(count, 4);
        REQUIRE_EQ("Bob bought 5 Apples. Thanks Bob!", s);
    }

    SUBCASE("VariableNumber") {
        std::string s;
        {
            std::vector<std::pair<std::string, std::string>> replacements;

            s = "abc";
            REQUIRE_EQ(0, turbo::str_replace_all(replacements, &s));
            REQUIRE_EQ("abc", s);

            s = "abc";
            replacements.push_back({"a", "A"});
            REQUIRE_EQ(1, turbo::str_replace_all(replacements, &s));
            REQUIRE_EQ("Abc", s);

            s = "abc";
            replacements.push_back({"b", "B"});
            REQUIRE_EQ(2, turbo::str_replace_all(replacements, &s));
            REQUIRE_EQ("ABc", s);

            s = "abc";
            replacements.push_back({"d", "D"});
            REQUIRE_EQ(2, turbo::str_replace_all(replacements, &s));
            REQUIRE_EQ("ABc", s);

            REQUIRE_EQ("ABcABc", turbo::str_replace_all("abcabc", replacements));
        }

        {
            std::map<const char *, const char *> replacements;
            replacements["aa"] = "x";
            replacements["a"] = "X";
            s = "aaa";
            REQUIRE_EQ(2, turbo::str_replace_all(replacements, &s));
            REQUIRE_EQ("xX", s);

            REQUIRE_EQ("xxX", turbo::str_replace_all("aaaaa", replacements));
        }

        {
            std::list<std::pair<std::string_view, std::string_view>> replacements = {
                    {"a", "x"},
                    {"b", "y"},
                    {"c", "z"}};

            std::string s = turbo::str_replace_all("abc", replacements);
            REQUIRE_EQ(s, "xyz");
        }

        {
            using X = std::tuple<std::string_view, std::string, int>;
            std::vector<X> replacements(3);
            replacements[0] = X{"a", "x", 1};
            replacements[1] = X{"b", "y", 0};
            replacements[2] = X{"c", "z", -1};

            std::string s = turbo::str_replace_all("abc", replacements);
            REQUIRE_EQ(s, "xyz");
        }

        {
            std::vector<Cont> replacements(3);
            replacements[0] = Cont{"a:x"};
            replacements[1] = Cont{"b:y"};
            replacements[2] = Cont{"c:z"};

            std::string s = turbo::str_replace_all("abc", replacements);
            REQUIRE_EQ(s, "xyz");
        }
    }

// Same as above, but using the in-place variant of turbo::str_replace_all,
// that returns the # of replacements performed.
    SUBCASE("Inplace") {
        std::string s;
        int reps;

        // Empty string.
        s = "";
        reps = turbo::str_replace_all({{"",  ""},
                                       {"x", ""},
                                       {"",  "y"},
                                       {"x", "y"}}, &s);
        REQUIRE_EQ(reps, 0);
        REQUIRE_EQ(s, "");

        // Empty substring.
        s = "abc";
        reps = turbo::str_replace_all({{"",  ""},
                                       {"",  "y"},
                                       {"x", ""}}, &s);
        REQUIRE_EQ(reps, 0);
        REQUIRE_EQ(s, "abc");

        // Replace entire string, one char at a time
        s = "abc";
        reps = turbo::str_replace_all({{"a", "x"},
                                       {"b", "y"},
                                       {"c", "z"}}, &s);
        REQUIRE_EQ(reps, 3);
        REQUIRE_EQ(s, "xyz");
        s = "zxy";
        reps = turbo::str_replace_all({{"z", "x"},
                                       {"x", "y"},
                                       {"y", "z"}}, &s);
        REQUIRE_EQ(reps, 3);
        REQUIRE_EQ(s, "xyz");

        // Replace once at the start (longer matches take precedence)
        s = "abc";
        reps = turbo::str_replace_all({{"a",   "x"},
                                       {"ab",  "xy"},
                                       {"abc", "xyz"}}, &s);
        REQUIRE_EQ(reps, 1);
        REQUIRE_EQ(s, "xyz");

        // Replace once in the middle.
        s = "Abc!";
        reps = turbo::str_replace_all(
                {{"a",  "x"},
                 {"ab", "xy"},
                 {"b",  "y"},
                 {"bc", "yz"},
                 {"c",  "z"}}, &s);
        REQUIRE_EQ(reps, 1);
        REQUIRE_EQ(s, "Ayz!");

        // Replace once at the end.
        s = "Abc!";
        reps = turbo::str_replace_all(
                {{"a",   "x"},
                 {"ab",  "xy"},
                 {"b",   "y"},
                 {"bc!", "yz?"},
                 {"c!",  "z;"}}, &s);
        REQUIRE_EQ(reps, 1);
        REQUIRE_EQ(s, "Ayz?");

        // Replace multiple times with varying lengths of original/replacement.
        s = "ababa";
        reps = turbo::str_replace_all({{"a", "xxx"},
                                       {"b", "XXXX"}}, &s);
        REQUIRE_EQ(reps, 5);
        REQUIRE_EQ(s, "xxxXXXXxxxXXXXxxx");

        // Overlapping matches are replaced greedily.
        s = "aaa";
        reps = turbo::str_replace_all({{"aa", "x"},
                                       {"a",  "X"}}, &s);
        REQUIRE_EQ(reps, 2);
        REQUIRE_EQ(s, "xX");
        s = "aaa";
        reps = turbo::str_replace_all({{"a",  "X"},
                                       {"aa", "x"}}, &s);
        REQUIRE_EQ(reps, 2);
        REQUIRE_EQ(s, "xX");

        // Two well-known sentences
        s = "the quick brown fox jumped over the lazy dogs";
        reps = turbo::str_replace_all(
                {
                        {"brown",    "box"},
                        {"dogs",     "jugs"},
                        {"fox",      "with"},
                        {"jumped",   "five"},
                        {"over",     "dozen"},
                        {"quick",    "my"},
                        {"the",      "pack"},
                        {"the lazy", "liquor"},
                },
                &s);
        REQUIRE_EQ(reps, 8);
        REQUIRE_EQ(s, "pack my box with five dozen liquor jugs");
    }
}
