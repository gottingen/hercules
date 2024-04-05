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

#include <collie/testing/doctest.h>
#include <collie/strings/inlined_string.h>

#define CHECK_STREQ(a, b) CHECK_EQ(std::string_view((a)), std::string_view((b)))

using namespace collie;
namespace {

    // Test fixture class
    class SmallStringTest {
    public:
        typedef InlinedString<40> StringType;

        StringType theString;

        void assertEmpty(StringType &v) {
            // Size tests
            CHECK_EQ(0u, v.size());
            CHECK(v.empty());
            // Iterator tests
            CHECK_EQ(v.begin(), v.end());
        }
    };

    // New string test.
    /*
    TEST_CASE_FIXTURE(SmallStringTest, "EmptyStringTest") {
    SCOPED_TRACE("EmptyStringTest");
    assertEmpty(theString);
    CHECK(theString.rbegin() == theString.rend());

}*/

    TEST_CASE_FIXTURE(SmallStringTest, "AssignRepeated") {
        theString.assign(3, 'a');
        CHECK_EQ(3u, theString.size());
        CHECK_STREQ("aaa", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "AssignIterPair") {
        std::string_view abc = "abc";
        theString.assign(abc.begin(), abc.end());
        CHECK_EQ(3u, theString.size());
        CHECK_STREQ("abc", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "AssignStringRef") {
        std::string_view abc = "abc";
        theString.assign(abc);
        CHECK_EQ(3u, theString.size());
        CHECK_STREQ("abc", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "AssignInlinedVector") {
        std::string_view abc = "abc";
        InlinedVector<char, 10> abcVec(abc.begin(), abc.end());
        theString.assign(abcVec);
        CHECK_EQ(3u, theString.size());
        CHECK_STREQ("abc", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "AssignStringRefs") {
        theString.assign({"abc", "def", "ghi"});
        CHECK_EQ(9u, theString.size());
        CHECK_STREQ("abcdefghi", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "AppendIterPair") {
        std::string_view abc = "abc";
        theString.append(abc.begin(), abc.end());
        theString.append(abc.begin(), abc.end());
        CHECK_EQ(6u, theString.size());
        CHECK_STREQ("abcabc", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "AppendStringRef") {
        std::string_view abc = "abc";
        theString.append(abc);
        theString.append(abc);
        CHECK_EQ(6u, theString.size());
        CHECK_STREQ("abcabc", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "AppendInlinedVector") {
        std::string_view abc = "abc";
        InlinedVector<char, 10> abcVec(abc.begin(), abc.end());
        theString.append(abcVec);
        theString.append(abcVec);
        CHECK_EQ(6u, theString.size());
        CHECK_STREQ("abcabc", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "AppendStringRefs") {
        theString.append({"abc", "def", "ghi"});
        CHECK_EQ(9u, theString.size());
        CHECK_STREQ("abcdefghi", theString.c_str());
        std::string_view Jkl = "jkl";
        std::string Mno = "mno";
        InlinedString<4> Pqr("pqr");
        const char *Stu = "stu";
        theString.append({Jkl, Mno, Pqr, Stu});
        CHECK_EQ(21u, theString.size());
        CHECK_STREQ("abcdefghijklmnopqrstu", theString.c_str());
    }

    TEST_CASE_FIXTURE(SmallStringTest, "StringRefConversion") {
        std::string_view abc = "abc";
        theString.assign(abc.begin(), abc.end());
        std::string_view theStringRef = theString;
        CHECK_EQ("abc", theStringRef);
    }

    TEST_CASE_FIXTURE(SmallStringTest, "StdStringConversion") {
        std::string_view abc = "abc";
        theString.assign(abc.begin(), abc.end());
        std::string theStdString = std::string(theString);
        CHECK_EQ("abc", theStdString);
    }

    TEST_CASE_FIXTURE(SmallStringTest, "Substr") {
        theString = "hello";
        CHECK_EQ("lo", theString.substr(3));
        CHECK_EQ("", theString.substr(100));
        CHECK_EQ("hello", theString.substr(0, 100));
        CHECK_EQ("o", theString.substr(4, 10));
    }

    TEST_CASE_FIXTURE(SmallStringTest, "Slice") {
        theString = "hello";
        CHECK_EQ("l", theString.slice(2, 3));
        CHECK_EQ("ell", theString.slice(1, 4));
        CHECK_EQ("llo", theString.slice(2, 100));
        CHECK_EQ("", theString.slice(2, 1));
        CHECK_EQ("", theString.slice(10, 20));
    }

    TEST_CASE_FIXTURE(SmallStringTest, "Find") {
        theString = "hello";
        CHECK_EQ(2U, theString.find('l'));
        CHECK_EQ(std::string_view::npos, theString.find('z'));
        CHECK_EQ(std::string_view::npos, theString.find("helloworld"));
        CHECK_EQ(0U, theString.find("hello"));
        CHECK_EQ(1U, theString.find("ello"));
        CHECK_EQ(std::string_view::npos, theString.find("zz"));
        CHECK_EQ(2U, theString.find("ll", 2));
        CHECK_EQ(std::string_view::npos, theString.find("ll", 3));
        CHECK_EQ(0U, theString.find(""));

        CHECK_EQ(3U, theString.rfind('l'));
        CHECK_EQ(std::string_view::npos, theString.rfind('z'));
        CHECK_EQ(std::string_view::npos, theString.rfind("helloworld"));
        CHECK_EQ(0U, theString.rfind("hello"));
        CHECK_EQ(1U, theString.rfind("ello"));
        CHECK_EQ(std::string_view::npos, theString.rfind("zz"));

        CHECK_EQ(2U, theString.find_first_of('l'));
        CHECK_EQ(1U, theString.find_first_of("el"));
        CHECK_EQ(std::string_view::npos, theString.find_first_of("xyz"));

        CHECK_EQ(1U, theString.find_first_not_of('h'));
        CHECK_EQ(4U, theString.find_first_not_of("hel"));
        CHECK_EQ(std::string_view::npos, theString.find_first_not_of("hello"));

        theString = "hellx xello hell ello world foo bar hello";
        CHECK_EQ(36U, theString.find("hello"));
        CHECK_EQ(28U, theString.find("foo"));
        CHECK_EQ(12U, theString.find("hell", 2));
        CHECK_EQ(0U, theString.find(""));
    }

    TEST_CASE_FIXTURE(SmallStringTest, "Realloc") {
        theString = "abcd";
        theString.reserve(100);
        CHECK_EQ(std::string_view("abcd"), theString);
        unsigned const N = 100000;
        theString.reserve(N);
        for (unsigned i = 0; i < N - 4; ++i)
            theString.push_back('y');
        CHECK_EQ("abcdyyy", theString.slice(0, 7));
    }

    TEST_CASE_FIXTURE(SmallStringTest, "Comparisons") {
        CHECK_GT(0, InlinedString<10>("aab").compare("aad"));
        CHECK_EQ(0, InlinedString<10>("aab").compare("aab"));
        CHECK_LT(0, InlinedString<10>("aab").compare("aaa"));
        CHECK_GT(0, InlinedString<10>("aab").compare("aabb"));
        CHECK_LT(0, InlinedString<10>("aab").compare("aa"));
        CHECK_LT(0, InlinedString<10>("\xFF").compare("\1"));

    }

} // namespace
