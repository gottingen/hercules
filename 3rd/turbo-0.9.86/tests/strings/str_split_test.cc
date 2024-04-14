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

#include "turbo/strings/str_split.h"

#include <deque>
#include <initializer_list>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "turbo/container/btree_map.h"
#include "turbo/container/btree_set.h"
#include "turbo/container/flat_hash_map.h"
#include "turbo/container/node_hash_map.h"
#include "turbo/platform/port.h"
#include "turbo/platform/dynamic_annotations.h"
#include "turbo/strings/numbers.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

namespace {

    TEST_CASE("split") {
        SUBCASE("TraitsTest") {
  static_assert(!turbo::strings_internal::SplitterIsConvertibleTo<int>::value,
                "");
  static_assert(
      !turbo::strings_internal::SplitterIsConvertibleTo<std::string>::value, "");
  static_assert(turbo::strings_internal::SplitterIsConvertibleTo<
                    std::vector<std::string>>::value,
                "");
  static_assert(
      !turbo::strings_internal::SplitterIsConvertibleTo<std::vector<int>>::value,
      "");
  static_assert(turbo::strings_internal::SplitterIsConvertibleTo<
                    std::vector<std::string_view>>::value,
                "");
  static_assert(turbo::strings_internal::SplitterIsConvertibleTo<
                    std::map<std::string, std::string>>::value,
                "");
  static_assert(turbo::strings_internal::SplitterIsConvertibleTo<
                    std::map<std::string_view, std::string_view>>::value,
                "");
  static_assert(!turbo::strings_internal::SplitterIsConvertibleTo<
                    std::map<int, std::string>>::value,
                "");
  static_assert(!turbo::strings_internal::SplitterIsConvertibleTo<
                    std::map<std::string, int>>::value,
                "");
}

// This tests the overall split API, which is made up of the turbo::str_split()
// function and the Delimiter objects in the turbo:: namespace.
// This TEST macro is outside of any namespace to require full specification of
// namespaces just like callers will need to use.
SUBCASE(" APIExamples") {
  {
    // Passes string delimiter. Assumes the default of by_string.
    std::vector<std::string> v = turbo::str_split("a,b,c", ",");  // NOLINT
      REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});

    // Equivalent to...
    using turbo::by_string;
    v = turbo::str_split("a,b,c", by_string(","));
      REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});

    // Equivalent to...
      //REQUIRE_EQ(turbo::str_split("a,b,c", by_string(",")),
      //          std::vector<std::string_view>{"a", "b", "c"});
  }

  {
    // Same as above, but using a single character as the delimiter.
    std::vector<std::string> v = turbo::str_split("a,b,c", ',');
      REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});

    // Equivalent to...
    using turbo::by_char;
    v = turbo::str_split("a,b,c", by_char(','));
      REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
  }

  {
    // Uses the Literal string "=>" as the delimiter.
    const std::vector<std::string> v = turbo::str_split("a=>b=>c", "=>");
      REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
  }

  {
    // The substrings are returned as string_views, eliminating copying.
    std::vector<std::string_view> v = turbo::str_split("a,b,c", ',');
      REQUIRE_EQ(v, std::vector<std::string_view>{"a", "b", "c"});
  }

  {
    // Leading and trailing empty substrings.
    std::vector<std::string> v = turbo::str_split(",a,b,c,", ',');
      REQUIRE_EQ(v, std::vector<std::string>{"", "a", "b", "c", ""});
  }

  {
    // Splits on a delimiter that is not found.
    std::vector<std::string> v = turbo::str_split("abc", ',');
      REQUIRE_EQ(v, std::vector<std::string>{"abc"});
  }

  {
    // Splits the input string into individual characters by using an empty
    // string as the delimiter.
    std::vector<std::string> v = turbo::str_split("abc", "");
    REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
  }

  {
    // Splits string data with embedded NUL characters, using NUL as the
    // delimiter. A simple delimiter of "\0" doesn't work because strlen() will
    // say that's the empty string when constructing the std::string_view
    // delimiter. Instead, a non-empty string containing NUL can be used as the
    // delimiter.
    std::string embedded_nulls("a\0b\0c", 5);
    std::string null_delim("\0", 1);
    std::vector<std::string> v = turbo::str_split(embedded_nulls, null_delim);
    REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
  }

  {
    // Stores first two split strings as the members in a std::pair.
    std::pair<std::string, std::string> p = turbo::str_split("a,b,c", ',');
    REQUIRE_EQ("a", p.first);
    REQUIRE_EQ("b", p.second);
    // "c" is omitted because std::pair can hold only two elements.
  }

  {
    // Results stored in std::set<std::string>
    std::set<std::string> v = turbo::str_split("a,b,c,a,b,c,a,b,c", ',');
    REQUIRE_EQ(v, std::set<std::string>{"a", "b", "c"});
  }

  {
    // Uses a non-const char* delimiter.
    char a[] = ",";
    char* d = a + 0;
    std::vector<std::string> v = turbo::str_split("a,b,c", d);
    REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
  }

  {
    // Results split using either of , or ;
    using turbo::by_any_char;
    std::vector<std::string> v = turbo::str_split("a,b;c", by_any_char(",;"));
    REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
  }

  {
    // Uses the skip_whitespace predicate.
    using turbo::skip_whitespace;
    std::vector<std::string> v =
        turbo::str_split(" a , ,,b,", ',', skip_whitespace());
    REQUIRE_EQ(v, std::vector<std::string>{" a ", "b"});
  }

  {
    // Uses the by_length delimiter.
    using turbo::by_length;
    std::vector<std::string> v = turbo::str_split("abcdefg", by_length(3));
    REQUIRE_EQ(v, std::vector<std::string>{"abc", "def", "g"});
  }

  {
    // Different forms of initialization / conversion.
    std::vector<std::string> v1 = turbo::str_split("a,b,c", ',');
    REQUIRE_EQ(v1, std::vector<std::string>{"a", "b", "c"});
    std::vector<std::string> v2(turbo::str_split("a,b,c", ','));
    REQUIRE_EQ(v2, std::vector<std::string>{"a", "b", "c"});
    auto v3 = std::vector<std::string>(turbo::str_split("a,b,c", ','));
    REQUIRE_EQ(v3, std::vector<std::string>{"a", "b", "c"});
    v3 = turbo::str_split("a,b,c", ',');
    REQUIRE_EQ(v3, std::vector<std::string>{"a", "b", "c"});
  }

  {
    // Results stored in a std::map.
    std::map<std::string, std::string> m = turbo::str_split("a,1,b,2,a,3", ',');
    REQUIRE_EQ(2, m.size());
    REQUIRE_EQ("3", m["a"]);
    REQUIRE_EQ("2", m["b"]);
  }

  {
    // Results stored in a std::multimap.
    std::multimap<std::string, std::string> m =
        turbo::str_split("a,1,b,2,a,3", ',');
    REQUIRE_EQ(3, m.size());
    auto it = m.find("a");
    REQUIRE_EQ("1", it->second);
    ++it;
    REQUIRE_EQ("3", it->second);
    it = m.find("b");
    REQUIRE_EQ("2", it->second);
  }

  {
    // Demonstrates use in a range-based for loop in C++11.
    std::string s = "x,x,x,x,x,x,x";
    for (std::string_view sp : turbo::str_split(s, ',')) {
      REQUIRE_EQ("x", sp);
    }
  }

  {
    // Demonstrates use with a Predicate in a range-based for loop.
    using turbo::skip_whitespace;
    std::string s = " ,x,,x,,x,x,x,,";
    for (std::string_view sp : turbo::str_split(s, ',', skip_whitespace())) {
      REQUIRE_EQ("x", sp);
    }
  }

  {
    // Demonstrates a "smart" split to std::map using two separate calls to
    // turbo::str_split. One call to split the records, and another call to split
    // the keys and values. This also uses the Limit delimiter so that the
    // std::string "a=b=c" will split to "a" -> "b=c".
    std::map<std::string, std::string> m;
    for (std::string_view sp : turbo::str_split("a=b=c,d=e,f=,g", ',')) {
      m.insert(turbo::str_split(sp, turbo::max_splits('=', 1)));
    }
    REQUIRE_EQ("b=c", m.find("a")->second);
    REQUIRE_EQ("e", m.find("d")->second);
    REQUIRE_EQ("", m.find("f")->second);
    REQUIRE_EQ("", m.find("g")->second);
  }
}

//
// Tests for SplitIterator
//

    struct {
        std::string in;
        std::vector<std::string> expect;
    } specs[] = {
            {"", {""}},
            {"foo", {"foo"}},
            {",", {"", ""}},
            {",foo", {"", "foo"}},
            {"foo,", {"foo", ""}},
            {",foo,", {"", "foo", ""}},
            {"foo,bar", {"foo", "bar"}},
    };

        TEST_CASE("SplitIterator") {
            SUBCASE(" Basics") {
                auto splitter = turbo::str_split("a,b", ',');
                auto it = splitter.begin();
                auto end = splitter.end();

                EXPECT_NE(it, end);
                REQUIRE_EQ("a", *it);  // tests dereference
                ++it;                 // tests preincrement
                EXPECT_NE(it, end);
                REQUIRE_EQ("b",
                           std::string(it->data(), it->size()));  // tests dereference as ptr
                it++;                                            // tests postincrement
                REQUIRE_EQ(it, end);
            }


            SUBCASE(" Predicate") {
                auto splitter = turbo::str_split("a,b,c", ',', Skip("b"));
                auto it = splitter.begin();
                auto end = splitter.end();

                EXPECT_NE(it, end);
                REQUIRE_EQ("a", *it);  // tests dereference
                ++it;                 // tests preincrement -- "b" should be skipped here.
                EXPECT_NE(it, end);
                REQUIRE_EQ("c",
                           std::string(it->data(), it->size()));  // tests dereference as ptr
                it++;                                            // tests postincrement
                REQUIRE_EQ(it, end);
            }

            SUBCASE(" EdgeCases") {
                // Expected input and output, assuming a delimiter of ','
                for (const auto &spec: specs) {
                    SCOPED_TRACE(spec.in);
                    auto splitter = turbo::str_split(spec.in, ',');
                    auto it = splitter.begin();
                    auto end = splitter.end();
                    for (const auto &expected: spec.expect) {
                        EXPECT_NE(it, end);
                        REQUIRE_EQ(expected, *it++);
                    }
                    REQUIRE_EQ(it, end);
                }
            }
        }
        /*
        SUBCASE("Splitter Const") {
  const auto splitter = turbo::str_split("a,b,c", ',');
  REQUIRE_EQ(splitter, ssplitter{"a", "b", "c"});
}*/
/*
SUBCASE(" EmptyAndNull") {
  // Attention: Splitting a null std::string_view is different than splitting
  // an empty std::string_view even though both string_views are considered
  // equal. This behavior is likely surprising and undesirable. However, to
  // maintain backward compatibility, there is a small "hack" in
  // str_split_internal.h that preserves this behavior. If that behavior is ever
  // changed/fixed, this test will need to be updated.
  REQUIRE_EQ(turbo::str_split(std::string_view(""), '-'), std::vector<std::string>{""});
  REQUIRE_EQ(turbo::str_split(std::string_view(), '-'), std::vector<std::string>{});
}
*/
 SUBCASE(" EqualityAsEndCondition") {
  auto splitter = turbo::str_split("a,b,c", ',');
  auto it = splitter.begin();
  auto it2 = it;

  // Increments it2 twice to point to "c" in the input text.
  ++it2;
  ++it2;
  REQUIRE_EQ("c", *it2);

  // This test uses a non-end SplitIterator as the terminating condition in a
  // for loop. This relies on SplitIterator equality for non-end SplitIterators
  // working correctly. At this point it2 points to "c", and we use that as the
  // "end" condition in this test.
  std::vector<std::string_view> v;
  for (; it != it2; ++it) {
    v.push_back(*it);
  }
  REQUIRE_EQ(v, ElementsAre("a", "b"));
}
}
//
// Tests for Splitter
//

TEST(Splitter, RangeIterators) {
  auto splitter = turbo::str_split("a,b,c", ',');
  std::vector<std::string_view> output;
  for (std::string_view p : splitter) {
    output.push_back(p);
  }
  REQUIRE_EQ(output, std::vector<std::string>{"a", "b", "c"});
}

// Some template functions for use in testing conversion operators
template <typename ContainerType, typename Splitter>
void TestConversionOperator(const Splitter& splitter) {
  ContainerType output = splitter;
  REQUIRE_EQ(output, UnorderedElementsAre("a", "b", "c", "d"));
}

template <typename MapType, typename Splitter>
void TestMapConversionOperator(const Splitter& splitter) {
  MapType m = splitter;
  REQUIRE_EQ(m, UnorderedElementsAre(Pair("a", "b"), Pair("c", "d")));
}

template <typename FirstType, typename SecondType, typename Splitter>
void TestPairConversionOperator(const Splitter& splitter) {
  std::pair<FirstType, SecondType> p = splitter;
  REQUIRE_EQ(p, (std::pair<FirstType, SecondType>("a", "b")));
}

TEST(Splitter, ConversionOperator) {
  auto splitter = turbo::str_split("a,b,c,d", ',');

  TestConversionOperator<std::vector<std::string_view>>(splitter);
  TestConversionOperator<std::vector<std::string>>(splitter);
  TestConversionOperator<std::list<std::string_view>>(splitter);
  TestConversionOperator<std::list<std::string>>(splitter);
  TestConversionOperator<std::deque<std::string_view>>(splitter);
  TestConversionOperator<std::deque<std::string>>(splitter);
  TestConversionOperator<std::set<std::string_view>>(splitter);
  TestConversionOperator<std::set<std::string>>(splitter);
  TestConversionOperator<std::multiset<std::string_view>>(splitter);
  TestConversionOperator<std::multiset<std::string>>(splitter);
  TestConversionOperator<turbo::btree_set<std::string_view>>(splitter);
  TestConversionOperator<turbo::btree_set<std::string>>(splitter);
  TestConversionOperator<turbo::btree_multiset<std::string_view>>(splitter);
  TestConversionOperator<turbo::btree_multiset<std::string>>(splitter);
  TestConversionOperator<std::unordered_set<std::string>>(splitter);

  // Tests conversion to map-like objects.

  TestMapConversionOperator<std::map<std::string_view, std::string_view>>(
      splitter);
  TestMapConversionOperator<std::map<std::string_view, std::string>>(splitter);
  TestMapConversionOperator<std::map<std::string, std::string_view>>(splitter);
  TestMapConversionOperator<std::map<std::string, std::string>>(splitter);
  TestMapConversionOperator<
      std::multimap<std::string_view, std::string_view>>(splitter);
  TestMapConversionOperator<std::multimap<std::string_view, std::string>>(
      splitter);
  TestMapConversionOperator<std::multimap<std::string, std::string_view>>(
      splitter);
  TestMapConversionOperator<std::multimap<std::string, std::string>>(splitter);
  TestMapConversionOperator<
      turbo::btree_map<std::string_view, std::string_view>>(splitter);
  TestMapConversionOperator<turbo::btree_map<std::string_view, std::string>>(
      splitter);
  TestMapConversionOperator<turbo::btree_map<std::string, std::string_view>>(
      splitter);
  TestMapConversionOperator<turbo::btree_map<std::string, std::string>>(
      splitter);
  TestMapConversionOperator<
      turbo::btree_multimap<std::string_view, std::string_view>>(splitter);
  TestMapConversionOperator<
      turbo::btree_multimap<std::string_view, std::string>>(splitter);
  TestMapConversionOperator<
      turbo::btree_multimap<std::string, std::string_view>>(splitter);
  TestMapConversionOperator<turbo::btree_multimap<std::string, std::string>>(
      splitter);
  TestMapConversionOperator<std::unordered_map<std::string, std::string>>(
      splitter);
  TestMapConversionOperator<
      turbo::node_hash_map<std::string_view, std::string_view>>(splitter);
  TestMapConversionOperator<
      turbo::node_hash_map<std::string_view, std::string>>(splitter);
  TestMapConversionOperator<
      turbo::node_hash_map<std::string, std::string_view>>(splitter);
  TestMapConversionOperator<
      turbo::flat_hash_map<std::string_view, std::string_view>>(splitter);
  TestMapConversionOperator<
      turbo::flat_hash_map<std::string_view, std::string>>(splitter);
  TestMapConversionOperator<
      turbo::flat_hash_map<std::string, std::string_view>>(splitter);

  // Tests conversion to std::pair

  TestPairConversionOperator<std::string_view, std::string_view>(splitter);
  TestPairConversionOperator<std::string_view, std::string>(splitter);
  TestPairConversionOperator<std::string, std::string_view>(splitter);
  TestPairConversionOperator<std::string, std::string>(splitter);
}

// A few additional tests for conversion to std::pair. This conversion is
// different from others because a std::pair always has exactly two elements:
// .first and .second. The split has to work even when the split has
// less-than, equal-to, and more-than 2 strings.
TEST(Splitter, ToPair) {
  {
    // Empty string
    std::pair<std::string, std::string> p = turbo::str_split("", ',');
    REQUIRE_EQ("", p.first);
    REQUIRE_EQ("", p.second);
  }

  {
    // Only first
    std::pair<std::string, std::string> p = turbo::str_split("a", ',');
    REQUIRE_EQ("a", p.first);
    REQUIRE_EQ("", p.second);
  }

  {
    // Only second
    std::pair<std::string, std::string> p = turbo::str_split(",b", ',');
    REQUIRE_EQ("", p.first);
    REQUIRE_EQ("b", p.second);
  }

  {
    // First and second.
    std::pair<std::string, std::string> p = turbo::str_split("a,b", ',');
    REQUIRE_EQ("a", p.first);
    REQUIRE_EQ("b", p.second);
  }

  {
    // First and second and then more stuff that will be ignored.
    std::pair<std::string, std::string> p = turbo::str_split("a,b,c", ',');
    REQUIRE_EQ("a", p.first);
    REQUIRE_EQ("b", p.second);
    // "c" is omitted.
  }
}

TEST(Splitter, Predicates) {
  static const char kTestChars[] = ",a, ,b,";
  using turbo::allow_empty;
  using turbo::skip_empty;
  using turbo::skip_whitespace;

  {
    // No predicate. Does not skip empties.
    auto splitter = turbo::str_split(kTestChars, ',');
    std::vector<std::string> v = splitter;
    REQUIRE_EQ(v, ElementsAre("", "a", " ", "b", ""));
  }

  {
    // Allows empty strings. Same behavior as no predicate at all.
    auto splitter = turbo::str_split(kTestChars, ',', allow_empty());
    std::vector<std::string> v_allowempty = splitter;
    REQUIRE_EQ(v_allowempty, ElementsAre("", "a", " ", "b", ""));

    // Ensures allow_empty equals the behavior with no predicate.
    auto splitter_nopredicate = turbo::str_split(kTestChars, ',');
    std::vector<std::string> v_nopredicate = splitter_nopredicate;
    REQUIRE_EQ(v_allowempty, v_nopredicate);
  }

  {
    // Skips empty strings.
    auto splitter = turbo::str_split(kTestChars, ',', skip_empty());
    std::vector<std::string> v = splitter;
    REQUIRE_EQ(v, ElementsAre("a", " ", "b"));
  }

  {
    // Skips empty and all-whitespace strings.
    auto splitter = turbo::str_split(kTestChars, ',', skip_whitespace());
    std::vector<std::string> v = splitter;
    REQUIRE_EQ(v, ElementsAre("a", "b"));
  }
}

//
// Tests for str_split()
//

SUBCASE(" Basics) {
  {
    // Doesn't really do anything useful because the return value is ignored,
    // but it should work.
    turbo::str_split("a,b,c", ',');
  }

  {
    std::vector<std::string_view> v = turbo::str_split("a,b,c", ',');
    REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
  }

  {
    std::vector<std::string> v = turbo::str_split("a,b,c", ',');
    REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
  }

  {
    // Ensures that assignment works. This requires a little extra work with
    // C++11 because of overloads with initializer_list.
    std::vector<std::string> v;
    v = turbo::str_split("a,b,c", ',');

    REQUIRE_EQ(v, std::vector<std::string>{"a", "b", "c"});
    std::map<std::string, std::string> m;
    m = turbo::str_split("a,b,c", ',');
    REQUIRE_EQ(2, m.size());
    std::unordered_map<std::string, std::string> hm;
    hm = turbo::str_split("a,b,c", ',');
    REQUIRE_EQ(2, hm.size());
  }
}

std::string_view ReturnStringView() { return "Hello World"; }
const char* ReturnConstCharP() { return "Hello World"; }
char* ReturnCharP() { return const_cast<char*>("Hello World"); }

SUBCASE(" AcceptsCertainTemporaries") {
  std::vector<std::string> v;
  v = turbo::str_split(ReturnStringView(), ' ');
  REQUIRE_EQ(v, ElementsAre("Hello", "World"));
  v = turbo::str_split(ReturnConstCharP(), ' ');
  REQUIRE_EQ(v, ElementsAre("Hello", "World"));
  v = turbo::str_split(ReturnCharP(), ' ');
  REQUIRE_EQ(v, ElementsAre("Hello", "World"));
}

SUBCASE(" Temporary") {
  // Use a std::string longer than the SSO length, so that when the temporary is
  // destroyed, if the splitter keeps a reference to the string's contents,
  // it'll reference freed memory instead of just dead on-stack memory.
  const char input[] = "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u";
  EXPECT_LT(sizeof(std::string), TURBO_ARRAY_SIZE(input))
      << "Input should be larger than fits on the stack.";

  // This happens more often in C++11 as part of a range-based for loop.
  auto splitter = turbo::str_split(std::string(input), ',');
  std::string expected = "a";
  for (std::string_view letter : splitter) {
    REQUIRE_EQ(expected, letter);
    ++expected[0];
  }
  REQUIRE_EQ("v", expected);

  // This happens more often in C++11 as part of a range-based for loop.
  auto std_splitter = turbo::str_split(std::string(input), ',');
  expected = "a";
  for (std::string_view letter : std_splitter) {
    REQUIRE_EQ(expected, letter);
    ++expected[0];
  }
  REQUIRE_EQ("v", expected);
}

template <typename T>
static std::unique_ptr<T> CopyToHeap(const T& value) {
  return std::unique_ptr<T>(new T(value));
}

SUBCASE(" LvalueCaptureIsCopyable") {
  std::string input = "a,b";
  auto heap_splitter = CopyToHeap(turbo::str_split(input, ','));
  auto stack_splitter = *heap_splitter;
  heap_splitter.reset();
  std::vector<std::string> result = stack_splitter;
  REQUIRE_EQ(result, testing::ElementsAre("a", "b"));
}

SUBCASE(" TemporaryCaptureIsCopyable") {
  auto heap_splitter = CopyToHeap(turbo::str_split(std::string("a,b"), ','));
  auto stack_splitter = *heap_splitter;
  heap_splitter.reset();
  std::vector<std::string> result = stack_splitter;
  REQUIRE_EQ(result, testing::ElementsAre("a", "b"));
}

SUBCASE(" SplitterIsCopyableAndMoveable") {
  auto a = turbo::str_split("foo", '-');

  // Ensures that the following expressions compile.
  auto b = a;             // Copy construct
  auto c = std::move(a);  // Move construct
  b = c;                  // Copy assign
  c = std::move(b);       // Move assign

  REQUIRE_EQ(c, ElementsAre("foo"));
}

SUBCASE(" StringDelimiter") {
  {
    std::vector<std::string_view> v = turbo::str_split("a,b", ',');
    REQUIRE_EQ(v, ElementsAre("a", "b"));
  }

  {
    std::vector<std::string_view> v = turbo::str_split("a,b", std::string(","));
    REQUIRE_EQ(v, ElementsAre("a", "b"));
  }

  {
    std::vector<std::string_view> v =
        turbo::str_split("a,b", std::string_view(","));
    REQUIRE_EQ(v, ElementsAre("a", "b"));
  }
}

#if !defined(__cpp_char8_t)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++2a-compat"
#endif
SUBCASE(" UTF8") {
  // Tests splitting utf8 strings and utf8 delimiters.
  std::string utf8_string = u8"\u03BA\u1F79\u03C3\u03BC\u03B5";
  {
    // A utf8 input string with an ascii delimiter.
    std::string to_split = "a," + utf8_string;
    std::vector<std::string_view> v = turbo::str_split(to_split, ',');
    REQUIRE_EQ(v, ElementsAre("a", utf8_string));
  }

  {
    // A utf8 input string and a utf8 delimiter.
    std::string to_split = "a," + utf8_string + ",b";
    std::string unicode_delimiter = "," + utf8_string + ",";
    std::vector<std::string_view> v =
        turbo::str_split(to_split, unicode_delimiter);
    REQUIRE_EQ(v, ElementsAre("a", "b"));
  }

  {
    // A utf8 input string and by_any_char with ascii chars.
    std::vector<std::string_view> v =
        turbo::str_split(u8"Foo h\u00E4llo th\u4E1Ere", turbo::by_any_char(" \t"));
    REQUIRE_EQ(v, ElementsAre("Foo", u8"h\u00E4llo", u8"th\u4E1Ere"));
  }
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif  // !defined(__cpp_char8_t)

SUBCASE(" EmptyStringDelimiter") {
  {
    std::vector<std::string> v = turbo::str_split("", "");
    REQUIRE_EQ(v, ElementsAre(""));
  }

  {
    std::vector<std::string> v = turbo::str_split("a", "");
    REQUIRE_EQ(v, ElementsAre("a"));
  }

  {
    std::vector<std::string> v = turbo::str_split("ab", "");
    REQUIRE_EQ(v, ElementsAre("a", "b"));
  }

  {
    std::vector<std::string> v = turbo::str_split("a b", "");
    REQUIRE_EQ(v, ElementsAre("a", " ", "b"));
  }
}

SUBCASE(" SubstrDelimiter") {
  std::vector<std::string_view> results;
  std::string_view delim("//");

  results = turbo::str_split("", delim);
  REQUIRE_EQ(results, ElementsAre(""));

  results = turbo::str_split("//", delim);
  REQUIRE_EQ(results, ElementsAre("", ""));

  results = turbo::str_split("ab", delim);
  REQUIRE_EQ(results, ElementsAre("ab"));

  results = turbo::str_split("ab//", delim);
  REQUIRE_EQ(results, ElementsAre("ab", ""));

  results = turbo::str_split("ab/", delim);
  REQUIRE_EQ(results, ElementsAre("ab/"));

  results = turbo::str_split("a/b", delim);
  REQUIRE_EQ(results, ElementsAre("a/b"));

  results = turbo::str_split("a//b", delim);
  REQUIRE_EQ(results, ElementsAre("a", "b"));

  results = turbo::str_split("a///b", delim);
  REQUIRE_EQ(results, ElementsAre("a", "/b"));

  results = turbo::str_split("a////b", delim);
  REQUIRE_EQ(results, ElementsAre("a", "", "b"));
}

SUBCASE(" EmptyResults") {
  std::vector<std::string_view> results;

  results = turbo::str_split("", '#');
  REQUIRE_EQ(results, ElementsAre(""));

  results = turbo::str_split("#", '#');
  REQUIRE_EQ(results, ElementsAre("", ""));

  results = turbo::str_split("#cd", '#');
  REQUIRE_EQ(results, ElementsAre("", "cd"));

  results = turbo::str_split("ab#cd#", '#');
  REQUIRE_EQ(results, ElementsAre("ab", "cd", ""));

  results = turbo::str_split("ab##cd", '#');
  REQUIRE_EQ(results, ElementsAre("ab", "", "cd"));

  results = turbo::str_split("ab##", '#');
  REQUIRE_EQ(results, ElementsAre("ab", "", ""));

  results = turbo::str_split("ab#ab#", '#');
  REQUIRE_EQ(results, ElementsAre("ab", "ab", ""));

  results = turbo::str_split("aaaa", 'a');
  REQUIRE_EQ(results, ElementsAre("", "", "", "", ""));

  results = turbo::str_split("", '#', turbo::skip_empty());
  REQUIRE_EQ(results, ElementsAre());
}

template <typename Delimiter>
static bool IsFoundAtStartingPos(std::string_view text, Delimiter d,
                                 size_t starting_pos, int expected_pos) {
  std::string_view found = d.Find(text, starting_pos);
  return found.data() != text.data() + text.size() &&
         expected_pos == found.data() - text.data();
}

// Helper function for testing Delimiter objects. Returns true if the given
// Delimiter is found in the given string at the given position. This function
// tests two cases:
//   1. The actual text given, staring at position 0
//   2. The text given with leading padding that should be ignored
template <typename Delimiter>
static bool IsFoundAt(std::string_view text, Delimiter d, int expected_pos) {
  const std::string leading_text = ",x,y,z,";
  return IsFoundAtStartingPos(text, d, 0, expected_pos) &&
         IsFoundAtStartingPos(leading_text + std::string(text), d,
                              leading_text.length(),
                              expected_pos + leading_text.length());
}

//
// Tests for by_string
//

// Tests using any delimiter that represents a single comma.
template <typename Delimiter>
void TestComma(Delimiter d) {
  EXPECT_TRUE(IsFoundAt(",", d, 0));
  EXPECT_TRUE(IsFoundAt("a,", d, 1));
  EXPECT_TRUE(IsFoundAt(",b", d, 0));
  EXPECT_TRUE(IsFoundAt("a,b", d, 1));
  EXPECT_TRUE(IsFoundAt("a,b,", d, 1));
  EXPECT_TRUE(IsFoundAt("a,b,c", d, 1));
  EXPECT_FALSE(IsFoundAt("", d, -1));
  EXPECT_FALSE(IsFoundAt(" ", d, -1));
  EXPECT_FALSE(IsFoundAt("a", d, -1));
  EXPECT_FALSE(IsFoundAt("a b c", d, -1));
  EXPECT_FALSE(IsFoundAt("a;b;c", d, -1));
  EXPECT_FALSE(IsFoundAt(";", d, -1));
}

TEST(Delimiter, by_string) {
  using turbo::by_string;
  TestComma(by_string(","));

  // Works as named variable.
  by_string comma_string(",");
  TestComma(comma_string);

  // The first occurrence of empty string ("") in a string is at position 0.
  // There is a test below that demonstrates this for std::string_view::find().
  // If the by_string delimiter returned position 0 for this, there would
  // be an infinite loop in the SplitIterator code. To avoid this, empty string
  // is a special case in that it always returns the item at position 1.
  std::string_view abc("abc");
  REQUIRE_EQ(0, abc.find(""));  // "" is found at position 0
  by_string empty("");
  EXPECT_FALSE(IsFoundAt("", empty, 0));
  EXPECT_FALSE(IsFoundAt("a", empty, 0));
  EXPECT_TRUE(IsFoundAt("ab", empty, 1));
  EXPECT_TRUE(IsFoundAt("abc", empty, 1));
}

SUBCASE(" by_char") {
  using turbo::by_char;
  TestComma(by_char(','));

  // Works as named variable.
  by_char comma_char(',');
  TestComma(comma_char);
}

//
// Tests for by_any_char
//

TEST(Delimiter, by_any_char) {
  using turbo::by_any_char;
  by_any_char one_delim(",");
  // Found
  EXPECT_TRUE(IsFoundAt(",", one_delim, 0));
  EXPECT_TRUE(IsFoundAt("a,", one_delim, 1));
  EXPECT_TRUE(IsFoundAt("a,b", one_delim, 1));
  EXPECT_TRUE(IsFoundAt(",b", one_delim, 0));
  // Not found
  EXPECT_FALSE(IsFoundAt("", one_delim, -1));
  EXPECT_FALSE(IsFoundAt(" ", one_delim, -1));
  EXPECT_FALSE(IsFoundAt("a", one_delim, -1));
  EXPECT_FALSE(IsFoundAt("a;b;c", one_delim, -1));
  EXPECT_FALSE(IsFoundAt(";", one_delim, -1));

  by_any_char two_delims(",;");
  // Found
  EXPECT_TRUE(IsFoundAt(",", two_delims, 0));
  EXPECT_TRUE(IsFoundAt(";", two_delims, 0));
  EXPECT_TRUE(IsFoundAt(",;", two_delims, 0));
  EXPECT_TRUE(IsFoundAt(";,", two_delims, 0));
  EXPECT_TRUE(IsFoundAt(",;b", two_delims, 0));
  EXPECT_TRUE(IsFoundAt(";,b", two_delims, 0));
  EXPECT_TRUE(IsFoundAt("a;,", two_delims, 1));
  EXPECT_TRUE(IsFoundAt("a,;", two_delims, 1));
  EXPECT_TRUE(IsFoundAt("a;,b", two_delims, 1));
  EXPECT_TRUE(IsFoundAt("a,;b", two_delims, 1));
  // Not found
  EXPECT_FALSE(IsFoundAt("", two_delims, -1));
  EXPECT_FALSE(IsFoundAt(" ", two_delims, -1));
  EXPECT_FALSE(IsFoundAt("a", two_delims, -1));
  EXPECT_FALSE(IsFoundAt("a=b=c", two_delims, -1));
  EXPECT_FALSE(IsFoundAt("=", two_delims, -1));

  // by_any_char behaves just like by_string when given a delimiter of empty
  // string. That is, it always returns a zero-length std::string_view
  // referring to the item at position 1, not position 0.
  by_any_char empty("");
  EXPECT_FALSE(IsFoundAt("", empty, 0));
  EXPECT_FALSE(IsFoundAt("a", empty, 0));
  EXPECT_TRUE(IsFoundAt("ab", empty, 1));
  EXPECT_TRUE(IsFoundAt("abc", empty, 1));
}

//
// Tests for by_length
//

TEST(Delimiter, by_length) {
  using turbo::by_length;

  by_length four_char_delim(4);

  // Found
  EXPECT_TRUE(IsFoundAt("abcde", four_char_delim, 4));
  EXPECT_TRUE(IsFoundAt("abcdefghijklmnopqrstuvwxyz", four_char_delim, 4));
  EXPECT_TRUE(IsFoundAt("a b,c\nd", four_char_delim, 4));
  // Not found
  EXPECT_FALSE(IsFoundAt("", four_char_delim, 0));
  EXPECT_FALSE(IsFoundAt("a", four_char_delim, 0));
  EXPECT_FALSE(IsFoundAt("ab", four_char_delim, 0));
  EXPECT_FALSE(IsFoundAt("abc", four_char_delim, 0));
  EXPECT_FALSE(IsFoundAt("abcd", four_char_delim, 0));
}

SUBCASE(" WorksWithLargeStrings") {
#if defined(TURBO_HAVE_ADDRESS_SANITIZER) || \
    defined(TURBO_HAVE_MEMORY_SANITIZER) || defined(TURBO_HAVE_THREAD_SANITIZER)
  constexpr size_t kSize = (uint32_t{1} << 26) + 1;  // 64M + 1 byte
#else
  constexpr size_t kSize = (uint32_t{1} << 31) + 1;  // 2G + 1 byte
#endif
  if (sizeof(size_t) > 4) {
    std::string s(kSize, 'x');
    s.back() = '-';
    std::vector<std::string_view> v = turbo::str_split(s, '-');
    REQUIRE_EQ(2, v.size());
    // The first element will contain 2G of 'x's.
    // testing::starts_with is too slow with a 2G string.
    REQUIRE_EQ('x', v[0][0]);
    REQUIRE_EQ('x', v[0][1]);
    REQUIRE_EQ('x', v[0][3]);
    REQUIRE_EQ("", v[1]);
  }
}
}
TEST(SplitInternalTest, TypeTraits) {
  EXPECT_FALSE(turbo::strings_internal::HasMappedType<int>::value);
  EXPECT_TRUE(
      (turbo::strings_internal::HasMappedType<std::map<int, int>>::value));
  EXPECT_FALSE(turbo::strings_internal::HasValueType<int>::value);
  EXPECT_TRUE(
      (turbo::strings_internal::HasValueType<std::map<int, int>>::value));
  EXPECT_FALSE(turbo::strings_internal::HasConstIterator<int>::value);
  EXPECT_TRUE(
      (turbo::strings_internal::HasConstIterator<std::map<int, int>>::value));
  EXPECT_FALSE(turbo::strings_internal::IsInitializerList<int>::value);
  EXPECT_TRUE((turbo::strings_internal::IsInitializerList<
               std::initializer_list<int>>::value));
}

}  // namespace
