// Copyright 2018 The Turbo Authors.
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

#include "turbo/container/flat_hash_set.h"

#include <vector>

#include "turbo/base/internal/raw_logging.h"
#include "hash_generator_testing.h"
#include "unordered_set_constructor_test.h"
#include "unordered_set_lookup_test.h"
#include "unordered_set_members_test.h"
#include "unordered_set_modifiers_test.h"
#include "turbo/memory/memory.h"
#include "turbo/strings/string_view.h"

namespace turbo {

    namespace container_internal {
        namespace {

            using ::turbo::container_internal::hash_internal::Enum;
            using ::turbo::container_internal::hash_internal::EnumClass;
            using ::testing::IsEmpty;
            using ::testing::Pointee;
            using ::testing::UnorderedElementsAre;
            using ::testing::UnorderedElementsAreArray;

// Check that turbo::flat_hash_set works in a global constructor.
            struct BeforeMain {
                BeforeMain() {
                    turbo::flat_hash_set<int> x;
                    x.insert(1);
                    TURBO_RAW_CHECK(!x.contains(0), "x should not contain 0");
                    TURBO_RAW_CHECK(x.contains(1), "x should contain 1");
                }
            };

            const BeforeMain before_main;

            template<class T>
            using Set =
                    turbo::flat_hash_set<T, StatefulTestingHash, StatefulTestingEqual, Alloc<T>>;

            using SetTypes =
                    ::testing::Types<Set<int>, Set<std::string>, Set<Enum>, Set<EnumClass>>;

            INSTANTIATE_TYPED_TEST_SUITE_P(FlatHashSet, ConstructorTest, SetTypes);
            INSTANTIATE_TYPED_TEST_SUITE_P(FlatHashSet, LookupTest, SetTypes);
            INSTANTIATE_TYPED_TEST_SUITE_P(FlatHashSet, MembersTest, SetTypes);
            INSTANTIATE_TYPED_TEST_SUITE_P(FlatHashSet, ModifiersTest, SetTypes);

            TEST(FlatHashSet, EmplaceString) {
                std::vector<std::string> v = {"a", "b"};
                turbo::flat_hash_set<std::string_view> hs(v.begin(), v.end());
                EXPECT_THAT(hs, UnorderedElementsAreArray(v));
            }

            TEST(FlatHashSet, BitfieldArgument) {
                union {
                    int n: 1;
                };
                n = 0;
                turbo::flat_hash_set<int> s = {n};
                s.insert(n);
                s.insert(s.end(), n);
                s.insert({n});
                s.erase(n);
                s.count(n);
                s.prefetch(n);
                s.find(n);
                s.contains(n);
                s.equal_range(n);
            }

            TEST(FlatHashSet, MergeExtractInsert) {
                struct Hash {
                    size_t operator()(const std::unique_ptr<int> &p) const { return *p; }
                };
                struct Eq {
                    bool operator()(const std::unique_ptr<int> &a,
                                    const std::unique_ptr<int> &b) const {
                        return *a == *b;
                    }
                };
                turbo::flat_hash_set<std::unique_ptr<int>, Hash, Eq> set1, set2;
                set1.insert(std::make_unique<int>(7));
                set1.insert(std::make_unique<int>(17));

                set2.insert(std::make_unique<int>(7));
                set2.insert(std::make_unique<int>(19));

                EXPECT_THAT(set1, UnorderedElementsAre(Pointee(7), Pointee(17)));
                EXPECT_THAT(set2, UnorderedElementsAre(Pointee(7), Pointee(19)));

                set1.merge(set2);

                EXPECT_THAT(set1, UnorderedElementsAre(Pointee(7), Pointee(17), Pointee(19)));
                EXPECT_THAT(set2, UnorderedElementsAre(Pointee(7)));

                auto node = set1.extract(std::make_unique<int>(7));
                EXPECT_TRUE(node);
                EXPECT_THAT(node.value(), Pointee(7));
                EXPECT_THAT(set1, UnorderedElementsAre(Pointee(17), Pointee(19)));

                auto insert_result = set2.insert(std::move(node));
                EXPECT_FALSE(node);
                EXPECT_FALSE(insert_result.inserted);
                EXPECT_TRUE(insert_result.node);
                EXPECT_THAT(insert_result.node.value(), Pointee(7));
                EXPECT_EQ(**insert_result.position, 7);
                EXPECT_NE(insert_result.position->get(), insert_result.node.value().get());
                EXPECT_THAT(set2, UnorderedElementsAre(Pointee(7)));

                node = set1.extract(std::make_unique<int>(17));
                EXPECT_TRUE(node);
                EXPECT_THAT(node.value(), Pointee(17));
                EXPECT_THAT(set1, UnorderedElementsAre(Pointee(19)));

                node.value() = std::make_unique<int>(23);

                insert_result = set2.insert(std::move(node));
                EXPECT_FALSE(node);
                EXPECT_TRUE(insert_result.inserted);
                EXPECT_FALSE(insert_result.node);
                EXPECT_EQ(**insert_result.position, 23);
                EXPECT_THAT(set2, UnorderedElementsAre(Pointee(7), Pointee(23)));
            }

            bool IsEven(int k) { return k % 2 == 0; }

            TEST(FlatHashSet, EraseIf) {
                // Erase all elements.
                {
                    flat_hash_set<int> s = {1, 2, 3, 4, 5};
                    EXPECT_EQ(erase_if(s, [](int) { return true; }), 5);
                    EXPECT_THAT(s, IsEmpty());
                }
                // Erase no elements.
                {
                    flat_hash_set<int> s = {1, 2, 3, 4, 5};
                    EXPECT_EQ(erase_if(s, [](int) { return false; }), 0);
                    EXPECT_THAT(s, UnorderedElementsAre(1, 2, 3, 4, 5));
                }
                // Erase specific elements.
                {
                    flat_hash_set<int> s = {1, 2, 3, 4, 5};
                    EXPECT_EQ(erase_if(s, [](int k) { return k % 2 == 1; }), 3);
                    EXPECT_THAT(s, UnorderedElementsAre(2, 4));
                }
                // Predicate is function reference.
                {
                    flat_hash_set<int> s = {1, 2, 3, 4, 5};
                    EXPECT_EQ(erase_if(s, IsEven), 2);
                    EXPECT_THAT(s, UnorderedElementsAre(1, 3, 5));
                }
                // Predicate is function pointer.
                {
                    flat_hash_set<int> s = {1, 2, 3, 4, 5};
                    EXPECT_EQ(erase_if(s, &IsEven), 2);
                    EXPECT_THAT(s, UnorderedElementsAre(1, 3, 5));
                }
            }

            TEST(IgnoreCase, set) {
                turbo::flat_ignore_case_hash_set<std::string> s;
                s.insert("abc");
                s.insert("ABC");
                EXPECT_EQ(s.size(), 1);
                EXPECT_TRUE(s.contains("abc"));
                EXPECT_TRUE(s.contains("ABC"));
                EXPECT_FALSE(s.contains("Abc"));
                EXPECT_FALSE(s.contains("aBc"));
                EXPECT_FALSE(s.contains("abC"));
            }

        }  // namespace
    }  // namespace container_internal

}  // namespace turbo
