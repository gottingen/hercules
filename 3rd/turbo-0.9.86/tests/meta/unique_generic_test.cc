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
// Created by jeff on 24-1-10.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

#include "turbo/meta/unique_generic.h"
#include <collie/container/algorithm.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace turbo {

    namespace {

        struct IntTraits {
            IntTraits(std::vector<int> *freed) : freed_ints(freed) {}

            static int InvalidValue() {
                return -1;
            }

            void Free(int value) {
                freed_ints->push_back(value);
            }

            std::vector<int> *freed_ints;
        };

        using ScopedInt = UniqueGeneric<int, IntTraits>;

    }  // namespace

    TEST_CASE("UniqueGenericTest, UniqueGeneric") {
        std::vector<int> values_freed;
        IntTraits traits(&values_freed);

        // Invalid case, delete should not be called.
        {
            ScopedInt a(IntTraits::InvalidValue(), traits);
        }
        REQUIRE(values_freed.empty());

        // Simple deleting case.
        static const int kFirst = 0;
        {
            ScopedInt a(kFirst, traits);
        }
        REQUIRE_EQ(1u, values_freed.size());
        REQUIRE_EQ(kFirst, values_freed[0]);
        values_freed.clear();

        // Release should return the right value and leave the object empty.
        {
            ScopedInt a(kFirst, traits);
            REQUIRE_EQ(kFirst, a.release());

            ScopedInt b(IntTraits::InvalidValue(), traits);
            REQUIRE_EQ(IntTraits::InvalidValue(), b.release());
        }
        REQUIRE(values_freed.empty());

        // Reset should free the old value, then the new one should go away when
        // it goes out of scope.
        static const int kSecond = 1;
        {
            ScopedInt b(kFirst, traits);
            b.reset(kSecond);
            REQUIRE_EQ(1u, values_freed.size());
            REQUIRE_EQ(kFirst, values_freed[0]);
        }
        REQUIRE_EQ(2u, values_freed.size());
        REQUIRE_EQ(kSecond, values_freed[1]);
        values_freed.clear();

        // Move constructor.
        {
            ScopedInt a(kFirst, traits);
            ScopedInt b(std::move(a));
            REQUIRE(values_freed.empty());  // Nothing should be freed.
            REQUIRE_EQ(IntTraits::InvalidValue(), a.get());
            REQUIRE_EQ(kFirst, b.get());
        }

        REQUIRE_EQ(1u, values_freed.size());
        REQUIRE_EQ(kFirst, values_freed[0]);
        values_freed.clear();

        // Move assign.
        {
            ScopedInt a(kFirst, traits);
            ScopedInt b(kSecond, traits);
            b = std::move(a);
            REQUIRE_EQ(1u, values_freed.size());
            REQUIRE_EQ(kSecond, values_freed[0]);
            REQUIRE_EQ(IntTraits::InvalidValue(), a.get());
            REQUIRE_EQ(kFirst, b.get());
        }

        REQUIRE_EQ(2u, values_freed.size());
        REQUIRE_EQ(kFirst, values_freed[1]);
        values_freed.clear();
    }

    TEST_CASE("UniqueGenericTest, Operators") {
        std::vector<int> values_freed;
        IntTraits traits(&values_freed);

        static const int kFirst = 0;
        static const int kSecond = 1;
        {
            ScopedInt a(kFirst, traits);
            REQUIRE(a == kFirst);
            REQUIRE_FALSE(a != kFirst);
            REQUIRE_FALSE(a == kSecond);
            REQUIRE(a != kSecond);

            REQUIRE(kFirst == a);
            REQUIRE_FALSE(kFirst != a);
            REQUIRE_FALSE(kSecond == a);
            REQUIRE(kSecond != a);
        }

        // is_valid().
        {
            ScopedInt a(kFirst, traits);
            REQUIRE(a.is_valid());
            a.

                    reset();
            REQUIRE_FALSE(a.is_valid());
        }
    }
    /*
    TEST_CASE("UniqueGenericTest, Receive") {
        std::vector<int> values_freed;
        IntTraits traits(&values_freed);
        auto a = std::make_unique<ScopedInt>(123, traits);

        REQUIRE_EQ(123, a->get());

        {
            ScopedInt::Receiver r(*a);
            REQUIRE_EQ(123, a->get());
            *r.

                    get() = 456;
            REQUIRE_EQ(123, a->get());
        }

        REQUIRE_EQ(456, a->get());

        {
            ScopedInt::Receiver r(*a);
            EXPECT_DEATH_IF_SUPPORTED(a
                                              .

                                                      reset(),

                                      "");
            EXPECT_DEATH_IF_SUPPORTED(ScopedInt::Receiver(*a)
                                              .

                                                      get(),

                                      "");
        }
    }
*/
    namespace {

        struct TrackedIntTraits : public UniqueGenericOwnershipTracking {
            using OwnerMap = std::unordered_map<int, const UniqueGeneric<int, TrackedIntTraits> *>;

            TrackedIntTraits(std::unordered_set<int> *freed, OwnerMap *owners)
                    : freed(freed), owners(owners) {}

            static int InvalidValue() { return -1; }

            void Free(int value) {
                auto it = owners->find(value);
                REQUIRE_EQ(owners->end(), it);

                REQUIRE_EQ(0U, freed->count(value));
                freed->insert(value);
            }

            void Acquire(const UniqueGeneric<int, TrackedIntTraits> &owner, int value) {
                auto it = owners->find(value);
                REQUIRE_EQ(owners->end(), it);
                (*owners)[value] = &owner;
            }

            void Release(const UniqueGeneric<int, TrackedIntTraits> &owner, int value) {
                auto it = owners->find(value);
                REQUIRE_NE(owners->end(), it);
                owners->erase(it);
            }

            std::unordered_set<int> *freed;
            OwnerMap *owners;
        };

        using ScopedTrackedInt = UniqueGeneric<int, TrackedIntTraits>;

    }  // namespace

    TEST_CASE("UniqueGenericTest, OwnershipTracking") {
        TrackedIntTraits::OwnerMap owners;
        std::unordered_set<int> freed;
        TrackedIntTraits traits(&freed, &owners);

#define ASSERT_OWNED(value, owner)            \
  REQUIRE(collie::contains(owners, value)); \
  REQUIRE_EQ(&owner, owners[value]);           \
  REQUIRE_FALSE(collie::contains(freed, value))

#define ASSERT_UNOWNED(value)                  \
  REQUIRE_FALSE(collie::contains(owners, value)); \
  REQUIRE_FALSE(collie::contains(freed, value))

#define ASSERT_FREED(value)                    \
  REQUIRE_FALSE(collie::contains(owners, value)); \
  REQUIRE(collie::contains(freed, value))

        // Constructor.
        {
            {
                ScopedTrackedInt a(0, traits);
                ASSERT_OWNED(0, a);
            }
            ASSERT_FREED(0);
        }

        owners.clear();
        freed.clear();

        // Reset.
        {
            ScopedTrackedInt a(0, traits);
            ASSERT_OWNED(0, a);
            a.reset(1);
            ASSERT_FREED(0);
            ASSERT_OWNED(1, a);
            a.reset();
            ASSERT_FREED(0);
            ASSERT_FREED(1);
        }

        owners.clear();
        freed.clear();

        // Release.
        {
            {
                ScopedTrackedInt a(0, traits);
                ASSERT_OWNED(0, a);
                int released = a.release();
                REQUIRE_EQ(0, released);
                ASSERT_UNOWNED(0);
            }
            ASSERT_UNOWNED(0);
        }

        owners.clear();
        freed.clear();

        // Move constructor.
        {
            ScopedTrackedInt a(0, traits);
            ASSERT_OWNED(0, a);
            {
                ScopedTrackedInt b(std::move(a));
                ASSERT_OWNED(0, b);
            }
            ASSERT_FREED(0);
        }

        owners.clear();
        freed.clear();

// Move assignment.
        {
            {
                ScopedTrackedInt a(0, traits);
                ScopedTrackedInt b(1, traits);
                ASSERT_OWNED(0, a);
                ASSERT_OWNED(1, b);
                a = std::move(b);
                ASSERT_OWNED(1, a);
                ASSERT_FREED(0);
            }
            ASSERT_FREED(1);
        }

        owners.clear();
        freed.clear();

#undef ASSERT_OWNED
#undef ASSERT_UNOWNED
#undef ASSERT_FREED
    }

// Cheesy manual "no compile" test for manually validating changes.
#if 0
    TEST(UniqueGenericTest, NoCompile) {
      // Assignment shouldn't work.
      /*{
        ScopedInt a(kFirst, traits);
        ScopedInt b(a);
      }*/

      // Comparison shouldn't work.
      /*{
        ScopedInt a(kFirst, traits);
        ScopedInt b(kFirst, traits);
        if (a == b) {
        }
      }*/

      // Implicit conversion to bool shouldn't work.
      /*{
        ScopedInt a(kFirst, traits);
        bool result = a;
      }*/
    }
#endif

}  // namespace base
