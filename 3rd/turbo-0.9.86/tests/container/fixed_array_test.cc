// Copyright 2019 The Turbo Authors.
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

#include "turbo/container/fixed_array.h"

#include <stdio.h>

#include <cstring>
#include <list>
#include <memory>
#include <numeric>
#include <scoped_allocator>
#include <stdexcept>
#include <string>
#include <vector>

#include "../base/exception_testing.h"
#include "turbo/container/internal/counting_allocator.h"
#include "tests/hash/hash_testing.h"
#include "turbo/memory/memory.h"
#include "turbo/platform/port.h"
#include "turbo/platform/options.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAreArray;

namespace {

// Helper routine to determine if a turbo::FixedArray used stack allocation.
template <typename ArrayType>
static bool IsOnStack(const ArrayType& a) {
  return a.size() <= ArrayType::inline_elements;
}

class ConstructionTester {
 public:
  ConstructionTester() : self_ptr_(this), value_(0) { constructions++; }
  ~ConstructionTester() {
    assert(self_ptr_ == this);
    self_ptr_ = nullptr;
    destructions++;
  }

  // These are incremented as elements are constructed and destructed so we can
  // be sure all elements are properly cleaned up.
  static int constructions;
  static int destructions;

  void CheckConstructed() { assert(self_ptr_ == this); }

  void set(int value) { value_ = value; }
  int get() { return value_; }

 private:
  // self_ptr_ should always point to 'this' -- that's how we can be sure the
  // constructor has been called.
  ConstructionTester* self_ptr_;
  int value_;
};

int ConstructionTester::constructions = 0;
int ConstructionTester::destructions = 0;

// ThreeInts will initialize its three ints to the value stored in
// ThreeInts::counter. The constructor increments counter so that each object
// in an array of ThreeInts will have different values.
class ThreeInts {
 public:
  ThreeInts() {
    x_ = counter;
    y_ = counter;
    z_ = counter;
    ++counter;
  }

  static int counter;

  int x_, y_, z_;
};

int ThreeInts::counter = 0;

TEST(FixedArrayTest, CopyCtor) {
  turbo::FixedArray<int, 10> on_stack(5);
  std::iota(on_stack.begin(), on_stack.end(), 0);
  turbo::FixedArray<int, 10> stack_copy = on_stack;
  EXPECT_THAT(stack_copy, ElementsAreArray(on_stack));
  EXPECT_TRUE(IsOnStack(stack_copy));

  turbo::FixedArray<int, 10> allocated(15);
  std::iota(allocated.begin(), allocated.end(), 0);
  turbo::FixedArray<int, 10> alloced_copy = allocated;
  EXPECT_THAT(alloced_copy, ElementsAreArray(allocated));
  EXPECT_FALSE(IsOnStack(alloced_copy));
}

TEST(FixedArrayTest, MoveCtor) {
  turbo::FixedArray<std::unique_ptr<int>, 10> on_stack(5);
  for (int i = 0; i < 5; ++i) {
    on_stack[i] = std::make_unique<int>(i);
  }

  turbo::FixedArray<std::unique_ptr<int>, 10> stack_copy = std::move(on_stack);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(*(stack_copy[i]), i);
  EXPECT_EQ(stack_copy.size(), on_stack.size());

  turbo::FixedArray<std::unique_ptr<int>, 10> allocated(15);
  for (int i = 0; i < 15; ++i) {
    allocated[i] = std::make_unique<int>(i);
  }

  turbo::FixedArray<std::unique_ptr<int>, 10> alloced_copy =
      std::move(allocated);
  for (int i = 0; i < 15; ++i) EXPECT_EQ(*(alloced_copy[i]), i);
  EXPECT_EQ(allocated.size(), alloced_copy.size());
}

TEST(FixedArrayTest, SmallObjects) {
  // Small object arrays
  {
    // Short arrays should be on the stack
    turbo::FixedArray<int> array(4);
    EXPECT_TRUE(IsOnStack(array));
  }

  {
    // Large arrays should be on the heap
    turbo::FixedArray<int> array(1048576);
    EXPECT_FALSE(IsOnStack(array));
  }

  {
    // Arrays of <= default size should be on the stack
    turbo::FixedArray<int, 100> array(100);
    EXPECT_TRUE(IsOnStack(array));
  }

  {
    // Arrays of > default size should be on the heap
    turbo::FixedArray<int, 100> array(101);
    EXPECT_FALSE(IsOnStack(array));
  }

  {
    // Arrays with different size elements should use approximately
    // same amount of stack space
    turbo::FixedArray<int> array1(0);
    turbo::FixedArray<char> array2(0);
    EXPECT_LE(sizeof(array1), sizeof(array2) + 100);
    EXPECT_LE(sizeof(array2), sizeof(array1) + 100);
  }

  {
    // Ensure that vectors are properly constructed inside a fixed array.
    turbo::FixedArray<std::vector<int>> array(2);
    EXPECT_EQ(0, array[0].size());
    EXPECT_EQ(0, array[1].size());
  }

  {
    // Regardless of turbo::FixedArray implementation, check that a type with a
    // low alignment requirement and a non power-of-two size is initialized
    // correctly.
    ThreeInts::counter = 1;
    turbo::FixedArray<ThreeInts> array(2);
    EXPECT_EQ(1, array[0].x_);
    EXPECT_EQ(1, array[0].y_);
    EXPECT_EQ(1, array[0].z_);
    EXPECT_EQ(2, array[1].x_);
    EXPECT_EQ(2, array[1].y_);
    EXPECT_EQ(2, array[1].z_);
  }
}

TEST(FixedArrayTest, AtThrows) {
  turbo::FixedArray<int> a = {1, 2, 3};
  EXPECT_EQ(a.at(2), 3);
  TURBO_BASE_INTERNAL_EXPECT_FAIL(a.at(3), std::out_of_range,
                                 "failed bounds check");
}

TEST(FixedArrayTest, Hardened) {
#if !defined(NDEBUG) || TURBO_OPTION_HARDENED
  turbo::FixedArray<int> a = {1, 2, 3};
  EXPECT_EQ(a[2], 3);
  EXPECT_DEATH_IF_SUPPORTED(a[3], "");
  EXPECT_DEATH_IF_SUPPORTED(a[-1], "");

  turbo::FixedArray<int> empty(0);
  EXPECT_DEATH_IF_SUPPORTED(empty[0], "");
  EXPECT_DEATH_IF_SUPPORTED(empty[-1], "");
  EXPECT_DEATH_IF_SUPPORTED(empty.front(), "");
  EXPECT_DEATH_IF_SUPPORTED(empty.back(), "");
#endif
}

TEST(FixedArrayRelationalsTest, EqualArrays) {
  for (int i = 0; i < 10; ++i) {
    turbo::FixedArray<int, 5> a1(i);
    std::iota(a1.begin(), a1.end(), 0);
    turbo::FixedArray<int, 5> a2(a1.begin(), a1.end());

    EXPECT_TRUE(a1 == a2);
    EXPECT_FALSE(a1 != a2);
    EXPECT_TRUE(a2 == a1);
    EXPECT_FALSE(a2 != a1);
    EXPECT_FALSE(a1 < a2);
    EXPECT_FALSE(a1 > a2);
    EXPECT_FALSE(a2 < a1);
    EXPECT_FALSE(a2 > a1);
    EXPECT_TRUE(a1 <= a2);
    EXPECT_TRUE(a1 >= a2);
    EXPECT_TRUE(a2 <= a1);
    EXPECT_TRUE(a2 >= a1);
  }
}

TEST(FixedArrayRelationalsTest, UnequalArrays) {
  for (int i = 1; i < 10; ++i) {
    turbo::FixedArray<int, 5> a1(i);
    std::iota(a1.begin(), a1.end(), 0);
    turbo::FixedArray<int, 5> a2(a1.begin(), a1.end());
    --a2[i / 2];

    EXPECT_FALSE(a1 == a2);
    EXPECT_TRUE(a1 != a2);
    EXPECT_FALSE(a2 == a1);
    EXPECT_TRUE(a2 != a1);
    EXPECT_FALSE(a1 < a2);
    EXPECT_TRUE(a1 > a2);
    EXPECT_TRUE(a2 < a1);
    EXPECT_FALSE(a2 > a1);
    EXPECT_FALSE(a1 <= a2);
    EXPECT_TRUE(a1 >= a2);
    EXPECT_TRUE(a2 <= a1);
    EXPECT_FALSE(a2 >= a1);
  }
}

template <int stack_elements>
static void TestArray(int n) {
  SCOPED_TRACE(n);
  SCOPED_TRACE(stack_elements);
  ConstructionTester::constructions = 0;
  ConstructionTester::destructions = 0;
  {
    turbo::FixedArray<ConstructionTester, stack_elements> array(n);

    EXPECT_THAT(array.size(), n);
    EXPECT_THAT(array.memsize(), sizeof(ConstructionTester) * n);
    EXPECT_THAT(array.begin() + n, array.end());

    // Check that all elements were constructed
    for (int i = 0; i < n; i++) {
      array[i].CheckConstructed();
    }
    // Check that no other elements were constructed
    EXPECT_THAT(ConstructionTester::constructions, n);

    // Test operator[]
    for (int i = 0; i < n; i++) {
      array[i].set(i);
    }
    for (int i = 0; i < n; i++) {
      EXPECT_THAT(array[i].get(), i);
      EXPECT_THAT(array.data()[i].get(), i);
    }

    // Test data()
    for (int i = 0; i < n; i++) {
      array.data()[i].set(i + 1);
    }
    for (int i = 0; i < n; i++) {
      EXPECT_THAT(array[i].get(), i + 1);
      EXPECT_THAT(array.data()[i].get(), i + 1);
    }
  }  // Close scope containing 'array'.

  // Check that all constructed elements were destructed.
  EXPECT_EQ(ConstructionTester::constructions,
            ConstructionTester::destructions);
}

template <int elements_per_inner_array, int inline_elements>
static void TestArrayOfArrays(int n) {
  SCOPED_TRACE(n);
  SCOPED_TRACE(inline_elements);
  SCOPED_TRACE(elements_per_inner_array);
  ConstructionTester::constructions = 0;
  ConstructionTester::destructions = 0;
  {
    using InnerArray = ConstructionTester[elements_per_inner_array];
    // Heap-allocate the FixedArray to avoid blowing the stack frame.
    auto array_ptr =
        std::make_unique<turbo::FixedArray<InnerArray, inline_elements>>(n);
    auto& array = *array_ptr;

    ASSERT_EQ(array.size(), n);
    ASSERT_EQ(array.memsize(),
              sizeof(ConstructionTester) * elements_per_inner_array * n);
    ASSERT_EQ(array.begin() + n, array.end());

    // Check that all elements were constructed
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < elements_per_inner_array; j++) {
        (array[i])[j].CheckConstructed();
      }
    }
    // Check that no other elements were constructed
    ASSERT_EQ(ConstructionTester::constructions, n * elements_per_inner_array);

    // Test operator[]
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < elements_per_inner_array; j++) {
        (array[i])[j].set(i * elements_per_inner_array + j);
      }
    }
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < elements_per_inner_array; j++) {
        ASSERT_EQ((array[i])[j].get(), i * elements_per_inner_array + j);
        ASSERT_EQ((array.data()[i])[j].get(), i * elements_per_inner_array + j);
      }
    }

    // Test data()
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < elements_per_inner_array; j++) {
        (array.data()[i])[j].set((i + 1) * elements_per_inner_array + j);
      }
    }
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < elements_per_inner_array; j++) {
        ASSERT_EQ((array[i])[j].get(), (i + 1) * elements_per_inner_array + j);
        ASSERT_EQ((array.data()[i])[j].get(),
                  (i + 1) * elements_per_inner_array + j);
      }
    }
  }  // Close scope containing 'array'.

  // Check that all constructed elements were destructed.
  EXPECT_EQ(ConstructionTester::constructions,
            ConstructionTester::destructions);
}

TEST(IteratorConstructorTest, NonInline) {
  int const kInput[] = {2, 3, 5, 7, 11, 13, 17};
  turbo::FixedArray<int, TURBO_ARRAY_SIZE(kInput) - 1> const fixed(
      kInput, kInput + TURBO_ARRAY_SIZE(kInput));
  ASSERT_EQ(TURBO_ARRAY_SIZE(kInput), fixed.size());
  for (size_t i = 0; i < TURBO_ARRAY_SIZE(kInput); ++i) {
    ASSERT_EQ(kInput[i], fixed[i]);
  }
}

TEST(IteratorConstructorTest, Inline) {
  int const kInput[] = {2, 3, 5, 7, 11, 13, 17};
  turbo::FixedArray<int, TURBO_ARRAY_SIZE(kInput)> const fixed(
      kInput, kInput + TURBO_ARRAY_SIZE(kInput));
  ASSERT_EQ(TURBO_ARRAY_SIZE(kInput), fixed.size());
  for (size_t i = 0; i < TURBO_ARRAY_SIZE(kInput); ++i) {
    ASSERT_EQ(kInput[i], fixed[i]);
  }
}

TEST(IteratorConstructorTest, NonPod) {
  char const* kInput[] = {"red",  "orange", "yellow", "green",
                          "blue", "indigo", "violet"};
  turbo::FixedArray<std::string> const fixed(kInput,
                                            kInput + TURBO_ARRAY_SIZE(kInput));
  ASSERT_EQ(TURBO_ARRAY_SIZE(kInput), fixed.size());
  for (size_t i = 0; i < TURBO_ARRAY_SIZE(kInput); ++i) {
    ASSERT_EQ(kInput[i], fixed[i]);
  }
}

TEST(IteratorConstructorTest, FromEmptyVector) {
  std::vector<int> const empty;
  turbo::FixedArray<int> const fixed(empty.begin(), empty.end());
  EXPECT_EQ(0, fixed.size());
  EXPECT_EQ(empty.size(), fixed.size());
}

TEST(IteratorConstructorTest, FromNonEmptyVector) {
  int const kInput[] = {2, 3, 5, 7, 11, 13, 17};
  std::vector<int> const items(kInput, kInput + TURBO_ARRAY_SIZE(kInput));
  turbo::FixedArray<int> const fixed(items.begin(), items.end());
  ASSERT_EQ(items.size(), fixed.size());
  for (size_t i = 0; i < items.size(); ++i) {
    ASSERT_EQ(items[i], fixed[i]);
  }
}

TEST(IteratorConstructorTest, FromBidirectionalIteratorRange) {
  int const kInput[] = {2, 3, 5, 7, 11, 13, 17};
  std::list<int> const items(kInput, kInput + TURBO_ARRAY_SIZE(kInput));
  turbo::FixedArray<int> const fixed(items.begin(), items.end());
  EXPECT_THAT(fixed, testing::ElementsAreArray(kInput));
}

TEST(InitListConstructorTest, InitListConstruction) {
  turbo::FixedArray<int> fixed = {1, 2, 3};
  EXPECT_THAT(fixed, testing::ElementsAreArray({1, 2, 3}));
}

TEST(FillConstructorTest, NonEmptyArrays) {
  turbo::FixedArray<int> stack_array(4, 1);
  EXPECT_THAT(stack_array, testing::ElementsAreArray({1, 1, 1, 1}));

  turbo::FixedArray<int, 0> heap_array(4, 1);
  EXPECT_THAT(stack_array, testing::ElementsAreArray({1, 1, 1, 1}));
}

TEST(FillConstructorTest, EmptyArray) {
  turbo::FixedArray<int> empty_fill(0, 1);
  turbo::FixedArray<int> empty_size(0);
  EXPECT_EQ(empty_fill, empty_size);
}

TEST(FillConstructorTest, NotTriviallyCopyable) {
  std::string str = "abcd";
  turbo::FixedArray<std::string> strings = {str, str, str, str};

  turbo::FixedArray<std::string> array(4, str);
  EXPECT_EQ(array, strings);
}

TEST(FillConstructorTest, Disambiguation) {
  turbo::FixedArray<size_t> a(1, 2);
  EXPECT_THAT(a, testing::ElementsAre(2));
}

TEST(FixedArrayTest, ManySizedArrays) {
  std::vector<int> sizes;
  for (int i = 1; i < 100; i++) sizes.push_back(i);
  for (int i = 100; i <= 1000; i += 100) sizes.push_back(i);
  for (int n : sizes) {
    TestArray<0>(n);
    TestArray<1>(n);
    TestArray<64>(n);
    TestArray<1000>(n);
  }
}

TEST(FixedArrayTest, ManySizedArraysOfArraysOf1) {
  for (int n = 1; n < 1000; n++) {
    ASSERT_NO_FATAL_FAILURE((TestArrayOfArrays<1, 0>(n)));
    ASSERT_NO_FATAL_FAILURE((TestArrayOfArrays<1, 1>(n)));
    ASSERT_NO_FATAL_FAILURE((TestArrayOfArrays<1, 64>(n)));
    ASSERT_NO_FATAL_FAILURE((TestArrayOfArrays<1, 1000>(n)));
  }
}

TEST(FixedArrayTest, ManySizedArraysOfArraysOf2) {
  for (int n = 1; n < 1000; n++) {
    TestArrayOfArrays<2, 0>(n);
    TestArrayOfArrays<2, 1>(n);
    TestArrayOfArrays<2, 64>(n);
    TestArrayOfArrays<2, 1000>(n);
  }
}

// If value_type is put inside of a struct container,
// we might evoke this error in a hardened build unless data() is carefully
// written, so check on that.
//     error: call to int __builtin___sprintf_chk(etc...)
//     will always overflow destination buffer [-Werror]
TEST(FixedArrayTest, AvoidParanoidDiagnostics) {
  turbo::FixedArray<char, 32> buf(32);
  sprintf(buf.data(), "foo");  // NOLINT(runtime/printf)
}

TEST(FixedArrayTest, TooBigInlinedSpace) {
  struct TooBig {
    char c[1 << 20];
  };  // too big for even one on the stack

  // Simulate the data members of turbo::FixedArray, a pointer and a size_t.
  struct Data {
    TooBig* p;
    size_t size;
  };

  // Make sure TooBig objects are not inlined for 0 or default size.
  static_assert(sizeof(turbo::FixedArray<TooBig, 0>) == sizeof(Data),
                "0-sized turbo::FixedArray should have same size as Data.");
  static_assert(alignof(turbo::FixedArray<TooBig, 0>) == alignof(Data),
                "0-sized turbo::FixedArray should have same alignment as Data.");
  static_assert(sizeof(turbo::FixedArray<TooBig>) == sizeof(Data),
                "default-sized turbo::FixedArray should have same size as Data");
  static_assert(
      alignof(turbo::FixedArray<TooBig>) == alignof(Data),
      "default-sized turbo::FixedArray should have same alignment as Data.");
}

// PickyDelete EXPECTs its class-scope deallocation funcs are unused.
struct PickyDelete {
  PickyDelete() {}
  ~PickyDelete() {}
  void operator delete(void* p) {
    EXPECT_TRUE(false) << __FUNCTION__;
    ::operator delete(p);
  }
  void operator delete[](void* p) {
    EXPECT_TRUE(false) << __FUNCTION__;
    ::operator delete[](p);
  }
};

TEST(FixedArrayTest, UsesGlobalAlloc) { turbo::FixedArray<PickyDelete, 0> a(5); }

TEST(FixedArrayTest, Data) {
  static const int kInput[] = {2, 3, 5, 7, 11, 13, 17};
  turbo::FixedArray<int> fa(std::begin(kInput), std::end(kInput));
  EXPECT_EQ(fa.data(), &*fa.begin());
  EXPECT_EQ(fa.data(), &fa[0]);

  const turbo::FixedArray<int>& cfa = fa;
  EXPECT_EQ(cfa.data(), &*cfa.begin());
  EXPECT_EQ(cfa.data(), &cfa[0]);
}

TEST(FixedArrayTest, Empty) {
  turbo::FixedArray<int> empty(0);
  turbo::FixedArray<int> inline_filled(1);
  turbo::FixedArray<int, 0> heap_filled(1);
  EXPECT_TRUE(empty.empty());
  EXPECT_FALSE(inline_filled.empty());
  EXPECT_FALSE(heap_filled.empty());
}

TEST(FixedArrayTest, FrontAndBack) {
  turbo::FixedArray<int, 3 * sizeof(int)> inlined = {1, 2, 3};
  EXPECT_EQ(inlined.front(), 1);
  EXPECT_EQ(inlined.back(), 3);

  turbo::FixedArray<int, 0> allocated = {1, 2, 3};
  EXPECT_EQ(allocated.front(), 1);
  EXPECT_EQ(allocated.back(), 3);

  turbo::FixedArray<int> one_element = {1};
  EXPECT_EQ(one_element.front(), one_element.back());
}

TEST(FixedArrayTest, ReverseIteratorInlined) {
  turbo::FixedArray<int, 5 * sizeof(int)> a = {0, 1, 2, 3, 4};

  int counter = 5;
  for (turbo::FixedArray<int>::reverse_iterator iter = a.rbegin();
       iter != a.rend(); ++iter) {
    counter--;
    EXPECT_EQ(counter, *iter);
  }
  EXPECT_EQ(counter, 0);

  counter = 5;
  for (turbo::FixedArray<int>::const_reverse_iterator iter = a.rbegin();
       iter != a.rend(); ++iter) {
    counter--;
    EXPECT_EQ(counter, *iter);
  }
  EXPECT_EQ(counter, 0);

  counter = 5;
  for (auto iter = a.crbegin(); iter != a.crend(); ++iter) {
    counter--;
    EXPECT_EQ(counter, *iter);
  }
  EXPECT_EQ(counter, 0);
}

TEST(FixedArrayTest, ReverseIteratorAllocated) {
  turbo::FixedArray<int, 0> a = {0, 1, 2, 3, 4};

  int counter = 5;
  for (turbo::FixedArray<int>::reverse_iterator iter = a.rbegin();
       iter != a.rend(); ++iter) {
    counter--;
    EXPECT_EQ(counter, *iter);
  }
  EXPECT_EQ(counter, 0);

  counter = 5;
  for (turbo::FixedArray<int>::const_reverse_iterator iter = a.rbegin();
       iter != a.rend(); ++iter) {
    counter--;
    EXPECT_EQ(counter, *iter);
  }
  EXPECT_EQ(counter, 0);

  counter = 5;
  for (auto iter = a.crbegin(); iter != a.crend(); ++iter) {
    counter--;
    EXPECT_EQ(counter, *iter);
  }
  EXPECT_EQ(counter, 0);
}

TEST(FixedArrayTest, Fill) {
  turbo::FixedArray<int, 5 * sizeof(int)> inlined(5);
  int fill_val = 42;
  inlined.fill(fill_val);
  for (int i : inlined) EXPECT_EQ(i, fill_val);

  turbo::FixedArray<int, 0> allocated(5);
  allocated.fill(fill_val);
  for (int i : allocated) EXPECT_EQ(i, fill_val);

  // It doesn't do anything, just make sure this compiles.
  turbo::FixedArray<int> empty(0);
  empty.fill(fill_val);
}

#ifndef __GNUC__
TEST(FixedArrayTest, DefaultCtorDoesNotValueInit) {
  using T = char;
  constexpr auto capacity = 10;
  using FixedArrType = turbo::FixedArray<T, capacity>;
  constexpr auto scrubbed_bits = 0x95;
  constexpr auto length = capacity / 2;

  alignas(FixedArrType) unsigned char buff[sizeof(FixedArrType)];
  std::memset(std::addressof(buff), scrubbed_bits, sizeof(FixedArrType));

  FixedArrType* arr =
      ::new (static_cast<void*>(std::addressof(buff))) FixedArrType(length);
  EXPECT_THAT(*arr, testing::Each(scrubbed_bits));
  arr->~FixedArrType();
}
#endif  // __GNUC__

TEST(AllocatorSupportTest, CountInlineAllocations) {
  constexpr size_t inlined_size = 4;
  using Alloc = turbo::container_internal::CountingAllocator<int>;
  using AllocFxdArr = turbo::FixedArray<int, inlined_size, Alloc>;

  int64_t allocated = 0;
  int64_t active_instances = 0;

  {
    const int ia[] = {0, 1, 2, 3, 4, 5, 6, 7};

    Alloc alloc(&allocated, &active_instances);

    AllocFxdArr arr(ia, ia + inlined_size, alloc);
    static_cast<void>(arr);
  }

  EXPECT_EQ(allocated, 0);
  EXPECT_EQ(active_instances, 0);
}

TEST(AllocatorSupportTest, CountOutoflineAllocations) {
  constexpr size_t inlined_size = 4;
  using Alloc = turbo::container_internal::CountingAllocator<int>;
  using AllocFxdArr = turbo::FixedArray<int, inlined_size, Alloc>;

  int64_t allocated = 0;
  int64_t active_instances = 0;

  {
    const int ia[] = {0, 1, 2, 3, 4, 5, 6, 7};
    Alloc alloc(&allocated, &active_instances);

    AllocFxdArr arr(ia, ia + TURBO_ARRAY_SIZE(ia), alloc);

    EXPECT_EQ(allocated, arr.size() * sizeof(int));
    static_cast<void>(arr);
  }

  EXPECT_EQ(active_instances, 0);
}

TEST(AllocatorSupportTest, CountCopyInlineAllocations) {
  constexpr size_t inlined_size = 4;
  using Alloc = turbo::container_internal::CountingAllocator<int>;
  using AllocFxdArr = turbo::FixedArray<int, inlined_size, Alloc>;

  int64_t allocated1 = 0;
  int64_t allocated2 = 0;
  int64_t active_instances = 0;
  Alloc alloc(&allocated1, &active_instances);
  Alloc alloc2(&allocated2, &active_instances);

  {
    int initial_value = 1;

    AllocFxdArr arr1(inlined_size / 2, initial_value, alloc);

    EXPECT_EQ(allocated1, 0);

    AllocFxdArr arr2(arr1, alloc2);

    EXPECT_EQ(allocated2, 0);
    static_cast<void>(arr1);
    static_cast<void>(arr2);
  }

  EXPECT_EQ(active_instances, 0);
}

TEST(AllocatorSupportTest, CountCopyOutoflineAllocations) {
  constexpr size_t inlined_size = 4;
  using Alloc = turbo::container_internal::CountingAllocator<int>;
  using AllocFxdArr = turbo::FixedArray<int, inlined_size, Alloc>;

  int64_t allocated1 = 0;
  int64_t allocated2 = 0;
  int64_t active_instances = 0;
  Alloc alloc(&allocated1, &active_instances);
  Alloc alloc2(&allocated2, &active_instances);

  {
    int initial_value = 1;

    AllocFxdArr arr1(inlined_size * 2, initial_value, alloc);

    EXPECT_EQ(allocated1, arr1.size() * sizeof(int));

    AllocFxdArr arr2(arr1, alloc2);

    EXPECT_EQ(allocated2, inlined_size * 2 * sizeof(int));
    static_cast<void>(arr1);
    static_cast<void>(arr2);
  }

  EXPECT_EQ(active_instances, 0);
}

TEST(AllocatorSupportTest, SizeValAllocConstructor) {
  using testing::AllOf;
  using testing::Each;
  using testing::SizeIs;

  constexpr size_t inlined_size = 4;
  using Alloc = turbo::container_internal::CountingAllocator<int>;
  using AllocFxdArr = turbo::FixedArray<int, inlined_size, Alloc>;

  {
    auto len = inlined_size / 2;
    auto val = 0;
    int64_t allocated = 0;
    AllocFxdArr arr(len, val, Alloc(&allocated));

    EXPECT_EQ(allocated, 0);
    EXPECT_THAT(arr, AllOf(SizeIs(len), Each(0)));
  }

  {
    auto len = inlined_size * 2;
    auto val = 0;
    int64_t allocated = 0;
    AllocFxdArr arr(len, val, Alloc(&allocated));

    EXPECT_EQ(allocated, len * sizeof(int));
    EXPECT_THAT(arr, AllOf(SizeIs(len), Each(0)));
  }
}

#ifdef TURBO_HAVE_ADDRESS_SANITIZER
TEST(FixedArrayTest, AddressSanitizerAnnotations1) {
  turbo::FixedArray<int, 32> a(10);
  int* raw = a.data();
  raw[0] = 0;
  raw[9] = 0;
  EXPECT_DEATH_IF_SUPPORTED(raw[-2] = 0, "container-overflow");
  EXPECT_DEATH_IF_SUPPORTED(raw[-1] = 0, "container-overflow");
  EXPECT_DEATH_IF_SUPPORTED(raw[10] = 0, "container-overflow");
  EXPECT_DEATH_IF_SUPPORTED(raw[31] = 0, "container-overflow");
}

TEST(FixedArrayTest, AddressSanitizerAnnotations2) {
  turbo::FixedArray<char, 17> a(12);
  char* raw = a.data();
  raw[0] = 0;
  raw[11] = 0;
  EXPECT_DEATH_IF_SUPPORTED(raw[-7] = 0, "container-overflow");
  EXPECT_DEATH_IF_SUPPORTED(raw[-1] = 0, "container-overflow");
  EXPECT_DEATH_IF_SUPPORTED(raw[12] = 0, "container-overflow");
  EXPECT_DEATH_IF_SUPPORTED(raw[17] = 0, "container-overflow");
}

TEST(FixedArrayTest, AddressSanitizerAnnotations3) {
  turbo::FixedArray<uint64_t, 20> a(20);
  uint64_t* raw = a.data();
  raw[0] = 0;
  raw[19] = 0;
  EXPECT_DEATH_IF_SUPPORTED(raw[-1] = 0, "container-overflow");
  EXPECT_DEATH_IF_SUPPORTED(raw[20] = 0, "container-overflow");
}

TEST(FixedArrayTest, AddressSanitizerAnnotations4) {
  turbo::FixedArray<ThreeInts> a(10);
  ThreeInts* raw = a.data();
  raw[0] = ThreeInts();
  raw[9] = ThreeInts();
  // Note: raw[-1] is pointing to 12 bytes before the container range. However,
  // there is only a 8-byte red zone before the container range, so we only
  // access the last 4 bytes of the struct to make sure it stays within the red
  // zone.
  EXPECT_DEATH_IF_SUPPORTED(raw[-1].z_ = 0, "container-overflow");
  EXPECT_DEATH_IF_SUPPORTED(raw[10] = ThreeInts(), "container-overflow");
  // The actual size of storage is kDefaultBytes=256, 21*12 = 252,
  // so reading raw[21] should still trigger the correct warning.
  EXPECT_DEATH_IF_SUPPORTED(raw[21] = ThreeInts(), "container-overflow");
}
#endif  // TURBO_HAVE_ADDRESS_SANITIZER

TEST(FixedArrayTest, TurboHashValueWorks) {
  using V = turbo::FixedArray<int>;
  std::vector<V> cases;

  // Generate a variety of vectors some of these are small enough for the inline
  // space but are stored out of line.
  for (int i = 0; i < 10; ++i) {
    V v(i);
    for (int j = 0; j < i; ++j) {
      v[j] = j;
    }
    cases.push_back(v);
  }

  EXPECT_TRUE(turbo::VerifyTypeImplementsTurboHashCorrectly(cases));
}

}  // namespace
