// Copyright 2020 The Turbo Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "turbo/base/internal/fast_type_id.h"

#include <cstdint>
#include <map>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

namespace {
    namespace bi = turbo::base_internal;

// NOLINTNEXTLINE
#define PRIM_TYPES(A)   \
  A(bool)               \
  A(short)              \
  A(unsigned short)     \
  A(int)                \
  A(unsigned int)       \
  A(long)               \
  A(unsigned long)      \
  A(long long)          \
  A(unsigned long long) \
  A(float)              \
  A(double)             \
  A(long double)

    TEST_CASE("FastTypeIdTest, PrimitiveTypes") {
        bi::FastTypeIdType type_ids[] = {
#define A(T) bi::FastTypeId<T>(),
                PRIM_TYPES(A)
#undef A
#define A(T) bi::FastTypeId<const T>(),
                PRIM_TYPES(A)
#undef A
#define A(T) bi::FastTypeId<volatile T>(),
                PRIM_TYPES(A)
#undef A
#define A(T) bi::FastTypeId<const volatile T>(),
                PRIM_TYPES(A)
#undef A
        };
        size_t total_type_ids = sizeof(type_ids) / sizeof(bi::FastTypeIdType);

        for (int i = 0; i < total_type_ids; ++i) {
            CHECK_EQ(type_ids[i], type_ids[i]);
            for (int j = 0; j < i; ++j) {
                CHECK_NE(type_ids[i], type_ids[j]);
            }
        }
    }

#define FIXED_WIDTH_TYPES(A) \
  A(int8_t)                  \
  A(uint8_t)                 \
  A(int16_t)                 \
  A(uint16_t)                \
  A(int32_t)                 \
  A(uint32_t)                \
  A(int64_t)                 \
  A(uint64_t)

    TEST_CASE("FastTypeIdTest, FixedWidthTypes") {
        bi::FastTypeIdType type_ids[] = {
#define A(T) bi::FastTypeId<T>(),
                FIXED_WIDTH_TYPES(A)
#undef A
#define A(T) bi::FastTypeId<const T>(),
                FIXED_WIDTH_TYPES(A)
#undef A
#define A(T) bi::FastTypeId<volatile T>(),
                FIXED_WIDTH_TYPES(A)
#undef A
#define A(T) bi::FastTypeId<const volatile T>(),
                FIXED_WIDTH_TYPES(A)
#undef A
        };
        size_t total_type_ids = sizeof(type_ids) / sizeof(bi::FastTypeIdType);

        for (int i = 0; i < total_type_ids; ++i) {
            CHECK_EQ(type_ids[i], type_ids[i]);
            for (int j = 0; j < i; ++j) {
                CHECK_NE(type_ids[i], type_ids[j]);
            }
        }
    }

    TEST_CASE("FastTypeIdTest, AliasTypes") {
        using int_alias = int;
        CHECK_EQ(bi::FastTypeId<int_alias>(), bi::FastTypeId<int>());
    }

    TEST_CASE("FastTypeIdTest, TemplateSpecializations") {
        CHECK_NE(bi::FastTypeId<std::vector<int>>(),
                  bi::FastTypeId<std::vector<long>>());

        CHECK_NE((bi::FastTypeId<std::map<int, float>>()),
                  (bi::FastTypeId<std::map<int, double>>()));
    }

    struct Base {
    };
    struct Derived : Base {
    };
    struct PDerived : private Base {
    };

    TEST_CASE("FastTypeIdTest, Inheritance") {
        CHECK_NE(bi::FastTypeId<Base>(), bi::FastTypeId<Derived>());
        CHECK_NE(bi::FastTypeId<Base>(), bi::FastTypeId<PDerived>());
    }

}  // namespace
