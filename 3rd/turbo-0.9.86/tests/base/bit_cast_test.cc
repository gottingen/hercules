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

// Unit test for bit_cast template.

#include <cstdint>
#include <cstring>

#include "turbo/base/casts.h"
#include "turbo/platform/port.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

namespace turbo {

    namespace {

        template<int N>
        struct marshall {
            char buf[N];
        };

        template<typename T>
        void TestMarshall(const T values[], int num_values) {
            for (int i = 0; i < num_values; ++i) {
                T t0 = values[i];
                marshall<sizeof(T)> m0 = turbo::bit_cast<marshall<sizeof(T)> >(t0);
                T t1 = turbo::bit_cast<T>(m0);
                marshall<sizeof(T)> m1 = turbo::bit_cast<marshall<sizeof(T)> >(t1);
                CHECK_EQ(0, memcmp(&t0, &t1, sizeof(T)));
                CHECK_EQ(0, memcmp(&m0, &m1, sizeof(T)));
            }
        }

        // Convert back and forth to an integral type.  The C++ standard does
        // not guarantee this will work, but we test that this works on all the
        // platforms we support.
        //
        // Likewise, we below make assumptions about sizeof(float) and
        // sizeof(double) which the standard does not guarantee, but which hold on the
        // platforms we support.

        template<typename T, typename I>
        void TestIntegral(const T values[], int num_values) {
            for (int i = 0; i < num_values; ++i) {
                T t0 = values[i];
                I i0 = turbo::bit_cast<I>(t0);
                T t1 = turbo::bit_cast<T>(i0);
                I i1 = turbo::bit_cast<I>(t1);
                CHECK_EQ(0, memcmp(&t0, &t1, sizeof(T)));
                CHECK_EQ(i0, i1);
            }
        }

        TEST_CASE("BitCast") {
            SUBCASE("Bool") {
                static const bool bool_list[] = {false, true};
                TestMarshall<bool>(bool_list, TURBO_ARRAY_SIZE(bool_list));
            }

            SUBCASE("Int32") {
                static const int32_t int_list[] =
                        {0, 1, 100, 2147483647, -1, -100, -2147483647, -2147483647 - 1};
                TestMarshall<int32_t>(int_list, TURBO_ARRAY_SIZE(int_list));
            }

            SUBCASE("Int64") {
                static const int64_t int64_list[] =
                        {0, 1, 1LL << 40, -1, -(1LL << 40)};
                TestMarshall<int64_t>(int64_list, TURBO_ARRAY_SIZE(int64_list));
            }

            SUBCASE("Uint64") {
                static const uint64_t uint64_list[] =
                        {0, 1, 1LLU << 40, 1LLU << 63};
                TestMarshall<uint64_t>(uint64_list, TURBO_ARRAY_SIZE(uint64_list));
            }

            SUBCASE("Float") {
                static const float float_list[] =
                        {
                                0.0f, 1.0f, -1.0f, 10.0f, -10.0f,
                                1e10f, 1e20f, 1e-10f, 1e-20f,
                                2.71828f, 3.14159f
                        };
                TestMarshall<float>(float_list, TURBO_ARRAY_SIZE(float_list));
                TestIntegral<float, int>(float_list, TURBO_ARRAY_SIZE(float_list));
                TestIntegral<float, unsigned>(float_list, TURBO_ARRAY_SIZE(float_list));
            }

            SUBCASE("Double") {
                static const double double_list[] =
                        {
                                0.0, 1.0, -1.0, 10.0, -10.0,
                                1e10, 1e100, 1e-10, 1e-100,
                                2.718281828459045,
                                3.141592653589793238462643383279502884197169399375105820974944
                        };
                TestMarshall<double>(double_list, TURBO_ARRAY_SIZE(double_list));
                TestIntegral<double, int64_t>(double_list, TURBO_ARRAY_SIZE(double_list));
                TestIntegral<double, uint64_t>(double_list, TURBO_ARRAY_SIZE(double_list));
            }
        }
    }  // namespace

}  // namespace turbo
