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
#define TURBO_OPTION_HARDENED 0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"
#include "turbo/container/batch.h"


TEST_CASE("batch printing") {

    turbo::println("{}", std::is_convertible_v<int, int64_t>);
    turbo::println("{}", std::is_convertible_v<int, float>);
    turbo::Batch<int, 4> batch(10);
    turbo::println("{}", batch);
}

TEST_CASE("batch initialization") {

    turbo::Batch<int, 4> batch({1u,2u,3u,4u});
    REQUIRE_EQ(batch.size(), 4);
    REQUIRE_EQ(batch[0], 1);
    REQUIRE_EQ(batch[1], 2);
    REQUIRE_EQ(batch[2], 3);
    REQUIRE_EQ(batch[3], 4);
    turbo::println("{}", batch);

    turbo::Batch<int, 4> batch2({1.0,2.0,3.0,4.0});
    REQUIRE_EQ(batch2.size(), 4);
    REQUIRE_EQ(batch2[0], 1);
    REQUIRE_EQ(batch2[1], 2);
    REQUIRE_EQ(batch2[2], 3);
    REQUIRE_EQ(batch2[3], 4);
    turbo::println("{}", batch2);

    turbo::Batch<int, 4> batch3(std::vector<float>{1.0,2.0,3.0,4.0});
    REQUIRE_EQ(batch3.size(), 4);
    REQUIRE_EQ(batch3[0], 1);
    REQUIRE_EQ(batch3[1], 2);
    REQUIRE_EQ(batch3[2], 3);
    REQUIRE_EQ(batch3[3], 4);
    turbo::println("{}", batch3);

    turbo::Batch<int, 4> batch4 = {1.0,2.0,3.0,4.0};
    REQUIRE_EQ(batch4.size(), 4);
    REQUIRE_EQ(batch4[0], 1);
    REQUIRE_EQ(batch4[1], 2);
    REQUIRE_EQ(batch4[2], 3);
    REQUIRE_EQ(batch4[3], 4);
    turbo::println("{}", batch4);

    turbo::Batch<int, 4> batch5 = {1,2,3,4};
    batch5 *= 2;
    REQUIRE_EQ(batch5.size(), 4);
    REQUIRE_EQ(batch5[0], 2);
    REQUIRE_EQ(batch5[1], 4);
    REQUIRE_EQ(batch5[2], 6);
    REQUIRE_EQ(batch5[3], 8);
    turbo::println("{}", batch5);

    batch5 += 1;
    REQUIRE_EQ(batch5.size(), 4);
    REQUIRE_EQ(batch5[0], 3);
    REQUIRE_EQ(batch5[1], 5);
    REQUIRE_EQ(batch5[2], 7);
    REQUIRE_EQ(batch5[3], 9);
    turbo::println("{}", batch5);

    batch5[0] = 42;
    REQUIRE_EQ(batch5.size(), 4);
    REQUIRE_EQ(batch5[0], 42);
    REQUIRE_EQ(batch5[1], 5);
    REQUIRE_EQ(batch5[2], 7);
    REQUIRE_EQ(batch5[3], 9);


}
