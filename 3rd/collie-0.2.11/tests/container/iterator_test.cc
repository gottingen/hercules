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
#include <collie/container/iterator.h>

// --------------------------------------------------------
// Testcase: distance
// --------------------------------------------------------
TEST_CASE("distance.integral" * doctest::timeout(300)) {

    auto count = [](int beg, int end, int step) {
        size_t c = 0;
        for (int i = beg; step > 0 ? i < end : i > end; i += step) {
            ++c;
        }
        return c;
    };

    for (int beg = -50; beg <= 50; ++beg) {
        for (int end = -50; end <= 50; ++end) {
            if (beg < end) {   // positive step
                for (int s = 1; s <= 50; s++) {
                    REQUIRE((collie::distance(beg, end, s) == count(beg, end, s)));
                }
            } else {            // negative step
                for (int s = -1; s >= -50; s--) {
                    REQUIRE((collie::distance(beg, end, s) == count(beg, end, s)));
                }
            }
        }
    }

}