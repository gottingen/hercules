// Copyright 2023 The titan-search Authors.
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
//
#include "turbo/files/system.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

namespace turbo {
    TEST_CASE("system, executable_path")
    {
        std::string exec_path = executable_path();
        CHECK(!exec_path.empty());
    }

    TEST_CASE("system, prefix_path")
    {
        std::string prefix = prefix_path();
        std::string exec_path = executable_path();

        CHECK_NE(prefix.size(), exec_path.size());
        CHECK(std::equal(prefix.cbegin(), prefix.cend(), exec_path.cbegin()));
        CHECK(((exec_path.find("test_turbo") != std::string::npos) ||
                    (exec_path.find("system_test") != std::string::npos)));
    }
}
