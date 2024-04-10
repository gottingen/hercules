// Copyright 2024 The EA Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#include "testing.h"

using namespace hercules;
using namespace std;


INSTANTIATE_TEST_SUITE_P(
        CoreTests, HerculesTest,
        testing::Combine(
                testing::Values(
                        "core/helloworld.hs",
                        "core/arithmetic.hs",
                        "core/parser.hs",
                        "core/generics.hs",
                        "core/generators.hs",
                        "core/exceptions.hs",
                        "core/containers.hs",
                        "core/trees.hs",
                        "core/range.hs",
                        "core/bltin.hs",
                        "core/arguments.hs",
                        "core/match.hs",
                        "core/serialization.hs",
                        "core/empty.hs"
                ),
                testing::Values(true, false),
                testing::Values(""),
                testing::Values(""),
                testing::Values(0),
                testing::Values(false),
                testing::Values(false)
        ),
        getTestNameFromParam);
// clang-format on

int main(int argc, char *argv[]) {
    argv0 = ast::executable_path(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
