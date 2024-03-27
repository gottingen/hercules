// Copyright 2024 The Elastic AI Search Authors.
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

#include <collie/module/module.h>

using namespace collie;

int main() {
    constexpr std::string_view r1 = ">=1.2.7 <1.3.0";
    static_assert(module_range::satisfies("1.2.7"_version, r1));
    static_assert(module_range::satisfies("1.2.8"_version, r1));
    static_assert(module_range::satisfies("1.2.99"_version, r1));
    static_assert(!module_range::satisfies("1.2.6"_version, r1));
    static_assert(!module_range::satisfies("1.3.0"_version, r1));
    static_assert(!module_range::satisfies("1.1.0"_version, r1));

    constexpr std::string_view r2 = "1.2.7 || >=1.2.9 <2.0.0";
    static_assert(module_range::satisfies(ModuleVersion{1, 2, 7}, r2));
    static_assert(module_range::satisfies({1, 2, 9}, r2));
    static_assert(!module_range::satisfies("1.2.8"_version, r2));
    static_assert(!module_range::satisfies("2.0.0"_version, r2));

    // By default, we exclude prerelease tag from comparison.
    constexpr std::string_view r3 = ">1.2.3-alpha.3";
    static_assert(module_range::satisfies("1.2.3-alpha.7"_version, r3));
    static_assert(!module_range::satisfies("3.4.5-alpha.9"_version, r3));

    static_assert(module_range::satisfies("3.4.5-alpha.9"_version, r3, module_range::satisfies_option::include_prerelease));

    return 0;
}
