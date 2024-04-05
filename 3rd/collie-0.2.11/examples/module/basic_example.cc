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

#include <iostream>

using namespace collie;

int main() {
    constexpr ModuleVersion v_default;
    static_assert(v_default == ModuleVersion(0, 1, 0, prerelease::none, 0));
    std::cout << v_default << std::endl; // 0.1.0

    constexpr ModuleVersion v1{1, 4, 3};
    constexpr ModuleVersion v2{"1.2.4-alpha.10"};
    std::cout << v1 << std::endl; // 1.4.3
    std::cout << v2 << std::endl; // 1.2.4-alpha.10
    static_assert(v1 != v2);
    static_assert(!(v1 == v2));
    static_assert(v1 > v2);
    static_assert(v1 >= v2);
    static_assert(!(v1 < v2));
    static_assert(!(v1 <= v2));

    ModuleVersion v_s;
    v_s.from_string("1.2.3-rc.1");
    std::string s1 = v_s.to_string();
    std::cout << s1 << std::endl; // 1.2.3-rc.1
    v_s.prerelease_number = 0;
    std::string s2 = v_s.to_string();
    std::cout << s2 << std::endl; // 1.2.3-rc

    constexpr ModuleVersion vo = "2.0.0-rc.3"_version;
    std::cout << vo << std::endl; // 2.0.0-rc.3

    return 0;
}
