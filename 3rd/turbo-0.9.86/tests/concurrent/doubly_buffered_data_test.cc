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

#include "turbo/container/doubly_buffered_data.h"
#include <map>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

TEST_CASE("doubly_data") {

    typedef turbo::DoublyBufferedData<std::map<std::string, std::string>> double_map;
    double_map dmap;
    auto f = [](std::map<std::string, std::string> &m) {
        m["hello"] = "world";
        return 1;
    };
    dmap.Modify(f);
    double_map::ScopedPtr ptr;
    dmap.Read(&ptr);
    REQUIRE_EQ(ptr.get()->find("hello")->second, "world");
}