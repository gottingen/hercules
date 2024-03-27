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

#include <hercules/runtime/hs_str.h>
#include <hercules/runtime/lib.h>
#include <collie/filesystem/fs.h>
#include <string>
#include <string_view>
#include <vector>

#if __WIN32
std::string_view lib_ext = ".dll";
#elif __linux__
std::string_view lib_ext = ".so";
#elif __APPLE__
std::string_view lib_ext = ".dylib";
#endif

HS_FUNC hs_str_t echo_hello() {
    return hs_string_conv("Hello, world!");
}

HS_FUNC hs_str_t say_hello(hs_str_t world) {
    return hs_string_conv("Hello, " + std::string(world.str, world.len) + "!");
}

HS_FUNC hs_str_t find_library(hs_str_t name) {
    std::vector<std::string> path_prefixes = {
            "/usr/local/lib",
            "/usr/lib",
            "/lib"
    };

    auto lib_name = "lib" + std::string(name.str, name.len);
    lib_name += lib_ext;
    for (auto &prefix: path_prefixes) {
        auto path = collie::filesystem::path(prefix) / lib_name;
        if (collie::filesystem::exists(path)) {
            return hs_string_conv(path.string());
        }
    }
    return hs_string_conv("");
}

HS_FUNC hs_str_t find_include(hs_str_t name) {
    std::vector<std::string> path_prefixes = {
            "/usr/local/include",
            "/usr/include",
            "/include"
    };

    auto inc_name = std::string(name.str, name.len);
    for (auto &prefix: path_prefixes) {
        auto path = collie::filesystem::path(prefix) / inc_name;
        if (collie::filesystem::exists(path)) {
            return hs_string_conv(path.string());
        }
    }
    return hs_string_conv("");
}