// Copyright 2024 The Elastic-AI Authors.
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


#ifndef COLLIE_STRINGS_CAT_H_
#define COLLIE_STRINGS_CAT_H_

#include <string>
#include <vector>
#include <string_view>
#include <collie/strings/fmt/format.h>

namespace collie {

    template <typename T>
    std::string str_cat(const T &t) {
        return fmt::format("{}", t);
    }

    template <typename T, typename... Args>
    std::string str_cat(const T &t, const Args &... args) {
        return fmt::format("{}{}", t, str_cat(args...));
    }

    template <typename... Args>
    std::string str_cat(const std::vector<std::string> &vec, const Args &... args) {
        std::string result;
        for (const auto &s : vec) {
            result += s;
        }
        return str_cat(result, args...);
    }

    template <typename... Args>
    std::string str_cat(const std::vector<std::string_view> &vec, const Args &... args) {
        std::string result;
        for (const auto &s : vec) {
            result += s;
        }
        return str_cat(result, args...);
    }

    template <typename T>
    std::string &str_cat_append(std::string &result, const T &t) {
        result += fmt::format("{}", t);
        return result;
    }

    template <typename T, typename... Args>
    std::string &str_cat_append(std::string &result, const T &t, const Args &... args) {
        result += fmt::format("{}{}", t, str_cat(args...));
        return result;
    }

    template <typename... Args>
    std::string &str_cat_append(std::string &result, const std::vector<std::string> &vec, const Args &... args) {
        for (const auto &s : vec) {
            result += s;
        }
        return str_cat(result, args...);
    }

    template <typename... Args>
    std::string &str_cat_append(std::string &result, const std::vector<std::string_view> &vec, const Args &... args) {
        for (const auto &s : vec) {
            result += s;
        }
        return str_cat(result, args...);
    }
}  // namespace collie

#endif  // COLLIE_STRINGS_CAT_H_
