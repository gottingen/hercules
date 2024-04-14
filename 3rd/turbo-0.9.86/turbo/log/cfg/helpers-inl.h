// Copyright 2023 The titan-search Authors.
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

#pragma once

#include <turbo/log/cfg/helpers.h>

#include <turbo/log/tlog.h>
#include "turbo/log/details/os.h"
#include "turbo/log/details/registry.h"

#include <algorithm>
#include <string>
#include <utility>
#include <sstream>

namespace turbo::tlog {
    namespace cfg {
        namespace helpers {

// inplace convert to lowercase
            inline std::string &to_lower_(std::string &str) {
                std::transform(
                        str.begin(), str.end(), str.begin(),
                        [](char ch) { return static_cast<char>((ch >= 'A' && ch <= 'Z') ? ch + ('a' - 'A') : ch); });
                return str;
            }

// inplace trim spaces
            inline std::string &trim_(std::string &str) {
                const char *spaces = " \n\r\t";
                str.erase(str.find_last_not_of(spaces) + 1);
                str.erase(0, str.find_first_not_of(spaces));
                return str;
            }

            // return (name,value) trimmed pair from given "name=value" string.
            // return empty string on missing parts
            // "key=val" => ("key", "val")
            // " key  =  val " => ("key", "val")
            // "key=" => ("key", "")
            // "val" => ("", "val")

            inline std::pair<std::string, std::string> extract_kv_(char sep, const std::string &str) {
                auto n = str.find(sep);
                std::string k, v;
                if (n == std::string::npos) {
                    v = str;
                } else {
                    k = str.substr(0, n);
                    v = str.substr(n + 1);
                }
                return std::make_pair(trim_(k), trim_(v));
            }

            // return vector of key/value pairs from sequence of "K1=V1,K2=V2,.."
            // "a=AAA,b=BBB,c=CCC,.." => {("a","AAA"),("b","BBB"),("c", "CCC"),...}
            inline std::unordered_map<std::string, std::string> extract_key_vals_(const std::string &str) {
                std::string token;
                std::istringstream token_stream(str);
                std::unordered_map<std::string, std::string> rv{};
                while (std::getline(token_stream, token, ',')) {
                    if (token.empty()) {
                        continue;
                    }
                    auto kv = extract_kv_('=', token);
                    rv[kv.first] = kv.second;
                }
                return rv;
            }

            void load_levels(const std::string &input) {
                if (input.empty() || input.size() > 512) {
                    return;
                }

                auto key_vals = extract_key_vals_(input);
                std::unordered_map<std::string, level::level_enum> levels;
                level::level_enum global_level = level::info;
                bool global_level_found = false;

                for (auto &name_level: key_vals) {
                    auto &logger_name = name_level.first;
                    auto level_name = to_lower_(name_level.second);
                    auto level = level::from_str(level_name);
                    // ignore unrecognized level names
                    if (level == level::off && level_name != "off") {
                        continue;
                    }
                    if (logger_name.empty()) // no logger name indicate global level
                    {
                        global_level_found = true;
                        global_level = level;
                    } else {
                        levels[logger_name] = level;
                    }
                }

                details::registry::instance().set_levels(std::move(levels),
                                                         global_level_found ? &global_level : nullptr);
            }

        } // namespace helpers
    } // namespace cfg
} // namespace turbo::tlog
