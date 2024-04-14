// Copyright 2023 The Turbo Authors.
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

#ifndef TURBO_FLAGS_SPLIT_H_
#define TURBO_FLAGS_SPLIT_H_

#include <string>
#include <tuple>
#include <utility>
#include <vector>


#include "turbo/platform/port.h"

namespace turbo::detail {


    // Returns false if not a short option. Otherwise, sets opt name and rest and returns true
    bool split_short(const std::string &current, std::string &name, std::string &rest);

    // Returns false if not a long option. Otherwise, sets opt name and other side of = and returns true
    bool split_long(const std::string &current, std::string &name, std::string &value);

    // Returns false if not a windows style option. Otherwise, sets opt name and value and returns true
    bool split_windows_style(const std::string &current, std::string &name, std::string &value);

    // Splits a string into multiple long and short names
    std::vector<std::string> split_names(std::string current);

    /// extract default flag values either {def} or starting with a !
    std::vector<std::pair<std::string, std::string>> get_default_flag_values(const std::string &str);

    /// Get a vector of short names, one of long names, and a single name
    std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>
    get_names(const std::vector<std::string> &input);

}  // namespace turbo::detail
#endif  // TURBO_FLAGS_SPLIT_H_
