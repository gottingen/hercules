// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <collie/cli/macros.h>

namespace collie::detail {

    // Returns false if not a short option. Otherwise, sets opt name and rest and returns true
    inline bool split_short(const std::string &current, std::string &name, std::string &rest);

    // Returns false if not a long option. Otherwise, sets opt name and other side of = and returns true
    inline bool split_long(const std::string &current, std::string &name, std::string &value);

    // Returns false if not a windows style option. Otherwise, sets opt name and value and returns true
    inline bool split_windows_style(const std::string &current, std::string &name, std::string &value);

    // Splits a string into multiple long and short names
    inline std::vector<std::string> split_names(std::string current);

    /// extract default flag values either {def} or starting with a !
    inline std::vector<std::pair<std::string, std::string>> get_default_flag_values(const std::string &str);

    /// Get a vector of short names, one of long names, and a single name
    inline std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>
    get_names(const std::vector<std::string> &input);

}  // namespace collie::detail

#include <collie/cli/impl/split_inl.h>
