// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <collie/cli/app.h>
#include <collie/cli/config_fwd.h>
#include <collie/cli/string_tools.h>

namespace collie::detail {

    std::string convert_arg_for_ini(const std::string &arg,
                                    char stringQuote = '"',
                                    char literalQuote = '\'',
                                    bool disable_multi_line = false);

    /// Comma separated join, adds quotes if needed
    std::string ini_join(const std::vector<std::string> &args,
                         char sepChar = ',',
                         char arrayStart = '[',
                         char arrayEnd = ']',
                         char stringQuote = '"',
                         char literalQuote = '\'');

    void clean_name_string(std::string &name, const std::string &keyChars);

    std::vector<std::string> generate_parents(const std::string &section, std::string &name, char parentSeparator);

    /// assuming non default segments do a check on the close and open of the segments in a configItem structure
    void checkParentSegments(std::vector<ConfigItem> &output, const std::string &currentSection, char parentSeparator);
}  // namespace collie::detail

#include <collie/cli/impl/config_inl.h>

