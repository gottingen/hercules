// Copyright 2023 The titan-search Authors.
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
#ifndef TURBO_FLAGS_INTERNAL_PARSE_H_
#define TURBO_FLAGS_INTERNAL_PARSE_H_

#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <string_view>

#include "turbo/platform/port.h"
#include "turbo/flags/declare.h"
#include "turbo/flags/internal/usage.h"

TURBO_DECLARE_FLAG(std::vector<std::string>, flagfile);
TURBO_DECLARE_FLAG(std::vector<std::string>, fromenv);
TURBO_DECLARE_FLAG(std::vector<std::string>, tryfromenv);
TURBO_DECLARE_FLAG(std::vector<std::string>, undefok);

namespace turbo::flags_internal {

    enum class UsageFlagsAction {
        kHandleUsage, kIgnoreUsage
    };
    enum class OnUndefinedFlag {
        kIgnoreUndefined,
        kReportUndefined,
        kAbortIfUndefined
    };

    // This is not a public interface. This interface exists to expose the ability
    // to change help output stream in case of parsing errors. This is used by
    // internal unit tests to validate expected outputs.
    // When this was written, `EXPECT_EXIT` only supported matchers on stderr,
    // but not on stdout.
    std::vector<char *> ParseCommandLineImpl(
            int argc, char *argv[], UsageFlagsAction usage_flag_action,
            OnUndefinedFlag undef_flag_action,
            std::ostream &error_help_output = std::cout);

    // --------------------------------------------------------------------
    // Inspect original command line

    // Returns true if flag with specified name was either present on the original
    // command line or specified in flag file present on the original command line.
    bool WasPresentOnCommandLine(std::string_view flag_name);

    // Return existing flags similar to the parameter, in order to help in case of
    // misspellings.
    std::vector<std::string> GetMisspellingHints(std::string_view flag);

}  // namespace turbo::flags_internal

#endif  // TURBO_FLAGS_INTERNAL_PARSE_H_
